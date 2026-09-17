# Accelerate vLLM model loading on Amazon EKS using InstantTensor loader with GPUDirect Storage (GDS) on Amazon FSx for Lustre

This repository contains the container image, EKS manifests, and setup scripts to
benchmark vLLM cold-start model-loading time on a single `p5en.48xlarge`, comparing
three read paths for the [DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) full weights (892.7 GB).

Using the new [InstantTensor](https://docs.vllm.ai/en/latest/models/extensions/instanttensor/)
loader with NVIDIA GPUDirect Storage (GDS) on Amazon FSx for Lustre, we were able to cut the vLLM
model load time from 31 minutes to 51 seconds — a **~36x speedup** in our testing.

## Introduction

When a vLLM replica starts, it has to read the whole model checkpoint from storage into
GPU memory before it can serve requests. For a 892.7 GB model this weight-loading step
dominates cold start, and vLLM's default loader runs it on a single, CPU-bound path — reading
each file, deserializing it, then copying the tensors to each GPU.

This repository demonstrates how much that step can be cut by taking the read
path off the default CPU loader and onto a direct storage→HBM DMA transfer via NVIDIA GPUDirect Storage
(GDS), using the InstantTensor loader on Amazon FSx for Lustre. For comparison, we also
include S3 parallel streaming with the Run:AI Model Streamer (another common approach),
and the vLLM default loader (via FSx) as the baseline.

We look at the weight-loading phase only, taken straight from vLLM's own `Model loading
took` log line at cold start — not inference latency, throughput, or accuracy.

### Why InstantTensor

It is worth noting that vLLM's default loader (`--load-format auto`) is CPU-based and never uses GDS.
Of the two GDS-capable loaders, only one keeps the fast path when using tensor parallelism (TP>1).

- **`fastsafetensors`** can use GDS, but vLLM forces `nogds=True` whenever TP>1
  (see upstream [PR #34070](https://github.com/vllm-project/vllm/pull/34070)) — so at TP=8 it silently falls back to the CPU path.
- **`instanttensor`** passes the process group through instead of disabling GDS, so it
  keeps the storage→HBM DMA at TP>1. That is why this PoC uses it.

## Architecture

- One `p5en.48xlarge` — 8× H200 (141 GiB each, 1,128 GiB HBM), 16 EFA interfaces.
- Amazon EKS; the GPU node joins on an on-demand capacity block. Weights served
  by vLLM 0.28.0, TP=8, `mp` executor, from a container image that adds the GDS
  toolchain and `instanttensor` on top of `vllm/vllm-openai:v0.28.0`.
- Amazon FSx for Lustre, PERSISTENT_2, 16 OSTs (75 GB/s provisioned), EFA-enabled.
- An S3 bucket in the same Region holds a second copy for the Run:AI arm.

## Measured Performance

For this proof-of-concept test, we measure the model loading time directly from vLLM's own
`Model loading took … seconds` log line — emitted once per rank by `gpu_model_runner.py` as each
worker finishes reading its shard of weights into GPU memory. Every number below is a cold-start
measurement (OS page cache dropped before each run), for the DeepSeek-V4-Pro-0813 full weights (66 shards, 892.7 GB).

| source / loader | load time (mean of n) | vs default |
|---|---|---|
| FSx Lustre, default vLLM (`auto`) | 1847.7 s (n=6) | 1.0× |
| S3 + Run:AI streamer (`concurrency: 32`) | ~350 s (n=3) | 5.3× |
| FSx Lustre + GDS (`instanttensor` CUFILE) | 50.9 s (n=6) | 36.3× |


## Repository Structure

```
cluster/         eksctl templates: EKS cluster + a plain p5en GPU nodegroup (no GDS in bootstrap)
image/           container image (vLLM + GDS toolchain + instanttensor + baked cufile.json)
scripts/         numbered flow: 01 mount/stripe + 02 stage weights (staging node),
                 03 GDS host setup (GPU node, post-Ready), 04 three-arm benchmark
manifests/       three vLLM serve manifests (default | GDS | S3) + fsx-lustre-pv-pvc.yaml
                 (static FSx CSI PV/PVC — the pods mount FSx via the CSI driver)
KNOWN_ISSUES.md  intrinsic traps (GDS host build, cufile.json, privileged pod, striping, FP8 pairing)
```

## Prerequisites

- An EKS-capable account with a `p5en.48xlarge` capacity block.
- `eksctl`, `kubectl`, `aws` CLI, `envsubst` (from `gettext`), and a Hugging Face token.
- An existing VPC with private and public subnets in two AZs (EKS control-plane minimum). One AZ must be the capacity-block AZ where the GPU node and FSx live.
- A container registry (ECR) for the image built from `image/Dockerfile`.

## Configuration

Set these once at the start of your shell session. Replace every `xxxxxxxx`
placeholder with the IDs from your account (the `cluster/` templates and
`manifests/` are rendered from these with `envsubst`).

```bash
# Identity / region
export AWS_REGION="${AWS_REGION:-us-west-2}"
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export CLUSTER="dsv4p-w2"

# Existing VPC + subnets (two AZs; EKS control-plane minimum).
# PRIVATE_SUBNET_A must be the capacity-block AZ where the GPU node and FSx live.
export VPC="vpc-xxxxxxxx"
export PRIVATE_SUBNET_A="subnet-xxxxxxxx"   # us-west-2b — FSx + GPU node live here
export PRIVATE_SUBNET_B="subnet-xxxxxxxx"   # us-west-2c — 2nd AZ for the control plane
export PUBLIC_SUBNET_A="subnet-xxxxxxxx"
export PUBLIC_SUBNET_B="subnet-xxxxxxxx"

# Capacity block reservation for the p5en.48xlarge
export CAPACITY_RESERVATION="cr-xxxxxxxx"

# Model
export MODEL_REPO="deepseek-ai/DeepSeek-V4-Pro-0813"
export MODEL_NAME="dsv4-pro"
export MODEL_DIR="/fsx/models/dsv4-pro-mxfp4"
export MODEL_BYTES="892744322880"

# S3 bucket (globally unique; public access blocked) and the container image
export S3_BUCKET="dsv4p-weights-${AWS_ACCOUNT_ID}-usw2"
export IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/vllm-gds:gds"

# FSx for Lustre — fill in after Step 2 creates the filesystem
export FSX_ID="fs-xxxxxxxx"
export FSX_MOUNT="xxxxxxxx"     # aws fsx describe-file-systems --file-system-id $FSX_ID \
                                #   --query 'FileSystems[0].LustreConfiguration.MountName' --output text
export FSX_DNS="${FSX_ID}.fsx.${AWS_REGION}.amazonaws.com"
```

## Deployment

### Step 1: FSx security group
```bash
SG=$(aws ec2 create-security-group --group-name fsx-gds-w2 --vpc-id $VPC \
       --description "FSx Lustre + EFA" --query GroupId --output text)
for D in ingress egress; do
  aws ec2 authorize-security-group-$D --group-id $SG --protocol -1 --source-group $SG
done
```

### Step 2: FSx for Lustre, 16 OSTs
```bash
FSX_ID=$(aws fsx create-file-system \
  --file-system-type LUSTRE --storage-capacity 76800 --storage-type SSD \
  --subnet-ids $PRIVATE_SUBNET_A --security-group-ids $SG \
  --lustre-configuration '{"DeploymentType":"PERSISTENT_2","PerUnitStorageThroughput":1000,"EfaEnabled":true,"MetadataConfiguration":{"Mode":"AUTOMATIC"}}' \
  --tags Key=Name,Value=fsx-dsv4p-16ost \
  --query FileSystem.FileSystemId --output text)
export FSX_ID
export FSX_MOUNT=$(aws fsx describe-file-systems --file-system-id $FSX_ID \
  --query 'FileSystems[0].LustreConfiguration.MountName' --output text)
export FSX_DNS="${FSX_ID}.fsx.${AWS_REGION}.amazonaws.com"
```
Want: 76800 / 4800 = **16 OSTs**, 75 GB/s provisioned.

### Step 3: EKS cluster, staging node, cross-SG rules
```bash
envsubst < cluster/cluster.yaml | eksctl create cluster -f - --kubeconfig $HOME/.kube/$CLUSTER.config
CLSG=$(aws eks describe-cluster --name $CLUSTER \
  --query cluster.resourcesVpcConfig.clusterSecurityGroupId --output text)
for P in "$SG $CLSG" "$CLSG $SG"; do set -- $P
  aws ec2 authorize-security-group-ingress --group-id $1 --protocol -1 --source-group $2
  aws ec2 authorize-security-group-egress  --group-id $1 --protocol -1 --source-group $2
done
```

### Step 4: Mount FSx, set striping, stage weights (staging node, via SSM)
```bash
bash scripts/01-mount-and-stripe.sh
bash scripts/02-stage-weights.sh          # curl, not huggingface_hub — see KNOWN_ISSUES
```

### Step 5: Upload a copy to S3 (for the S3 arm)
```bash
aws s3 cp $MODEL_DIR/ s3://$S3_BUCKET/$MODEL_NAME/ --recursive --only-show-errors
```

### Step 6: GPU node on the capacity block
```bash
envsubst < cluster/gpu-nodegroup-p5en.yaml | eksctl create nodegroup -f -
```

### Step 7: GDS host setup (run once, after the node is k8s-Ready)
Run on the GPU node via SSM. GDS setup is **not** in node bootstrap: the EFA-over-LNet step
must load LNet with its CPU-partition table, which only works once the node is fully up — doing
it in bootstrap loads LNet too early and the EFA CPT pinning fails (3 of 16 interfaces attach).
```bash
export FSX_DNS FSX_MOUNT   # from Step 2
bash scripts/03-host-gds.sh          # nvidia-fs (pinned v2.29.4) + EFA/LNet; expect 16 EFA NIDs
```

### Step 8: FSx CSI driver + static PV/PVC
The serve pods mount FSx via the **FSx for Lustre CSI driver** (static PV/PVC). Install the driver,
then bind the pre-created FSx from Step 2. GDS still takes the storage→HBM DMA fast-path through this
CSI-mounted volume: the CSI node plugin does a host-kernel `mount -t lustre` and bind-mounts it into
the pod, and the host's `nvidia-fs` + LNet-over-EFA (from Step 7) serve that mount — so the CSI layer
does not change the DMA path (`readMiB` still moves by the full checkpoint).
```bash
kubectl apply -k "github.com/kubernetes-sigs/aws-fsx-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.4"
envsubst '${FSX_ID} ${FSX_DNS} ${FSX_MOUNT}' < manifests/fsx-lustre-pv-pvc.yaml | kubectl apply -f -
```

### Step 9: Run the three-arm benchmark
```bash
bash scripts/04-loader-ab.sh              # renders manifests via envsubst; FSx default | S3 Run:AI | FSx GDS
```

## Cleanup

```bash
envsubst < cluster/gpu-nodegroup-p5en.yaml | eksctl delete nodegroup -f - --approve
eksctl delete cluster --name $CLUSTER
aws fsx delete-file-system --file-system-id $FSX_ID
aws s3 rm s3://$S3_BUCKET/$MODEL_NAME/ --recursive
aws s3 rb s3://$S3_BUCKET
```

## Conclusion

On this p5en.48xlarge node GDS loads the checkpoint ~36× faster than the default vLLM loader
and ~7× faster than a parallel S3 stream.

## Known Issues

See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## License

MIT-0. See [LICENSE](LICENSE).
