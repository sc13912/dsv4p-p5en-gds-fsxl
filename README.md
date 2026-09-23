# Accelerate vLLM model loading on Amazon EKS using InstantTensor loader with GPUDirect Storage (GDS) on Amazon FSx for Lustre

This repository contains the container image, EKS manifests, and setup scripts to
benchmark vLLM cold-start model-loading time on a single `p5en.48xlarge`, comparing
three read paths for the [DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) full weights (892.7 GB).

Using the new [InstantTensor](https://docs.vllm.ai/en/latest/models/extensions/instanttensor/)
loader with NVIDIA GPUDirect Storage (GDS) on Amazon FSx for Lustre, we were able to cut the vLLM
model load time from about 29 minutes to 45 seconds — a **~38x speedup** in our testing.

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
  toolchain and `instanttensor` on top of the AWS Deep Learning Container
  `public.ecr.aws/deep-learning-containers/vllm:0.28.0-gpu-py312-cu130-ubuntu24.04-ec2`
  (the same upstream vLLM 0.28.0 release, republished by AWS with EFA and aws-ofi-nccl).
- Amazon FSx for Lustre, PERSISTENT_2, 8 OSTs (37.5 GB/s provisioned), EFA-enabled.
- An S3 bucket in the same Region holds a second copy for the Run:AI arm.

## Measured Performance

For this proof-of-concept test, we measure the model loading time directly from vLLM's own
`Model loading took … seconds` log line — emitted once per rank by `model_runner.py` as each
worker finishes reading its shard of weights into GPU memory. Every number below is a cold-start
measurement, for the DeepSeek-V4-Pro-0813 full weights (66 shards, 892.7 GB).

Each arm runs in its own pod and `04-loader-ab.sh` deletes that deployment before starting the
next, which takes the arm's page cache with it - measured 22 GB still cached on a 2 TiB node
after the default arm had read 831 GiB. So no explicit cache drop is needed, but do not re-run a
single arm back-to-back and compare: that one is warm.

| source / loader | load time (mean of 3 runs) | vs default |
|---|---|---|
| FSx Lustre, default vLLM (`auto`) | 1712.7 s | 1.0× |
| S3 + Run:AI streamer (`concurrency: 32`) | 368.7 s | 4.6× |
| FSx Lustre + GDS (`instanttensor` CUFILE) | **45.1 s** | **38.0×** |

Per-run values: default 1690.7 / 1672.4 / 1775.0 s · S3 372.8 / 364.8 / 368.5 s ·
GDS 44.7 / 45.1 / 45.4 s. GDS is also ~8.2x faster than the S3 path.

Run-to-run variance differs sharply by arm: GDS spans 1.4% and S3 2.2%, but the default
loader spans about 40% across a wider sample (1543-2409 s over 12 cold runs in two
Regions). Treat the default figure as an order of magnitude, not a precise number.

### 8 vs 16 OSTs: 8 is the cost-effective choice

We also ran the same GDS arm on a 16-OST filesystem (75 GB/s provisioned, twice the cost):

| OSTs | provisioned | GDS load time (mean of 5) | transfer peak |
|---|---|---|---|
| 8 | 37.5 GB/s | 45.1 s | 42.5 GB/s |
| 16 | 75 GB/s | 41.4 s | 50.0 GB/s |

Doubling provisioned bandwidth bought **8.3%** on load time, and GDS never came close to
saturating even the 8-OST limit — the load is bound by fixed per-operation overhead
(cuFile setup, per-file open, and post-load weight processing), not by bandwidth. Only
about 18-20 s of the ~45 s total is actual DMA.

The default loader gained nothing measurable from the extra OSTs: it averages ~0.5 GB/s,
roughly 1.4% of what 8 OSTs already provide. A 16-OST sample (n=5) averaged 1878.5 s
against 1812.2 s at 8 OSTs (n=7) - a 3.7% difference swamped by the ~40% run-to-run
spread.

**So 8 OSTs is the better value for this workload**: half the FSx cost for ~8% on the GDS
arm and nothing on the others.


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
- `eksctl`, `kubectl`, `aws` CLI, `envsubst` (from `gettext`), `jq`, and `base64`.
- An existing VPC with private and public subnets in two AZs (EKS control-plane minimum). One AZ must be the capacity-block AZ where the GPU node and FSx live.
- Docker, to build the image from `image/Dockerfile` (Step 2 pushes it to ECR).

## Configuration

Set these once at the start of your shell session. Replace every `xxxxxxxx`
placeholder with the IDs from your account (the `cluster/` templates and
`manifests/` are rendered from these with `envsubst`).

```bash
# Identity / region
export AWS_REGION="${AWS_REGION:-us-east-2}"
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export CLUSTER="dsv4p-e2"

# Capacity block reservation for the p5en.48xlarge, and the AZ it was granted in.
# Read the AZ off the reservation - blocks are not always in the first AZ of the Region.
export CAPACITY_RESERVATION="cr-xxxxxxxx"
export AZ_A=$(aws ec2 describe-capacity-reservations --output text \
  --capacity-reservation-ids $CAPACITY_RESERVATION --query 'CapacityReservations[0].AvailabilityZone')
export AZ_B="${AWS_REGION}b"                # any second AZ for the control plane; must differ from AZ_A

# Existing VPC + subnets (two AZs; EKS control-plane minimum).
export VPC="vpc-xxxxxxxx"
export PRIVATE_SUBNET_A="subnet-xxxxxxxx"   # in $AZ_A — FSx + the GPU node live here
export PRIVATE_SUBNET_B="subnet-xxxxxxxx"   # in $AZ_B — 2nd AZ for the control plane
export PUBLIC_SUBNET_A="subnet-xxxxxxxx"    # in $AZ_A
export PUBLIC_SUBNET_B="subnet-xxxxxxxx"    # in $AZ_B

# Model
export MODEL_REPO="deepseek-ai/DeepSeek-V4-Pro-0813"
export MODEL_NAME="dsv4-pro"
export MODEL_DIR="/fsx/models/dsv4-pro-mxfp4"

# S3 bucket (globally unique; public access blocked) and the container image
export S3_BUCKET="dsv4p-weights-${AWS_ACCOUNT_ID}-use2"
export IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/vllm-gds:gds"

# FSx for Lustre — fill in after Step 4 creates the filesystem
export FSX_ID="fs-xxxxxxxx"
export FSX_MOUNT="xxxxxxxx"     # aws fsx describe-file-systems --file-system-id $FSX_ID \
                                #   --query 'FileSystems[0].LustreConfiguration.MountName' --output text
export FSX_DNS="${FSX_ID}.fsx.${AWS_REGION}.amazonaws.com"
```

## Deployment

### Step 1: S3 bucket for the weights
```bash
aws s3api create-bucket --bucket $S3_BUCKET --region $AWS_REGION \
  --create-bucket-configuration LocationConstraint=$AWS_REGION
aws s3api put-public-access-block --bucket $S3_BUCKET \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```
Check: `aws s3api get-public-access-block --bucket $S3_BUCKET` — all four `true`.

Create the bucket before Step 5: the staging node's S3 permissions are rendered from
`$S3_BUCKET`, so changing the name after the cluster exists leaves the policy on the old
bucket. In `us-east-1`, omit `--create-bucket-configuration`.

### Step 2: Container image
```bash
aws ecr create-repository --repository-name vllm-gds --region $AWS_REGION 2>/dev/null || true
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin \
      ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
docker build -t $IMAGE image/
docker push $IMAGE
```
Check: `aws ecr describe-images --repository-name vllm-gds --region $AWS_REGION \
  --image-ids imageTag=${IMAGE##*:} --query 'imageDetails[0].imagePushedAt' --output text`
— prints a timestamp.

### Step 3: FSx security group
```bash
SG=$(aws ec2 create-security-group --group-name fsx-gds-$CLUSTER --vpc-id $VPC \
       --description "FSx Lustre + EFA" --query GroupId --output text)
for D in ingress egress; do
  aws ec2 authorize-security-group-$D --group-id $SG --protocol -1 --source-group $SG
done
```

### Step 4: FSx for Lustre, 8 OSTs
```bash
FSX_ID=$(aws fsx create-file-system \
  --file-system-type LUSTRE --storage-capacity 38400 --storage-type SSD \
  --subnet-ids $PRIVATE_SUBNET_A --security-group-ids $SG \
  --lustre-configuration '{"DeploymentType":"PERSISTENT_2","PerUnitStorageThroughput":1000,"EfaEnabled":true,"MetadataConfiguration":{"Mode":"AUTOMATIC"}}' \
  --tags Key=Name,Value=fsx-dsv4p-8ost \
  --query FileSystem.FileSystemId --output text)
export FSX_ID
export FSX_MOUNT=$(aws fsx describe-file-systems --file-system-id $FSX_ID \
  --query 'FileSystems[0].LustreConfiguration.MountName' --output text)
export FSX_DNS="${FSX_ID}.fsx.${AWS_REGION}.amazonaws.com"
```
Want: 38400 / 4800 = **8 OSTs**, 37.5 GB/s provisioned.

### Step 5: EKS cluster, staging node, cross-SG rules
```bash
envsubst < cluster/cluster.yaml | eksctl create cluster -f - --kubeconfig $HOME/.kube/$CLUSTER.config
CLSG=$(aws eks describe-cluster --name $CLUSTER \
  --query cluster.resourcesVpcConfig.clusterSecurityGroupId --output text)
for P in "$SG $CLSG" "$CLSG $SG"; do set -- $P
  aws ec2 authorize-security-group-ingress --group-id $1 --protocol -1 --source-group $2
  aws ec2 authorize-security-group-egress  --group-id $1 --protocol -1 --source-group $2
done
```

### Step 6: Mount FSx, set striping, stage weights

`01` and `02` run **on the staging node**, not on your workstation. Define this helper once; it
ships a local script to a node over SSM, base64-encoded so quoting never bites, and waits for it.

```bash
run_on_node() {                      # run_on_node <instance-id> <script> [VAR=val ...]
  local id=$1 script=$2; shift 2
  local cid=$(aws ssm send-command --instance-ids "$id" \
    --document-name AWS-RunShellScript --timeout-seconds 3600 \
    --parameters commands="[\"echo $(base64 -w0 $script) | base64 -d > /tmp/s.sh; $* bash /tmp/s.sh\"]" \
    --query Command.CommandId --output text)
  local st=InProgress
  until [ "$st" != InProgress ] && [ "$st" != Pending ]; do
    sleep 15
    st=$(aws ssm get-command-invocation --command-id $cid --instance-id $id \
           --query Status --output text)
  done
  echo "--- $script: $st"
  aws ssm get-command-invocation --command-id $cid --instance-id $id \
    --query StandardOutputContent --output text
}

STAGING_NODE=$(aws ec2 describe-instances --region $AWS_REGION \
  --filters Name=tag:eks:nodegroup-name,Values=cpu-stage Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].InstanceId' --output text)

run_on_node $STAGING_NODE scripts/01-mount-and-stripe.sh \
  FSX_DNS=$FSX_DNS FSX_MOUNT=$FSX_MOUNT MODEL_DIR=$MODEL_DIR
run_on_node $STAGING_NODE scripts/02-stage-weights.sh \
  FSX_DNS=$FSX_DNS FSX_MOUNT=$FSX_MOUNT MODEL_DIR=$MODEL_DIR MODEL_REPO=$MODEL_REPO
```
Check: `01` prints `8` active OSTs and `stripe_count: -1`; `02` ends with
`all 71 files verified byte-exact`. `02` downloads 892.7 GB and takes roughly 20 minutes.

### Step 7: Upload a copy to S3 (for the S3 arm)
```bash
aws s3 cp $MODEL_DIR/ s3://$S3_BUCKET/$MODEL_NAME/ --recursive --only-show-errors
```

### Step 8: GPU node on the capacity block
```bash
envsubst < cluster/gpu-nodegroup-p5en.yaml | eksctl create nodegroup -f -
```

### Step 9: GDS host setup (run once, after the node is k8s-Ready)
Run on the GPU node via SSM. GDS setup is **not** in node bootstrap: the EFA-over-LNet step
must load LNet with its CPU-partition table, which only works once the node is fully up — doing
it in bootstrap loads LNet too early and the EFA CPT pinning fails (3 of 16 interfaces attach).
```bash
GPU_NODE=$(aws ec2 describe-instances --region $AWS_REGION \
  --filters Name=tag:eks:nodegroup-name,Values=gpu-p5en-cb Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].InstanceId' --output text)

run_on_node $GPU_NODE scripts/03-host-gds.sh
```
Check: ends with `nvidia-fs devices: 16    EFA NIDs: 16`. Takes about 10 minutes, most of it
compiling nvidia-fs.

Then confirm Lustre is really using EFA - the interfaces come up and sit idle if the security
groups do not authorise it, and Lustre falls back to TCP at a fraction of the bandwidth with no
error anywhere:
```bash
run_on_node $GPU_NODE /dev/stdin <<'EOS'
lnetctl net show -v | awk '/net type: /{t=$4} /recv_count:/{a[t]+=$2} END{for(k in a) print k, a[k]}'
EOS
```
Want: a non-zero and growing `efa` count once the benchmark runs. Counting EFA NIDs proves
nothing - 16 can be present and `up` while every byte travels over TCP.

### Step 10: FSx CSI driver + static PV/PVC
The serve pods mount FSx via the **FSx for Lustre CSI driver** (static PV/PVC). Install the driver,
then bind the pre-created FSx from Step 4. GDS still takes the storage→HBM DMA fast-path through this
CSI-mounted volume: the CSI node plugin does a host-kernel `mount -t lustre` and bind-mounts it into
the pod, and the host's `nvidia-fs` + LNet-over-EFA (from Step 9) serve that mount — so the CSI layer
does not change the DMA path (`readMiB` still moves by the full checkpoint).
```bash
kubectl apply -k "github.com/kubernetes-sigs/aws-fsx-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.4"
envsubst '${FSX_ID} ${FSX_DNS} ${FSX_MOUNT}' < manifests/fsx-lustre-pv-pvc.yaml | kubectl apply -f -
```

### Step 11: Run the three-arm benchmark
```bash
bash scripts/04-loader-ab.sh              # renders manifests via envsubst; FSx default | S3 Run:AI | FSx GDS
```
Check: about 50 minutes for all three arms. Each writes `dsv4p-{default,s3,gds}.log` and prints
eight `Model loading took` lines, one per tensor-parallel rank; take the slowest. `NOT READY`
means that arm never served and its log holds the reason.

Then confirm on the node that GDS carried the checkpoint, rather than the POSIX path silently
standing in for it:
```bash
run_on_node $GPU_NODE /dev/stdin <<'EOS'
grep -E 'readMiB' /proc/driver/nvidia-fs/stats
lnetctl net show -v | awk '/net type: /{t=$4} /recv_count:/{a[t]+=$2} END{for(k in a) print k, a[k]}'
EOS
```
Want: `readMiB` at least the size of the checkpoint (851371 MiB here) with `err=0`, and `efa`
far larger than `tcp`. Both counters are cumulative on the host and outlive the pod.

## Cleanup

```bash
envsubst < cluster/gpu-nodegroup-p5en.yaml | eksctl delete nodegroup -f - --approve
eksctl delete cluster --name $CLUSTER
aws fsx delete-file-system --file-system-id $FSX_ID
aws s3 rm s3://$S3_BUCKET/$MODEL_NAME/ --recursive
aws s3 rb s3://$S3_BUCKET
aws ecr delete-repository --repository-name vllm-gds --force --region $AWS_REGION
```
The FSx security group from Step 4 outlives the filesystem and will hold up a later VPC
deletion. Remove it once FSx is gone and has released its network interfaces:
```bash
while aws fsx describe-file-systems --file-system-ids $FSX_ID >/dev/null 2>&1; do sleep 30; done
aws ec2 delete-security-group --group-id $SG
```

## Conclusion

On this p5en.48xlarge node GDS loads the checkpoint **38× faster** than the default vLLM loader
and **8.2× faster** than a parallel S3 stream.

## Known Issues

See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## License

MIT-0. See [LICENSE](LICENSE).
