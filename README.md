# Accelerate vLLM model loading on Amazon EKS using InstantTensor loader with GPUDirect Storage (GDS) on Amazon FSx for Lustre

This repository aims to help customers running self-managed LLMs on Amazon EKS reduce vLLM
cold-start model-loading time. Using the [InstantTensor](https://docs.vllm.ai/en/latest/models/extensions/instanttensor/)
loader with NVIDIA GPUDirect Storage (GDS) on Amazon FSx for Lustre, we were able to cut the vLLM
weight loading time from about 28 minutes to 35 seconds — a **~48x speedup** in our testing.

It contains the AWS infrastructure, EKS manifests and setup scripts to benchmark vLLM cold-start
model-loading time on a single `p5en.48xlarge`, comparing three read paths for the 1.6-trillion
parameter [DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) full
weights (892.7 GB):

- FSx for Lustre with the default loader
- S3 + [Run:ai Model Streamer](https://github.com/dsx-ai-factory/model-streamer)
- FSx for Lustre with GDS

## Disclaimer

This sample repository is for a proof-of-concept test. It is provided for demonstration purposes only
and should be thoroughly reviewed for security, compliance, and cost implications before any production
use.

## Introduction

When a vLLM replica starts, it has to read the whole model checkpoint from storage into
GPU memory before it can serve requests. For a 892.7 GB model this weight-loading step
dominates cold start, and vLLM's default loader runs it on a single, CPU-bound path — reading
each file, deserializing it, then copying the tensors to each GPU.

This repository measures how much faster that step gets when the weights go straight from storage
into GPU memory over NVIDIA GPUDirect Storage (GDS), using the InstantTensor loader on Amazon FSx
for Lustre. For comparison we also run the same weights through the
[NVIDIA Run:AI Model Streamer](https://github.com/dsx-ai-factory/model-streamer) from S3, and
through vLLM's default loader as the baseline.

We look at the weight-loading phase only, using the `Loading weights took` line that vLLM prints
during a cold start. Inference latency and throughput are out of scope.

### Why InstantTensor

vLLM's default loader (`--load-format auto`) is CPU-based and never touches GDS.
Of the two GDS-capable loaders, only one keeps the fast path when using tensor parallelism (TP>1).

- **`fastsafetensors`** can use GDS, but vLLM forces `nogds=True` whenever TP>1
  (see upstream [PR #34070](https://github.com/vllm-project/vllm/pull/34070)) — so at TP=8 it silently falls back to the CPU path.
- **`instanttensor`** passes the process group through instead of disabling GDS, so it
  keeps the storage→HBM DMA at TP>1.

## Architecture

- One `p5en.48xlarge` — 8× H200 (141 GiB each, 1,128 GiB HBM), 16 EFA interfaces.
- Amazon EKS cluster where the GPU node joins on an on-demand capacity block.
- Weights are served by **vLLM 0.28.0**, **TP=8**, `mp` executor, from a container image that adds the GDS
  toolchain and `instanttensor` on top of the AWS Deep Learning Container (DLC)
  `public.ecr.aws/deep-learning-containers/vllm:0.28.0-gpu-py312-cu130-ubuntu24.04-ec2`
  (the same upstream vLLM 0.28.0 release, republished by AWS with EFA and aws-ofi-nccl).
- Amazon FSx for Lustre, **PERSISTENT_2**, **8 OSTs** (37.5 GB/s provisioned), EFA-enabled.
- An S3 bucket in the same Region holds another copy of the model weights for the Run:AI arm.

## Measured Performance

For this proof-of-concept test, we report two figures per testing arm. The first is how long vLLM
spends loading weights, which it reports in a `Loading weights took … seconds` line from
`default_loader.py`. Only rank 0 prints it. The second is the total model load time from
`model_runner.py`. Every rank prints that one, and we take the slowest. The difference between the
two is a post-load processing step, which we can measure on the two FSx arms and which comes to
about 9.6 seconds on both.

The Run:AI streamer never reports its own load time, so for that arm we read the elapsed time off
the loader's progress bar instead.

Every arm drops the host page cache (`sync; echo 3 > /proc/sys/vm/drop_caches`) before loading, so
each figure is a cold read. Without it the default loader can come in around 4x faster off a warm
cache, which understates the speedup rather than inflating it.

| source / loader | weight load | total model load | vs default |
|---|---|---|---|
| FSx Lustre, default vLLM (`auto`) | 1703.0 s | 1712.7 s | 1.0× |
| S3 + Run:AI streamer (`concurrency: 32`) | 316.3 s[^1] | 368.7 s | 5.4× |
| FSx Lustre + GDS (`instanttensor` CUFILE) | **35.5 s** | 45.1 s | **48.0×** |

Per-run weight-load values (tested in `us-east-2`):
- FSx for Lustre with default loader: 1681.4 / 1662.6 / 1765.1 s
- S3 + Run:ai model streamer: 246 / 351 / 352 s
- FSx for Lustre with GDS: 35.1 / 35.5 / 35.8 s

[^1]: Read from rank 0's progress bar; the S3 arm prints no other rank's. Its ranks' loading times vary by
    up to 57%, so a rank-0-only figure moves far more between runs than the total does.

The GDS arm spends 21 of its 35.5 seconds moving bytes from storage into GPU memory, at
42.5 GB/s. That is faster than the 37.5 GB/s the filesystem provisions, which is only possible
because the FSx file servers cache reads in memory. The remaining 14 seconds go on placing
roughly 150,000 tensors into model parameters, about 87 microseconds each. So the per-tensor
placement takes roughly 40% of the weight load, and that part is not bandwidth bound.

### 8 vs 16 OSTs: 8 is the cost-effective choice

We also ran the same GDS arm on a 16-OST filesystem (75 GB/s provisioned, twice the cost):

| OSTs | provisioned | weight load | DMA phase | DMA throughput |
|---|---|---|---|---|
| 8 | 37.5 GB/s | 35.5 s (n=3) | 21.0 s | 42.5 GB/s |
| 16 | 75 GB/s | 31.9 s (n=5) | 19.4 s | 46.0 GB/s |

Doubling the provisioned bandwidth bought **10.1%** on weight load. At 8 OSTs the loader is pressing
against the filesystem, reading at 113% of what it provisions. At 16 OSTs it reaches only 61%, so the
extra bandwidth sits unused and the loader itself becomes the limit.

**So 8 OSTs is more cost-effective for this workload**: half the FSx cost for ~10% weight load
difference.


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
- An existing VPC with private and public subnets across two AZs. One AZ must be the capacity-block AZ where the GPU node and FSx live.
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
Expect: `aws s3api get-public-access-block --bucket $S3_BUCKET` to show all four `true`.

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
Expect: `aws ecr describe-images --repository-name vllm-gds --region $AWS_REGION \
  --image-ids imageTag=${IMAGE##*:} --query 'imageDetails[0].imagePushedAt' --output text`
to print a timestamp.

### Step 3: FSx security group
```bash
SG=$(aws ec2 create-security-group --group-name fsx-gds-$CLUSTER --vpc-id $VPC \
       --description "FSx Lustre + EFA" --query GroupId --output text)
for D in ingress egress; do
  aws ec2 authorize-security-group-$D --group-id $SG --protocol -1 --source-group $SG
done
```
Expect: `aws ec2 describe-security-groups --group-ids $SG --query 'SecurityGroups[0].[IpPermissions[].UserIdGroupPairs[].GroupId, IpPermissionsEgress[].UserIdGroupPairs[].GroupId]' --output text`
to print `$SG` on both lines.

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
Expect: 38400 / 4800 = **8 OSTs**, 37.5 GB/s provisioned.

### Step 5: EKS cluster, staging node, cross-SG rules
```bash
envsubst < cluster/cluster.yaml | eksctl create cluster -f - --kubeconfig $HOME/.kube/$CLUSTER.config
export KUBECONFIG=$HOME/.kube/$CLUSTER.config
CLSG=$(aws eks describe-cluster --name $CLUSTER \
  --query cluster.resourcesVpcConfig.clusterSecurityGroupId --output text)
for P in "$SG $CLSG" "$CLSG $SG"; do set -- $P
  aws ec2 authorize-security-group-ingress --group-id $1 --protocol -1 --source-group $2
  aws ec2 authorize-security-group-egress  --group-id $1 --protocol -1 --source-group $2
done
```
Expect: `aws ec2 describe-security-groups --group-ids $SG $CLSG --query 'SecurityGroups[].[GroupId, IpPermissions[].UserIdGroupPairs[].GroupId, IpPermissionsEgress[].UserIdGroupPairs[].GroupId]' --output text`
to list each group under the other's ingress and egress, and `kubectl get nodes` to show the staging node `Ready`.

### Step 6: Mount FSx, set striping, stage weights

`01` and `02` run **on the staging node**, not on your workstation. Define this helper once; it
ships a local script to a node over SSM, base64-encoded so quoting never bites, and waits for it.

```bash
run_on_node() {                      # run_on_node <instance-id> <script> [VAR=val ...]
  local id=$1 script=$2; shift 2
  local cid=$(aws ssm send-command --instance-ids "$id" \
    --document-name AWS-RunShellScript --timeout-seconds 3600 \
    --parameters commands="[\"echo $(base64 < $script | tr -d '\n') | base64 -d > /tmp/s.sh; $* bash /tmp/s.sh\"]" \
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
Expect: `01` to print `8` active OSTs and `stripe_count: -1`, and `02` to end with
`all 71 files verified byte-exact`. `02` downloads 892.7 GB and takes roughly 15 minutes.

### Step 7: Upload a copy to S3 (for the S3 arm)
Also on the staging node: `$MODEL_DIR` lives on FSx, which is not mounted on your workstation.
```bash
run_on_node $STAGING_NODE /dev/stdin \
  MODEL_DIR=$MODEL_DIR S3_BUCKET=$S3_BUCKET MODEL_NAME=$MODEL_NAME <<'EOS'
aws s3 cp $MODEL_DIR/ s3://$S3_BUCKET/$MODEL_NAME/ --recursive --only-show-errors
EOS
```
Expect: `aws s3 ls s3://$S3_BUCKET/$MODEL_NAME/ --summarize | tail -2` to show 71 objects, 892,762,344,570 bytes. Takes
about 20 minutes.

### Step 8: GPU node on the capacity block
```bash
envsubst < cluster/gpu-nodegroup-p5en.yaml | eksctl create nodegroup -f -
```
Expect: `kubectl get nodes -l role=gpu` to show one node `Ready` within about 5 minutes.

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
Expect: about 3 minutes. The build log is longer than SSM returns, so the script's final line is
usually cut off in `run_on_node`'s output; verify on the node instead:
```bash
run_on_node $GPU_NODE /dev/stdin <<'EOS'
echo "nvidia-fs devices: $(ls /dev/nvidia-fs* | wc -l)    EFA NIDs: $(lnetctl net show | grep -c '@efa')"
EOS
```
Expect: `nvidia-fs devices: 16    EFA NIDs: 16`.

Then confirm Lustre is really using EFA - the interfaces come up and sit idle if the security
groups do not authorise it, and Lustre falls back to TCP at a fraction of the bandwidth with no
error anywhere:
```bash
run_on_node $GPU_NODE /dev/stdin <<'EOS'
lnetctl net show -v | awk '/net type: /{t=$4} /recv_count:/{a[t]+=$2} END{for(k in a) print k, a[k]}'
EOS
```
Expect: a non-zero and growing `efa` count once the benchmark runs. Counting EFA NIDs proves
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
Expect: `kubectl get pvc fsx-dsv4p-pvc` to show `Bound`.

### Step 11: Run the three-arm benchmark
```bash
bash scripts/04-loader-ab.sh              # renders manifests via envsubst; FSx default | S3 Run:AI | FSx GDS
```
Expect: about an hour for all three arms. Each writes `dsv4p-{default,s3,gds}.log`. For every
arm the script prints the weight-load figure first, then eight `Model loading took` lines, one per
tensor-parallel rank - take the slowest of those. On the S3 arm the weight-load figure comes from
the loader's progress bar, since the Run:AI streamer does not log its own. `NOT READY` means that
arm never served and its log holds the reason.

Then confirm on the node that GDS carried the checkpoint, rather than the POSIX path silently
standing in for it:
```bash
run_on_node $GPU_NODE /dev/stdin <<'EOS'
grep -E 'readMiB' /proc/driver/nvidia-fs/stats
lnetctl net show -v | awk '/net type: /{t=$4} /recv_count:/{a[t]+=$2} END{for(k in a) print k, a[k]}'
EOS
```
Expect: `readMiB` at least the size of the checkpoint (851371 MiB here) with `err=0`, and `efa`
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

On a single p5en.48xlarge, moving the read path onto GPUDirect Storage cut weight loading for a
1.6-trillion parameter checkpoint from about 28 minutes to 35 seconds. That is 48× faster than
vLLM's default loader, and 8.9× faster than streaming the same weights in parallel from S3.

Two things matter more than raw filesystem bandwidth. The loader has to keep the GDS path alive
under tensor parallelism, which is why this PoC uses `instanttensor` rather than
`fastsafetensors`. And the checkpoint has to be striped across every OST before it is written,
because Lustre only applies striping to new files. Get the loader wrong and GDS is silently
disabled; get the striping wrong and you read from one OST instead of eight.

A bigger filesystem buys less than you would expect. Doubling from 8 to 16 OSTs improved weight
loading by only 10%. For a modern MoE model, 40% of the weight load goes on per-tensor work that
no amount of bandwidth will speed up.

To reproduce this, follow the Deployment steps above. The traps we hit on the way are in
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## Known Issues

See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Contributing

Contributions welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) and
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
