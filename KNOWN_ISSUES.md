# Known Issues and Lessons Learned

Issues and lessons from building this PoC.

## nvidia-fs Build (NVFS_MAX_PEER_DEVS)

The EKS AL2023 GPU AMI ships no nvidia-fs at all, so `cuFileDriverOpen` returns error 5001 until
`scripts/03-host-gds.sh` builds and loads it. Two wrinkles make that build fiddly. First, nvidia-fs
needs `nv-p2p.h` from the NVIDIA **driver** sources, and the AMI deletes those once it has built
the driver. The script gets them back from the driver RPM, which is still cached under
`/opt/nvidia`. Second, the build must set `NVFS_MAX_PEER_DEVS=128`; the default of 64 is too small
for p5en's EFA + ENA + NVMe devices, and it is a compile-time array bound with no runtime
equivalent, so a packaged module built at 64 cannot be made to work.

FSx Lustre GDS is gated by an allowlist in AWS's `configure-efa-fsx-lustre-client.py`, currently
p5.48xlarge, p5e.48xlarge, p5en.48xlarge, p6-b200.48xlarge and p6-b300.48xlarge — check it before
reserving instances, since the script hard-fails on anything else.

**Why post-Ready scripts, not `preBootstrapCommands`:** the EFA-over-LNet step pins each of the 16 EFA
interfaces to an LNet CPU partition, which requires LNet to load with its 16-partition CPU table.
If bootstrap loads LNet early (e.g. a `modprobe lustre`), it comes up with the default partitions
and the CPT pinning is rejected — only ~3 of 16 attach (→ ~3× slower loads).

The NVIDIA GPU Operator's `gds.enabled=true` is the usual way to install nvidia-fs on Kubernetes,
but it cannot be used here, for three reasons. GDS ships only as a sidecar of the operator's
driver DaemonSet, and that DaemonSet is never created when `driver.enabled=false`, which is the
setting AWS requires on this AMI because the driver is already installed. No `nvidia-fs` container
image is published for Amazon Linux. And the operator documents GDS support for local NVMe and
remote NFS only, so it would not set up the FSx-Lustre-over-EFA path anyway. In production you
would fold `scripts/03-host-gds.sh` into a custom AMI or a DaemonSet rather than leave it a manual
step.

## cufile.json Pinned Memory / execution block

A minimal `cufile.json` (no `execution` block) leaves `parallel_io=false` and
`max_io_threads=0` — ~1.4× slower GDS loads. The tuned config is baked into the image at
`/etc/cufile.json` (`libcufile`'s default path); verify the active config with `gdscheck -p`,
not by reading the JSON.

## Why the pod runs `privileged`

The serve pod runs `privileged: true` (plus `IPC_LOCK`, which cuFile needs to pin memory).
cuFile requires the host's `/dev/nvidia-fs*` char devices. On Kubernetes, mounting those device
nodes into a pod makes them *visible* but does not grant the device-cgroup permission to *open* them —
`cuFileDriverOpen` returns `5001` non-privileged even with all 16 devices mounted.
Kubernetes has no pod-spec equivalent without a **device plugin**, so this PoC uses a `privileged` pod.
A potential least-privilege fix would be a device plugin that exposes `/dev/nvidia-fs*`.

## Lustre Stripe Size

`lfs setstripe -c -1 -S 16M` on a directory only sets the default for *new* files;
set it before staging. `huggingface_hub` writes `stripe_count=1` in every mode, so
stage with `curl` to have files born striped. If you expand an existing filesystem
(8→16 OST), existing files keep their old layout — `lfs migrate -c -1 -S 16M` each.

## FP8 Quantization Pairing

`--kv-cache-dtype fp8` alone can crash; pair it with the model's attention config
(prefill query quantization + a supported MLA prefill backend) as the checkpoint
requires.

## AWS `setup.sh` Exits Non-Zero After Succeeding

`configure-efa-fsx-lustre-client/setup.sh` ends with `systemctl enable --now`, which on a
first boot prints `Job for configure-efa-fsx-lustre-client.service canceled` and returns
non-zero even though the unit succeeded — systemd supersedes the `--now` start job with the
one from `enable`'s own dependency chain. `scripts/03-host-gds.sh` therefore ignores its exit
status and checks real state instead (`systemctl is-active`, `/dev/nvidia-fs*`, EFA NID count).
Do not "tidy away" the `|| true` — without it the script aborts before GDS is usable, and without
the state checks a node with 16 EFA NIDs but no `nvidia_fs` looks healthy while cuFile silently
falls back. It leaves `nvidia_fs` alone: AWS's sample never references nvidia at all.

## EFA silently falls back to TCP without an egress rule naming the FSx SG

Symptom: everything reports success and throughput is ~5% of what the filesystem provisions. The
mount works, `lfs df` lists every OST, AWS's configurator prints "Successfully added all EFA
interfaces", `lnetctl net show` lists all 16 EFA NIDs `up`, and weight loading still takes 223 s
instead of 35 s. The DMA phase runs at 4.25 GB/s rather than 42.5 GB/s. The only evidence is in
`dmesg`:

```
LNetError: kefalnd_tx_complete() Device[rdmapXXs0] QP[0] received TX[CONN_PROBE]
           completion with err. opcode[0] status[21] vendor[15] peer_ni[...@efa]
```

`vendor[15]` is `EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE` — "never received a response".
The packets are being dropped by the VPC, and Lustre quietly uses the single ENA interface instead.
It cannot self-heal: `kefalnd` does not report failures to LNet's health framework, so the EFA
interface stays at full health while carrying nothing.

Cause: **EFA is not authorised by CIDR rules.** Per
[the FSx security-group guidance](https://docs.aws.amazon.com/fsx/latest/LustreGuide/limit-access-security-groups.html),
"CIDR-based rules, including 0.0.0.0/0, do not satisfy EFA requirements even if they allow all
traffic on all ports. You must explicitly specify a security group ID." An EKS node SG has
`egress -1 -> 0.0.0.0/0` by default, which looks permissive and authorises nothing for EFA.

Step 5 creates all four required rules, so following it avoids this entirely:

| security group | direction | must name |
|---|---|---|
| FSx SG | ingress + egress | itself, and the cluster SG by ID |
| cluster SG | ingress + egress | the FSx SG by ID |

The easy mistake is adding three of the four and leaving the cluster SG's **egress** to the FSx SG
to `0.0.0.0/0`. Measured on the same node minutes apart: 568 MB/s single-stream and 2.08 GB/s across
8 streams without the rule, 3.9 GB/s and 34.36 GB/s with it.

Verify by counting traffic, never by counting interfaces:
```bash
lnetctl net show -v | awk '/net type: /{t=$4} /recv_count:/{a[t]+=$2} END{for(k in a) print k, a[k]}'
```
The `efa` count must be non-zero and rising during a read. 16 NIDs can be present and `up` while
every byte travels over TCP.
