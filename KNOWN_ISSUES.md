# Known Issues

Known issues hit while building this PoC.

## nvidia-fs Build (NVFS_MAX_PEER_DEVS)

The EKS AL2023 GPU AMI ships no nvidia-fs at all, so `cuFileDriverOpen` returns error 5001 until
`scripts/03-host-gds.sh` builds and loads it. Building it on this AMI needs two extra steps.
First, nvidia-fs needs `nv-p2p.h` from the NVIDIA **driver** sources, and the AMI deletes those
once it has built the driver. The script gets them back from the driver RPM, which is still cached under
`/opt/nvidia`. Second, the build must set `NVFS_MAX_PEER_DEVS=128`. The default of 64 is too low
for the number of EFA, ENA and NVMe devices on a p5en. The limit is fixed when the module is
compiled and cannot be changed when it loads, so a prebuilt module compiled with the default will
not work.

FSx Lustre GDS is gated by an allowlist in AWS's `configure-efa-fsx-lustre-client.py`, currently
p5.48xlarge, p5e.48xlarge, p5en.48xlarge, p6-b200.48xlarge and p6-b300.48xlarge. Check it before
reserving instances, since the script hard-fails on anything else.

AWS's `configure-efa-fsx-lustre-client` script assigns each EFA interface to an LNet CPU partition.
The settings go into `/etc/modprobe.d`, and the kernel reads them only when the Lustre modules load.
So if the modules are already loaded, the new settings never take effect. This can happen if you
repeat Step 9. AWS's script won't complain, but only a few of the 16 EFA interfaces will attach,
and Lustre will run at a fraction of its bandwidth. That's why `scripts/03-host-gds.sh` unloads the
Lustre modules with `lustre_rmmod` before calling AWS's script. It also fails if fewer than 8
interfaces attach.

The NVIDIA GPU Operator's `gds.enabled=true` is the usual way to install nvidia-fs on Kubernetes,
but it cannot be used here, for three reasons:

1. GDS ships only as a sidecar of the operator's driver DaemonSet. That DaemonSet is never created
   when `driver.enabled=false`, which is the setting AWS requires on this AMI because the driver is
   already installed.
2. No `nvidia-fs` container image is published for Amazon Linux.
3. The operator documents GDS support for local NVMe and remote NFS only, so it would not set up
   the FSx for Lustre over EFA path anyway.

In production you would fold `scripts/03-host-gds.sh` into a custom AMI or a DaemonSet rather than
leave it as a manual step.

## cufile.json execution block

A minimal `cufile.json` (no `execution` block) leaves `parallel_io=false` and
`max_io_threads=0`, which made GDS loading about 1.4× slower on this workload (69.6 s against
44.6 s total model load). The tuned config is baked into the image at
`/etc/cufile.json` (`libcufile`'s default path); verify the active config with `gdscheck -p`,
not by reading the JSON.

## Why the pod runs `privileged`

The serve pod runs `privileged: true` (plus `IPC_LOCK`, which cuFile needs to pin memory).
cuFile requires the host's `/dev/nvidia-fs*` char devices. On Kubernetes, mounting those device
nodes into a pod makes them *visible* but does not grant the device-cgroup permission to *open* them:
`cuFileDriverOpen` returns `5001` in a non-privileged pod even with all 16 devices mounted.
Kubernetes has no pod-spec equivalent without a **device plugin**, so this PoC uses a `privileged` pod.
A potential least-privilege fix would be a device plugin that exposes `/dev/nvidia-fs*`.
The pod also needs host privileges to drop the page cache before each cold read.

## Lustre Stripe Size

`lfs setstripe -c -1 -S 16M` on a directory only sets the default for *new* files;
set it before staging. `huggingface_hub` writes `stripe_count=1` in every mode, so
stage with `curl` to have files born striped. If you expand an existing filesystem
(8→16 OST), existing files keep their old layout, so run `lfs migrate -c -1 -S 16M` on each.
