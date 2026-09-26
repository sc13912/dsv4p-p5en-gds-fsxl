# Known Issues

Genuine limitations hit while building this PoC.

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

AWS's `configure-efa-fsx-lustre-client` sets the LNet CPU-partition table through `/etc/modprobe.d`,
which the kernel reads only when `libcfs` loads, and the script does not unload an already-loaded
Lustre stack first. If Lustre is loaded before it runs, only a few EFA interfaces attach and nothing
reports an error. `scripts/03-host-gds.sh` runs `lustre_rmmod` immediately before it.

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

## AWS `setup.sh` Exits Non-Zero After Succeeding

`configure-efa-fsx-lustre-client/setup.sh` ends with `systemctl enable --now`, which on a
first boot prints `Job for configure-efa-fsx-lustre-client.service canceled` and returns
non-zero even though the unit succeeded — systemd supersedes the `--now` start job with the
one from `enable`'s own dependency chain. `scripts/03-host-gds.sh` therefore ignores its exit
status and checks real state instead (`systemctl is-active`, `/dev/nvidia-fs*`, EFA NID count).
Do not "tidy away" the `|| true` — without it the script aborts before GDS is usable, and without
the state checks a node with 16 EFA NIDs but no `nvidia_fs` looks healthy while cuFile silently
falls back. It leaves `nvidia_fs` alone: AWS's sample never references nvidia at all.
