# Known Issues and Lessons Learned

Issues and lessons from building this PoC.

## nvidia-fs Build (NVFS_MAX_PEER_DEVS)

The EKS AL2023 GPU AMI ships nvidia-fs source but does not build or load it;
`cuFileDriverOpen` returns error 5001. The packaged build also uses
`NVFS_MAX_PEER_DEVS=64`, too small for p5en's 16 EFA + ENA + NVMe devices.

FSx Lustre GDS is supported only on p5 / p5e / p5en / p6-b200 — confirm the instance type is in AWS's
`configure-efa-fsx-lustre-client.py` allowlist before reserving instances.

**Why post-Ready scripts, not `preBootstrapCommands`:** the EFA-over-LNet step pins each of the 16 EFA
interfaces to an LNet CPU partition, which requires LNet to load with its 16-partition CPU table.
If bootstrap loads LNet early (e.g. a `modprobe lustre`), it comes up with the default partitions
and the CPT pinning is rejected — only ~3 of 16 attach (→ ~3× slower loads).

The NVIDIA GPU Operator (`gds.enabled=true`) is the usual way to install nvidia-fs on
Kubernetes, but it builds at the default `NVFS_MAX_PEER_DEVS=64` (no override) and targets
NVMe / NFS-over-RDMA — it does not set up the FSx-Lustre-over-EFA path (LNet-over-EFA, the
AWS `--optimized-for-gds` tuning, the mount). A production deployment would fold `scripts/03-host-gds.sh`
into a custom AMI or a DaemonSet rather than a manual step.

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
