#!/usr/bin/env bash
# GDS host setup — run once per GPU node after it has joined the cluster.
set -euo pipefail

dnf install -y lustre-client make gcc   # kernel headers already ship at /usr/src/kernels/$(uname -r)

# nvidia-fs (GDS kernel module) from source, pinned v2.29.4, NVFS_MAX_PEER_DEVS=128
# (default 64 is too small for p5en's 16 EFA + ENA + NVMe devices).
NVFS_SRC=/tmp/gds-nvidia-fs-2.29.4/src
cd /tmp && curl -sSL https://github.com/NVIDIA/gds-nvidia-fs/archive/refs/tags/v2.29.4.tar.gz | tar -xz
cd "$NVFS_SRC"
# nvidia-fs includes nv-p2p.h from the driver sources, which the AL2023 GPU AMI deletes after
# building the driver. The driver RPM is still cached on the AMI, so unpack it.
rpm2cpio /opt/nvidia/current/flavors/open/.rpms/kmod-nvidia-open-dkms-*.rpm | (cd / && cpio -idmu --quiet)
NVIDIA_SRC_DIR="$(ls -d /usr/src/nvidia-*/kernel-open/nvidia | head -1)"
[ -f "$NVIDIA_SRC_DIR/nv-p2p.h" ] || { echo "FATAL: nv-p2p.h not found (NVIDIA_SRC_DIR=$NVIDIA_SRC_DIR)"; exit 1; }
echo "NVIDIA_SRC_DIR=$NVIDIA_SRC_DIR"
NVIDIA_SRC_DIR="$NVIDIA_SRC_DIR" \
  NVFS_MAX_PEER_DEVS=128 NVFS_MAX_PCI_DEPTH=16 make -j"$(nproc)"
rmmod nvidia_fs 2>/dev/null || true
insmod ./nvidia-fs.ko

# Put Lustre's traffic onto the EFA interfaces for GDS, using AWS's own script. It writes libcfs
# CPU-partition options that only take effect when the module loads, so unload any Lustre/LNet
# stack first. Ignore its exit code: on a first boot the closing `systemctl enable --now` returns
# non-zero with "Job for ... canceled" even though the service started fine, so we check the unit
# state below instead. It leaves nvidia_fs untouched.
cd /tmp && curl -sO https://docs.aws.amazon.com/fsx/latest/LustreGuide/samples/configure-efa-fsx-lustre-client.zip
unzip -oq configure-efa-fsx-lustre-client.zip
lustre_rmmod 2>/dev/null || true
( cd configure-efa-fsx-lustre-client && bash ./setup.sh --optimized-for-gds ) || true
systemctl is-active --quiet configure-efa-fsx-lustre-client.service \
  || { echo "FATAL: configure-efa-fsx-lustre-client.service is not active"; exit 1; }

# Verify GDS is genuinely available.
echo 1 > /sys/module/nvidia_fs/parameters/rw_stats_enabled
NFS_DEVS="$(ls /dev/nvidia-fs* 2>/dev/null | wc -l)"
[ "$NFS_DEVS" -ge 1 ] || { echo "FATAL: nvidia-fs loaded but no /dev/nvidia-fs* device nodes"; exit 1; }

# Both must hold, or GDS is silently unavailable while the node still looks healthy.
EFA_NIDS="$(lnetctl net show | grep -c '@efa')"
echo "nvidia-fs devices: $NFS_DEVS    EFA NIDs: $EFA_NIDS"   # expect 16 and 16 on p5en.48xlarge
[ "$EFA_NIDS" -ge 8 ] || { echo "FATAL: only $EFA_NIDS EFA NIDs attached (expected 16)"; exit 1; }
