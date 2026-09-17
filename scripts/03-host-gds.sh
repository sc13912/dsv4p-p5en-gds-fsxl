#!/usr/bin/env bash
# GDS host setup — run ONCE per GPU node AFTER it is k8s-Ready, NOT in node bootstrap.
# Why not bootstrap: the EFA-over-LNet step pins each EFA interface to an LNet CPU partition, which
# requires LNet to load with its 16-partition CPU table. If anything loads LNet earlier (e.g. a
# `modprobe lustre` in preBootstrap), it comes up with the default partitions and the CPT pinning is
# rejected — only ~3 of 16 interfaces attach. Running this after the node is up lets the AWS script
# load LNet cleanly with the right table, so all 16 attach.
#
# Requires FSX_DNS and FSX_MOUNT in the environment.
set -euo pipefail
: "${FSX_DNS:?}" "${FSX_MOUNT:?}"

# Lustre client (lfs + mount.lustre) — install only; do NOT `modprobe lustre` here (leave LNet unloaded
# so the EFA script below can load it with the correct CPU-partition table).
dnf install -y lustre-client
dnf install -y make gcc "kernel-devel-$(uname -r)" || dnf install -y make gcc

# nvidia-fs (GDS kernel module) from source, pinned v2.29.4, NVFS_MAX_PEER_DEVS=128
# (default 64 is too small for p5en's 16 EFA + ENA + NVMe devices).
cd /tmp && curl -sSL https://github.com/NVIDIA/gds-nvidia-fs/archive/refs/tags/v2.29.4.tar.gz | tar -xz
cd /tmp/gds-nvidia-fs-2.29.4/src
NVIDIA_SRC_DIR="$(ls -d /usr/src/nvidia-open-*/kernel-open/nvidia | head -1)" \
  NVFS_MAX_PEER_DEVS=128 NVFS_MAX_PCI_DEPTH=16 make -j"$(nproc)"
rmmod nvidia_fs 2>/dev/null || true
insmod ./nvidia-fs.ko
echo 1 > /sys/module/nvidia_fs/parameters/rw_stats_enabled

# EFA over LNet for GDS (AWS's official script; loads LNet with the 16-partition CPU table).
cd /tmp && curl -sO https://docs.aws.amazon.com/fsx/latest/LustreGuide/samples/configure-efa-fsx-lustre-client.zip
unzip -oq configure-efa-fsx-lustre-client.zip
( cd configure-efa-fsx-lustre-client && bash ./setup.sh --optimized-for-gds )
echo "EFA NIDs: $(lnetctl net show | grep -c efa)"   # expect 16
