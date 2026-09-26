#!/usr/bin/env bash
# Mount FSx and stripe across every OST. Run on the staging node BEFORE staging.
set -euo pipefail
: "${FSX_DNS:?}" "${FSX_MOUNT:?}" "${MODEL_DIR:?}"

mkdir -p /fsx
mountpoint -q /fsx || mount -t lustre -o noatime,flock "${FSX_DNS}@tcp:/${FSX_MOUNT}" /fsx
lfs osts /fsx | grep -c ACTIVE                      # expect 8 on the 38400 GiB filesystem

mkdir -p "$MODEL_DIR"
lfs setstripe -c -1 -S 16M "$MODEL_DIR"             # every new file spans all OSTs
lfs getstripe -d "$MODEL_DIR"
