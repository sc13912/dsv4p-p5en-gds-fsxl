#!/usr/bin/env bash
# Mount FSx and set 16-way striping. Run on the staging node BEFORE staging.
set -euo pipefail
: "${FSX_DNS:?}" "${FSX_MOUNT:?}" "${MODEL_DIR:?}"

mkdir -p /fsx
mountpoint -q /fsx || mount -t lustre -o noatime,flock "${FSX_DNS}@tcp:/${FSX_MOUNT}" /fsx
lfs osts /fsx | grep -c ACTIVE                      # want 16

mkdir -p "$MODEL_DIR"
lfs setstripe -c -1 -S 16M "$MODEL_DIR"             # every new file spans all OSTs
lfs getstripe -d "$MODEL_DIR"
