#!/usr/bin/env bash
# Stage the checkpoint onto the striped FSx dir. Use curl, not huggingface_hub,
# which resets stripe_count to 1; curl inherits the dir's 16-OST default.
set -euo pipefail
: "${MODEL_REPO:?}" "${MODEL_DIR:?}" "${MODEL_BYTES:?}"
cd "$MODEL_DIR"; mountpoint -q /fsx || { echo "/fsx not mounted"; exit 1; }

# Download the top-level model files, 16 in parallel (curl resumes and retries per file).
curl -sL "https://huggingface.co/api/models/${MODEL_REPO}/tree/main?recursive=1" \
 | python3 -c 'import json,sys
for f in json.load(sys.stdin):
    p = f["path"]
    if "/" not in p and p.endswith((".safetensors",".json",".txt",".model",".py")):
        print(p)' \
 | xargs -P 16 -I@ curl -fL --retry 8 -C - -o "@" \
     "https://huggingface.co/${MODEL_REPO}/resolve/main/@?download=true"

# Verify total size and that the shards inherited the 16-OST stripe.
[ "$(du -cb *.safetensors | tail -1 | cut -f1)" = "$MODEL_BYTES" ] || { echo "size mismatch"; exit 1; }
lfs getstripe -c *.safetensors | sort -u    # expect a single value: 16
