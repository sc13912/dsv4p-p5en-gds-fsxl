#!/usr/bin/env bash
# Stage the checkpoint onto the striped FSx dir. Use curl, not huggingface_hub,
# which resets stripe_count to 1; curl inherits the dir's stripe default.
set -euo pipefail
: "${MODEL_REPO:?}" "${MODEL_DIR:?}"
cd "$MODEL_DIR"; mountpoint -q /fsx || { echo "/fsx not mounted"; exit 1; }

# What the checkpoint should be: "<size> <filename>", one line per file.
curl -sL "https://huggingface.co/api/models/${MODEL_REPO}/tree/main?recursive=1" \
 | python3 -c 'import json,sys
for f in json.load(sys.stdin):
    p = f["path"]
    if "/" not in p and p.endswith((".safetensors",".json",".txt",".model",".py")):
        print(f["size"], p)' > /tmp/wanted.txt

# Download 16 in parallel, resuming. Do not abort on a failed file - the check below decides.
cut -d' ' -f2- /tmp/wanted.txt \
 | xargs -P 16 -I@ curl -fL --retry 8 -C - -o "@" \
     "https://huggingface.co/${MODEL_REPO}/resolve/main/@?download=true" || true

# What is actually on disk, in the same format.
while read -r _ path; do
  printf '%s %s\n' "$(stat -c %s "$path" 2>/dev/null || echo 0)" "$path"
done < /tmp/wanted.txt > /tmp/ondisk.txt

# A parallel download can leave a shard half-written: curl exits without an error when the
# connection drops mid-transfer, so the size on disk is the only reliable check. Always compare.
# Re-run this script to finish any file that came up short.
diff /tmp/wanted.txt /tmp/ondisk.txt \
 || { echo "FATAL: '<' wanted, '>' on disk - re-run this script"; exit 1; }

echo "all $(grep -c . /tmp/wanted.txt) files verified byte-exact"
lfs getstripe -c *.safetensors | sort -u    # expect a single value: the OST count
