#!/usr/bin/env bash
# Run the three loader arms one at a time and print each load time.
# Single node: each arm is deleted before the next is applied.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1   # run from repo root, so manifests/ resolves regardless of CWD
: "${IMAGE:?}" "${S3_BUCKET:?}" "${MODEL_NAME:?}"
render(){ envsubst '${IMAGE} ${S3_BUCKET} ${MODEL_NAME}' < "manifests/$1"; }  # runtime ${MODEL_PATH} etc. stay intact

run(){ # manifest  app-label  description
  echo "=== $3 ==="
  render "$1" | kubectl apply -f -
  kubectl wait --for=jsonpath='{.status.phase}'=Running pod -l "app=$2" --timeout=600s  # pod start incl. cold image pull; NOT the load
  kubectl logs -f -l "app=$2" | grep -m1 "Model loading took"                            # the load itself is untimed
  render "$1" | kubectl delete -f - --wait=true
}

run 00-serve-dsv4p-default.yaml dsv4p-default "FSx Lustre, default vLLM loader"
run 02-serve-dsv4p-s3.yaml      dsv4p-s3      "S3 + Run:AI streamer (concurrency 32)"
run 01-serve-dsv4p-gds.yaml     dsv4p-gds         "FSx Lustre + GDS (instanttensor CUFILE)"
