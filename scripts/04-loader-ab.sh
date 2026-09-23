#!/usr/bin/env bash
# Run the three loader arms one at a time and print each load time.
# Single node: each arm is deleted before the next is applied.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1   # run from repo root, so manifests/ resolves regardless of CWD
: "${IMAGE:?}" "${S3_BUCKET:?}" "${MODEL_NAME:?}" "${AWS_REGION:?}"
render(){ envsubst '${IMAGE} ${S3_BUCKET} ${MODEL_NAME} ${AWS_REGION}' < "manifests/$1"; }  # runtime ${MODEL_PATH} etc. stay intact

run(){ # manifest  app-label  description
  echo "=== $3 ==="
  render "$1" | kubectl apply -f -
  kubectl wait --for=jsonpath='{.status.phase}'=Running pod -l "app=$2" --timeout=600s  # pod start incl. cold image pull; NOT the load
  for i in $(seq 1 480); do   # wait for the API server, up to 40 min (the default loader needs ~29)
    kubectl logs -l "app=$2" --tail=-1 2>/dev/null | grep -q "Application startup complete" && break
    sleep 5
  done
  kubectl logs -l "app=$2" --tail=-1 > "$2.log"   # keep the entire log; never pipe a -f stream, it truncates
  grep "Model loading took" "$2.log" || echo "NOT READY after 40 min - check $2.log"
  render "$1" | kubectl delete -f - --wait=true
}

run 00-serve-dsv4p-default.yaml dsv4p-default "FSx Lustre, default vLLM loader"
run 02-serve-dsv4p-s3.yaml      dsv4p-s3      "S3 + Run:AI streamer (concurrency 32)"
run 01-serve-dsv4p-gds.yaml     dsv4p-gds     "FSx Lustre + GDS (instanttensor CUFILE)"
