#!/bin/bash
# Run the torch reference (ref/torchref) in the SGLang image on this box.
#
#   golden/run_ref.sh tokens <make_tokens.py args...>        build a token file (paths under /work)
#   golden/run_ref.sh run <name> <tokens.json> [ref_run.py args...]
#   golden/run_ref.sh py <script under /work> [args...]         any script, e.g. golden/check_prose.py
#
# The checkpoint is the box's local copy (~/DeepSeek-V4.1-Flash-UNCENSORED-FP8, or rank 0's
# /data/models/...), overridable with CKPT. Outputs go to $OUT/<name> (default ~/dsgb10-data/golden).
# ALT_CKPT=<dir> also mounts a second checkpoint at /models-alt (for ref_run.py --alt-ckpt /models-alt).
# Long runs: launch detached, e.g.
#   setsid nohup golden/run_ref.sh run prose golden/prose.tokens.json > ~/dsgb10-data/golden/prose.log 2>&1 < /dev/null &
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
CKPT=${CKPT:-$HOME/DeepSeek-V4.1-Flash-UNCENSORED-FP8}
[ -d "$CKPT" ] || CKPT=/data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8
ALT=()
[ -n "${ALT_CKPT:-}" ] && ALT=(-v "$ALT_CKPT":/models-alt:ro)
OUT=${OUT:-$HOME/dsgb10-data/golden}
mkdir -p "$OUT"

run() {
  docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCHREF_ACT="${TORCHREF_ACT:-}" \
    -v "$CKPT":/models:ro "${ALT[@]}" -v "$HERE":/work -v "$OUT":/out -w /work/ref/torchref \
    --entrypoint /opt/sglang/bin/python3 dsv41-4x-spark:canary-roce "$@"
}

case "${1:-}" in
  tokens) shift; run make_tokens.py "$@" ;;
  run) name=$2; tokens=$3; shift 3; run ref_run.py --ckpt /models --tokens "/work/$tokens" --out-dir "/out/$name" "$@" ;;
  py) shift; run "$@" ;;
  *) echo "usage: $0 tokens <args> | run <name> <tokens.json> [ref_run.py args] | py <script> [args]" >&2; exit 2 ;;
esac
