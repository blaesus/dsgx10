#!/bin/bash
# Run an engine/attn/tests generator in the canary-roce image.
# usage: engine/attn/tests/gen.sh <script.py> <out-subdir> [args...]
# Output lands in ~/dsgb10-data/attntests/<out-subdir>. CKPT overrides the checkpoint path.
set -e
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
SCRIPT=$1; SUB=$2; shift 2
CKPT=${CKPT:-/data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8}
OUT=$HOME/dsgb10-data/attntests
mkdir -p "$OUT/$SUB"
docker run --rm --gpus all --ipc=host --entrypoint bash \
  -v "$CKPT":/models:ro -v "$ROOT":/work:ro -v "$OUT":/out \
  dsv41-4x-spark:canary-roce -c "cd /work/engine/attn/tests && /opt/sglang/bin/python3 $SCRIPT --out /out/$SUB $* ; chown -R $(id -u):$(id -g) /out/$SUB"
