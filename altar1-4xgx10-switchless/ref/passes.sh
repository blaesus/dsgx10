#!/bin/bash
# The reference's passes, in order, each resumable (a rerun continues where a pass stopped; a finished pass is
# skipped), each retried up to 8 times after a kill. Run detached from the ref directory:
#
#   cd ref && OMP_NUM_THREADS=8 setsid nohup taskset -c 10-19 nice -n 19 ionice -c3 ./passes.sh > LOG 2>&1 &
#
# Environment: PY (the venv's python), CKPT (the checkpoint), GOLD (the output root), THREADS (8),
# PASSES (default: the list below, by name).
#   a1a-bf16, a1a-f32  short + the first 2,400 tokens of prose, every position saved: the layer checks' inputs
#                      (check_layer.py) and the first short golden data, in both orders
#   all-f32            the four sequences, --combine f32: the level-2 golden set (the contract's order)
#   all                the four sequences, --combine bf16 (transformers' order): the order-variance partner
#   mtp-nvfp4          the MTP layer replayed on all-f32's final hidden with its experts at NVFP4 (decision 5)
set -u
PY=${PY:-~/altar1-ref/venv/bin/python}
CKPT=${CKPT:-~/altar-1}
GOLD=${GOLD:-~/altar1-ref/golden}
THREADS=${THREADS:-8}
VER=${VER:-$CKPT/verify.log}
PASSES=${PASSES:-a1a-bf16 a1a-f32 all-f32 all mtp-nvfp4}
SEQS="tokens/short.tokens.json tokens/prose.tokens.json tokens/python.tokens.json tokens/agentic.tokens.json"
echo 1000 > /proc/self/oom_score_adj
for p in $PASSES; do
  case $p in
    a1a-bf16) args=(--combine bf16 --tokens tokens/short.tokens.json tokens/prose.tokens.json --max-tokens 2400 --all-positions-below 4096) ;;
    a1a-f32)  args=(--combine f32 --tokens tokens/short.tokens.json tokens/prose.tokens.json --max-tokens 2400 --all-positions-below 4096) ;;
    all-f32)  args=(--combine f32 --tokens $SEQS) ;;
    all)      args=(--combine bf16 --tokens $SEQS) ;;
    mtp-nvfp4) args=(--combine f32 --tokens $SEQS --mtp-from "$GOLD/all-f32" --mtp-experts nvfp4_rtn) ;;
    *) echo "unknown pass $p"; exit 2 ;;
  esac
  out=$GOLD/$p
  for try in 1 2 3 4 5 6 7 8; do
    if [ -f "$out/state/progress.json" ] && grep -q '"done"' "$out/state/progress.json"; then break; fi
    if [ "$p" = mtp-nvfp4 ] && [ -f "$out/mtp/rows_main.npy" ]; then break; fi
    echo "$(date -Is) pass $p try $try"
    "$PY" -I glmref.py --ckpt "$CKPT" --verified "$VER" --wait --threads "$THREADS" --out-dir "$out" "${args[@]}"
    rc=$?
    echo "$(date -Is) pass $p exit $rc"
    [ $rc -eq 0 ] && break
    sleep 120
  done
done
echo "$(date -Is) passes done"
