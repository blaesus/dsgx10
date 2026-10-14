#!/bin/bash
# The reference's passes, in order, each resumable (a rerun continues where a pass stopped; a finished pass is
# skipped), each retried up to 8 times after a kill. Run detached from the ref directory:
#
#   cd ref && OMP_NUM_THREADS=8 setsid nohup taskset -c 10-19 nice -n 19 ionice -c3 ./passes.sh > LOG 2>&1 &
#
# Environment: PY (the venv's python), CKPT (the checkpoint), GOLD (the output root), CHECKS (the root of
# check runs that are not golden sets), THREADS (8), PASSES (default: the list below, by name).
#   a1a-bf16, a1a-f32  short + the first 2,400 tokens of prose, every position saved: the layer checks' inputs
#                      (check_layer.py) and the first short golden data, in both orders (expanded attention)
#   all-f32            the four sequences, --combine f32, expanded attention: the level-2 golden set until
#                      DESIGN decision 12 (2026-10-14)
#   all                the four sequences, --combine bf16 (transformers' order): the order-variance partner
#   mtp-nvfp4          the MTP layer replayed on all-f32's final hidden with its experts at NVFP4 (decision 5)
#   a1a-f32-abs        a1a-f32 with --attn absorbed: the contract's order (decisions 8 and 12), short + prose2400
#   short-f32-abs      short alone, as a1a-f32-abs (into CHECKS): batch invariance of the absorbed form
#   all-f32-abs        all-f32 with --attn absorbed: the level-2 golden set in the contract's order
set -u
PY=${PY:-~/altar1-ref/venv/bin/python}
CKPT=${CKPT:-~/altar-1}
GOLD=${GOLD:-~/altar1-ref/golden}
CHECKS=${CHECKS:-~/altar1-ref/checks}
THREADS=${THREADS:-8}
START_KIB=${START_KIB:-16777216}
VER=${VER:-$CKPT/verify.log}
PASSES=${PASSES:-a1a-bf16 a1a-f32 all-f32 all mtp-nvfp4 a1a-f32-abs short-f32-abs all-f32-abs}
SEQS="tokens/short.tokens.json tokens/prose.tokens.json tokens/python.tokens.json tokens/agentic.tokens.json"
echo 1000 > /proc/self/oom_score_adj
for p in $PASSES; do
  out=$GOLD/$p
  case $p in
    a1a-bf16) args=(--combine bf16 --attn expanded --tokens tokens/short.tokens.json tokens/prose.tokens.json --max-tokens 2400 --all-positions-below 4096) ;;
    a1a-f32)  args=(--combine f32 --attn expanded --tokens tokens/short.tokens.json tokens/prose.tokens.json --max-tokens 2400 --all-positions-below 4096) ;;
    all-f32)  args=(--combine f32 --attn expanded --tokens $SEQS) ;;
    all)      args=(--combine bf16 --attn expanded --tokens $SEQS) ;;
    mtp-nvfp4) args=(--combine f32 --attn expanded --tokens $SEQS --mtp-from "$GOLD/all-f32" --mtp-experts nvfp4_rtn) ;;
    a1a-f32-abs) args=(--combine f32 --attn absorbed --tokens tokens/short.tokens.json tokens/prose.tokens.json --max-tokens 2400 --all-positions-below 4096) ;;
    short-f32-abs) args=(--combine f32 --attn absorbed --tokens tokens/short.tokens.json --max-tokens 2400 --all-positions-below 4096)
                   out=$CHECKS/$p ;;
    all-f32-abs) args=(--combine f32 --attn absorbed --tokens $SEQS) ;;
    *) echo "unknown pass $p"; exit 2 ;;
  esac
  for try in 1 2 3 4 5 6 7 8; do
    if [ -f "$out/state/progress.json" ] && grep -q '"done"' "$out/state/progress.json"; then break; fi
    if [ "$p" = mtp-nvfp4 ] && [ -f "$out/mtp/rows_main.npy" ]; then break; fi
    # rank 2 shares its memory with the engines' services: start a try only with MemAvailable >= START_KIB (the
    # 8 GiB floor plus a pass's peak), waiting otherwise; guard.sh stops a running try below the floor
    said=0
    while [ "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" -lt "$START_KIB" ]; do
      [ $said = 0 ] && echo "$(date -Is) pass $p waits for MemAvailable >= $START_KIB KiB"
      said=1
      sleep 60
    done
    echo "$(date -Is) pass $p try $try"
    "$PY" -I glmref.py --ckpt "$CKPT" --verified "$VER" --wait --threads "$THREADS" --out-dir "$out" "${args[@]}"
    rc=$?
    echo "$(date -Is) pass $p exit $rc"
    [ $rc -eq 0 ] && break
    sleep 120
  done
done
echo "$(date -Is) passes done"
