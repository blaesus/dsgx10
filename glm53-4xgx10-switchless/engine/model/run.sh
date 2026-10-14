#!/bin/bash
# The model's GPU checks and measurements on the ring, resumable: every step not yet marked DONE in the results file
# runs, so a preempted job (ring.sh exit 76) is resumed by starting it again. One job, through the DeepSeek engine's
# ring.sh, from the repo root:
#   OWNER=glm53-model ds41f-4xgx10-switchless/engine/server/ring.sh glm53-4xgx10-switchless/engine/model/run.sh [step...]
# Steps (synthetic weights in the real layouts): check, bench, mtp (bench with one draft), l2pf, prefill, refhash,
# ab (the speed levers ABBASE, default none, against AB, default all, A/B/B/A at ABROWS: time and output bits),
# trace (graph steps at TROWS, default 1,6, under nsys on every rank: run/nsys/glm_run.<rank>.nsys-rep),
# pfab (prefill A B B A of the lever sets PFA, default 77: attn's gemm() and gemm_bf16() unpipelined, against PFB,
# default f7: the same with the collectives overlapped (kTunePfOverlap; T4 compared 17, the 16-row GEMVs, with 77)
# for chunks of PFCHUNKS rows at positions PFAT, default 2600, 8192, 8192 rows at 0, 0, 24576 (a "chunk" over 8192: a
# whole prompt in chunks); time and output bits; one MTP draft (PFDRAFTS) so that the MTP cache carries the final rows), pftrace (one chunk of
# PFCHUNKS, default 8192, at PFAT, default 24576, under nsys on every rank; tools/step_profile.py --prefill).
# Binaries: BIN (default ~/dsgb10-data/glm53-model/bin, installed by install.sh with the sources' hash); results:
# OUT (TSV; the log beside it). EXTRA: more glm_run options for every step.
# weights: every rank loads its fleet rank file to the GPU with the CRCs checked (glw_load --verify-crc), and rank 0
# runs glw_kcheck --gpu on layers 0, 3, 10, 78 of its file (the GEMVs on the file's bytes against a double reference).
# tf-prefill, tf-decode: the real rank files (WEIGHTS, "%d" the rank; each box its own copy) with synthetic routed
# experts while the pools are not filled, teacher-forced over the golden run's sequence (GOLDEN, SEQ, TOKENS), dumps
# into DATA/tf/SEQ-{prefill,decode}TFTAG; compare with tools/compare_golden.py. tfsyn-prefill, tfsyn-decode: the same on
# synthetic weights (DATA/tf/syn-SEQ-...TFTAG).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
DATA_NAME=${DATA_NAME:-glm53-model}  # the data tree under ~/dsgb10-data (install.sh, run4.sh read it too)
export DATA_NAME
BIN=${BIN:-$HOME/dsgb10-data/$DATA_NAME/bin}
OUT=${OUT:-$HOME/dsgb10-data/$DATA_NAME/results.tsv}
LOG=${OUT%.tsv}.log
EXTRA=${EXTRA:-}
DATA=$HOME/dsgb10-data/$DATA_NAME
WEIGHTS=${WEIGHTS:-fleet}
SEQ=${SEQ:-short}
GDATA=$HOME/dsgb10-data/glm53-model  # the golden run's tokens and rows
TOKENS=${TOKENS:-$GDATA/$SEQ.tokens.json}
ROWS=${ROWS:-$GDATA/${SEQ}_rows.npy}
mkdir -p "$(dirname "$OUT")"
SRC=$(cat "$BIN/SOURCES" 2> /dev/null)
echo "RUN	$(date '+%F %T %z')	$SRC" >> "$OUT"
weights_step() {  # on the ring: the loads on every box, then the GPU kernel check on rank 0
  local bin=$1 rc=0 r=1 w p pids=()
  . "$HERE/../../../ds41f-4xgx10-switchless/engine/server/fleet.sh"
  fleet_load "$HERE/../../fleet/homelab.conf" || return 2
  for w in "${FLEET_WORKERS[@]}"; do
    scp -q "$bin/glw_load" "$w:dsgb10-data/glm53-model/glw_load" || return 3
    ssh -o BatchMode=yes "$w" "timeout 300 ~/dsgb10-data/glm53-model/glw_load ~/glm53/weights/rank$r.glw --rank $r --verify-crc" &
    pids+=($!)
    r=$((r + 1))
  done
  timeout 300 "$bin/glw_load" /data/dsgb10/glm53/weights/rank0.glw --rank 0 --verify-crc || rc=1
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  timeout 300 "$bin/glw_kcheck" /data/dsgb10/glm53/weights/rank0.glw --rank 0 --gpu --layers 0,3,10,78 || rc=1
  return $rc
}

steps=("$@")
[ ${#steps[@]} -gt 0 ] || steps=(check bench prefill)
rc=0
G="$BIN/glm_run --synthetic --out $OUT"
for s in "${steps[@]}"; do
  grep -q "^DONE	$s	$SRC\$" "$OUT" && { echo "skip $s (done)"; continue; }
  echo "== $s $(date '+%T')" | tee -a "$LOG"
  T="--tag $s.$(date '+%FT%T')"
  case $s in
    check) cmd=($G $T --slots 6 --max-ctx 8192 --chunk 2048 --splits 1 $EXTRA check) ;;
    bench) cmd=($G $T --max-ctx 8192 --ctx 4096 $EXTRA bench) ;;
    mtp) cmd=($G $T --max-ctx 8192 --ctx 4096 --drafts 1 --rows 1,2,4 $EXTRA bench) ;;
    l2pf) cmd=($G $T --max-ctx 8192 --ctx 4096 --rows 1,6,16 $EXTRA l2pf) ;;
    prefill) cmd=($G $T --max-ctx 16384 --ctx 0 --chunk 8192 $EXTRA prefill) ;;
    ab) cmd=($G $T --max-ctx 8192 --ctx 4096 --rows ${ABROWS:-1,2,4,6,16} --ab ${AB:-ffffffff} --ab-base ${ABBASE:-0} $EXTRA ab) ;;
    trace) cmd=($G $T --max-ctx 8192 --ctx 4096 --rows ${TROWS:-1,6} $EXTRA trace) ;;
    pfab) cmd=($G $T --max-ctx 33792 --drafts ${PFDRAFTS:-1} --chunks ${PFCHUNKS:-2600,8192,8192} --at ${PFAT:-0,0,24576}
      --ab-base ${PFA:-77} --ab ${PFB:-f7} $EXTRA pfab) ;;
    pftrace) cmd=($G $T --max-ctx 33792 --chunks ${PFCHUNKS:-8192} --at ${PFAT:-24576} $EXTRA pftrace) ;;
    weights) cmd=() ;;
    tf-prefill | tf-decode)
      cmd=("$BIN/glm_run" --weights "$WEIGHTS" --fill-experts --out "$OUT" $T --max-ctx 4096 --tokens "$TOKENS"
        --sample-rows "$ROWS" --dump-layers 0,1,2,3,4,77 --via "${s#tf-}" --dir "$DATA/tf/$SEQ-${s#tf-}${TFTAG:-}" $EXTRA tf) ;;
    tfsyn-prefill | tfsyn-decode)  # the same on synthetic weights
      cmd=($G $T --max-ctx 4096 --tokens "$TOKENS" --sample-rows "$ROWS" --dump-layers 0,1,2,3,4,77 --via "${s#tfsyn-}"
        --dir "$DATA/tf/syn-$SEQ-${s#tfsyn-}${TFTAG:-}" $EXTRA tf) ;;
    refhash) cmd=($G $T --max-ctx 4096 --steps 64 $EXTRA refhash) ;;
    *) echo "unknown step $s"; rc=2; continue ;;
  esac
  if [ "$s" = weights ]; then weights_step "$BIN" 2>&1 | tee -a "$LOG"
  elif [ "$s" = trace ] || [ "$s" = pftrace ]; then NSYS=1 "$HERE/run4.sh" "${cmd[@]}" 2>&1 | tee -a "$LOG"
  else "$HERE/run4.sh" "${cmd[@]}" 2>&1 | tee -a "$LOG"; fi
  r=${PIPESTATUS[0]}
  if [ "$r" = 0 ]; then
    echo "DONE	$s	$SRC" >> "$OUT"
  else
    echo "FAILED	$s	rc $r	$(date '+%T')" >> "$OUT"
    rc=$r
  fi
done
exit $rc
