#!/bin/bash
# The engine/attn GPU checks and timings on rank 0's GPU, as one resumable job: each step not yet marked DONE in
# the results file runs (a preempted job is simply started again). GPU code: only through ds41f's ring.sh, e.g.
#   OWNER=altar1-attn ds41f-4xgx10-switchless/engine/server/ring.sh altar1-4xgx10-switchless/engine/attn/tools/run.sh
# Binaries: BIN (default ~/dsgb10-data/altar1-attn/build/attn, built beforehand with cmake -j4); golden data from
# tests/gen_ref.py in DATA (default ~/dsgb10-data/altar1-attn/data/{s300 (int4), s300b (bf16), s300f (fp8),
# s4600 (int4)}) and tools/real_ref.py (DATA/real). STEPS: the steps to run (default all). Results: $1 (default
# ~/dsgb10-data/altar1-attn/results.tsv), logs beside it.
set -u
D=$HOME/dsgb10-data/altar1-attn
BIN=${BIN:-$D/build/attn}
DATA=${DATA:-$D/data}
OUT=${1:-$D/results.tsv}
STEPS=${STEPS:-memcheck memcheck-b memcheck-e test300 test300b test300f test4600 real agemv apgemm bw}
SRC=$(cd "$(dirname "$0")/.." && cat attn.h attn_common.cuh gemv.cu gemm.cu prep.cu indexer.cu sparse_attn.cu tests/attn_tests.cu tests/bench_attn.cu | sha256sum | cut -c1-16)
mkdir -p "$(dirname "$OUT")"
{
  echo "RUN	$(date '+%F %T %z')	sources sha256 $SRC"
  echo "MEM	$(free -g | awk '/Mem:/ {print "total " $2 " GiB, used " $3 ", available " $7}')"
  echo "GPU	$(nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv,noheader 2> /dev/null)"
} >> "$OUT"
MC="/usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --error-exitcode 9"
rc=0
for s in $STEPS; do
  grep -q "^DONE	$s	$SRC\$" "$OUT" && { echo "skip $s (done)"; continue; }
  echo "== $s $(date '+%T')"
  LOG=${OUT%.tsv}.$s.log
  case $s in
    memcheck) timeout -s TERM 420 $MC "$BIN/attn_tests" "$DATA/s300" --only ADTCBFP > "$LOG" 2>&1 ;;
    memcheck-b) timeout -s TERM 300 $MC "$BIN/attn_tests" "$DATA/s300b" --only ATCBP > "$LOG" 2>&1 ;;
    memcheck-e) timeout -s TERM 300 $MC "$BIN/attn_tests" "$DATA/s4600" --only E > "$LOG" 2>&1 ;;
    test300) timeout -s TERM 240 "$BIN/attn_tests" "$DATA/s300" > "$LOG" 2>&1 ;;
    test300b) timeout -s TERM 240 "$BIN/attn_tests" "$DATA/s300b" > "$LOG" 2>&1 ;;
    test300f) timeout -s TERM 240 "$BIN/attn_tests" "$DATA/s300f" > "$LOG" 2>&1 ;;
    test4600) timeout -s TERM 300 "$BIN/attn_tests" "$DATA/s4600" > "$LOG" 2>&1 ;;
    real) timeout -s TERM 120 "$BIN/attn_tests" "$DATA/real" --only R > "$LOG" 2>&1 ;;
    *) timeout -s TERM 420 "$BIN/bench_attn" --only "$s" --out "$OUT" > "$LOG" 2>&1 ;;
  esac
  r=$?
  tail -3 "$LOG"
  if [ $r -eq 0 ]; then echo "DONE	$s	$SRC" >> "$OUT"; else echo "FAIL	$s	$r" >> "$OUT"; rc=$r; fi
done
echo "GPUEND	$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader 2> /dev/null)	$(date '+%F %T %z')" >> "$OUT"
exit $rc
