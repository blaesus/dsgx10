#!/bin/bash
# The routed-expert kernels' GPU checks and benches, resumable: every step not yet marked DONE in the results
# file runs, so a preempted job (ring.sh exit 76) is resumed by starting it again. GPU code: only through
# ds41f's ring.sh, from the repo root:
#   OWNER=altar1-experts ds41f-4xgx10-switchless/engine/server/ring.sh altar1-4xgx10-switchless/engine/experts/run.sh [step...]
# Steps:
#   t-cpu t-deq t-route        host checks; GPU decode of every format's values; routing at 168
#   t-i4 t-b16 t-q4            decode and prefill tests on a synthetic layer of the format (q4: with a fused FP8 shared expert)
#   t-eq                       an INT4 layer = the BF16 layer of its dequantised values, decode and prefill, bit for bit
#   r-<dir>                    test_experts --real REAL_ROOT/<dir> (int4_pack.py dumps of real experts)
#   bw                         the streaming read
#   d-<fmt>, d2-<fmt>          decode sweep at ROWS (default 1,6,16); d2- repeats it (A/B in both orders: d-i4 d-b16 d2-b16 d2-i4)
#   u-<fmt>                    decode up variants UPS in order at ROWS (default 0,6,6,0: the default against prefill order, A B B A)
#   p-<fmt>                    prefill at chunks 1024, 4096, 8192, up 0/1, down 0/1
#   pd-<fmt>                   prefill kernel parts switched off (INT4)
#   sh                         the BF16 shared expert through attn's GEMV at ROWS, every split-K
#   shf-<fmt>                  the BF16 shared expert fused into the decode kernels of a <fmt> layer: bits against the
#                              GEMVs, time A B B A against GEMVs + experts, at ROWS
#   topk                       tools/probe_topk_ties.py on the GPU (torch.topk's order over 168 experts) with PY
# Binaries: BIN (default ~/dsgb10-data/altar1-experts/bin, installed by install.sh from the build with the sources'
# hash); results: OUT.
set -u
BIN=${BIN:-$HOME/dsgb10-data/altar1-experts/bin}
OUT=${OUT:-$HOME/dsgb10-data/altar1-experts/results.tsv}
ROWS=${ROWS:-1,6,16}
PY=${PY:-python3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}  # the tests' CPU references: at most 8 threads on a box that serves
TOOLS=$(cd "$(dirname "$0")/../../tools" && pwd)
mkdir -p "$(dirname "$OUT")"
{
  echo "RUN	$(date '+%F %T %z')	$(cat "$BIN/SOURCES" 2> /dev/null)"
  echo "MEM	$(free -g | awk '/Mem:/ {print "total " $2 " GiB, used " $3 ", available " $7}')"
  echo "GPU	$(nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv,noheader 2> /dev/null)"
} >> "$OUT"
steps=("$@")
[ ${#steps[@]} -gt 0 ] || steps=(t-cpu t-deq t-route t-i4 t-b16 t-eq bw d-i4 d-b16 d2-b16 d2-i4 p-i4 p-b16 sh)
rc=0
BUDGET=${BUDGET:-570}  # s: the job stays inside a 10-minute ring window; steps left over run when it is started again
t0=$SECONDS
for s in "${steps[@]}"; do
  grep -q "^DONE	$s\$" "$OUT" && { echo "skip $s (done)"; continue; }
  left=$((BUDGET - (SECONDS - t0)))
  [ $left -ge 30 ] || { echo "budget spent: $s and the rest left for the next run"; break; }
  echo "== $s $(date '+%T')"
  case $s in
    t-cpu) cmd=("$BIN/test_experts" --cpu) ;;
    t-deq) cmd=("$BIN/test_experts" deq) ;;
    t-route) cmd=("$BIN/test_experts" route) ;;
    t-eq) cmd=("$BIN/test_experts" equiv) ;;
    t-*) cmd=("$BIN/test_experts" --fmt "${s#t-}" decode prefill) ;;
    r-*) cmd=("$BIN/test_experts" --real "${REAL_ROOT:?REAL_ROOT: the int4_pack.py dumps}/${s#r-}") ;;
    bw) cmd=("$BIN/bench_experts" --bw) ;;
    d-* | d2-*) cmd=("$BIN/bench_experts" --decode "${s#*-}" --rows "$ROWS" --sweep) ;;
    u-*) cmd=("$BIN/bench_experts" --decode "${s#u-}" --rows "$ROWS" --ups "${UPS:-0,6,6,0}") ;;
    p-*) cmd=("$BIN/bench_experts" --prefill "${s#p-}") ;;
    pd-*) cmd=("$BIN/bench_experts" --prefill "${s#pd-}" --chunks 8192 --up 1 --down 1 --diag) ;;
    sh) cmd=("$BIN/bench_experts" --shared --rows "$ROWS") ;;
    shf-*) cmd=("$BIN/bench_experts" --shfused "${s#shf-}" --rows "$ROWS") ;;
    topk) cmd=("$PY" -I "$TOOLS/probe_topk_ties.py" --device cuda) ;;
    *) echo "unknown step $s"; rc=2; continue ;;
  esac
  [ "$s" = topk ] || cmd+=(--out "$OUT")
  case $s in t-* | r-*) cmd=(nice -n 19 "${cmd[@]}") ;; esac  # CPU references at nice 19
  if timeout -s TERM $((left < 420 ? left : 420)) "${cmd[@]}" > >(tee -a "${OUT%.tsv}.log") 2>&1; then
    echo "DONE	$s" >> "$OUT"
  else
    r=$?
    echo "FAILED	$s	rc $r	$(date '+%T')" >> "$OUT"
    rc=$r
  fi
done
echo "GPUEND	$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader 2> /dev/null)	$(date '+%F %T %z')" >> "$OUT"
exit $rc
