#!/bin/bash
# The routed-expert kernels' GPU checks and benches, resumable: every step not yet marked DONE in the results
# file runs, so a preempted job (ring.sh exit 76) is resumed by starting it again. GPU code: only through
# ds41f's ring.sh, from the repo root:
#   OWNER=glm53-experts ds41f-4xgx10-switchless/engine/server/ring.sh glm53-4xgx10-switchless/engine/experts/run.sh [step...]
# Steps: test (cpu deq route decode prefill), bw, d-q4 d-s3 d-mix (decode sweeps), p-q4 p-s3 p-mix (prefill),
# f-q4 f-s3 (the FP4-activation prefill), pd-q4 (the gate/up kernel's parts switched off), py-q4 py-s3 (the
# chained down's L2 budgets), x<tag>-<fmt> (the prefill bench with PF_ARGS), r-<dir> (E1: test_experts --real
# REAL_ROOT/<dir>, s3x2_pack.py's dumps), u-<fmt> (E1: decode up variants UPS in order at UROWS). Default: all but pd, py, x, r, u. Binaries: BIN (default ~/dsgb10-data/glm53-experts/bin,
# installed by install.sh from the build with the sources' hash); results: OUT.
set -u
BIN=${BIN:-$HOME/dsgb10-data/glm53-experts/bin}
OUT=${OUT:-$HOME/dsgb10-data/glm53-experts/results.tsv}
mkdir -p "$(dirname "$OUT")"
{
  echo "RUN	$(date '+%F %T %z')	$(cat "$BIN/SOURCES" 2> /dev/null)"
  echo "MEM	$(free -g | awk '/Mem:/ {print "total " $2 " GiB, used " $3 ", available " $7}')"
  echo "GPU	$(nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv,noheader 2> /dev/null)"
} >> "$OUT"
steps=("$@")
[ ${#steps[@]} -gt 0 ] || steps=(test d-q4 d-s3 d-mix p-q4 p-s3 p-mix f-q4 f-s3)
rc=0
for s in "${steps[@]}"; do
  grep -q "^DONE	$s\$" "$OUT" && { echo "skip $s (done)"; continue; }
  echo "== $s $(date '+%T')"
  case $s in
    test) cmd=("$BIN/test_experts") ;;
    bw) cmd=("$BIN/bench_experts" --bw) ;;
    d-*) cmd=("$BIN/bench_experts" --decode "${s#d-}" --sweep) ;;
    p-*) cmd=("$BIN/bench_experts" --prefill "${s#p-}") ;;
    f-*) cmd=("$BIN/bench_experts" --prefill "${s#f-}" --fp4) ;;
    pd-*) cmd=("$BIN/bench_experts" --prefill "${s#pd-}" --chunks 8192 --up 0,1,2 --down 1 --cq 4 --side 0 --diag) ;;
    x*-*) read -ra xa <<< "${PF_ARGS:-}"; cmd=("$BIN/bench_experts" --prefill "${s#*-}" "${xa[@]}") ;;  # ad hoc: PF_ARGS, step x<tag>-<fmt>
    py-*) cmd=("$BIN/bench_experts" --prefill "${s#py-}" --down 1 --side 0 --yrkb 4096,8192,12288,16384,24576) ;;
    r-*) cmd=("$BIN/test_experts" --real "${REAL_ROOT:?REAL_ROOT: the s3x2_pack.py dumps}/${s#r-}") ;;  # E1: real S3x2 experts
    u-*) cmd=("$BIN/bench_experts" --decode "${s#u-}" --rows "${UROWS:-1,6,16}" --ups "${UPS:-0,6,6,0}") ;;  # up variants A B B A
    *) echo "unknown step $s"; rc=2; continue ;;
  esac
  if timeout -s TERM 420 "${cmd[@]}" --out "$OUT" > >(tee -a "${OUT%.tsv}.log") 2>&1; then
    echo "DONE	$s" >> "$OUT"
  else
    r=$?
    echo "FAILED	$s	rc $r	$(date '+%T')" >> "$OUT"
    rc=$r
  fi
done
echo "GPUEND	$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader 2> /dev/null)	$(date '+%F %T %z')" >> "$OUT"
exit $rc
