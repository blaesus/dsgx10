#!/bin/bash
# Runs every glm53 micro benchmark not yet marked DONE in the results file (idempotent and resumable: a preempted
# run is simply started again). GPU code: run it only through ds41f's ring.sh, e.g. from the repo root
#   OWNER=glm53-micro ds41f-4xgx10-switchless/engine/server/ring.sh glm53-4xgx10-switchless/micro/run.sh
# Binaries: ./build.sh (default ~/dsgb10-data/glm53-micro/bin). Results: ~/dsgb10-data/glm53-micro/results.tsv
set -u
cd "$(dirname "$0")"
BIN=${BIN:-$HOME/dsgb10-data/glm53-micro/bin}
OUT=${1:-$HOME/dsgb10-data/glm53-micro/results.tsv}
mkdir -p "$(dirname "$OUT")"
{
  echo "RUN	$(date '+%F %T %z')	sources sha256 $(cat common.cuh gemv.cu mma.cu | sha256sum | cut -c1-16)"
  echo "MEM	$(free -g | awk '/Mem:/ {print "total " $2 " GiB, used " $3 ", available " $7}')"
  echo "GPU	$(nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv,noheader 2> /dev/null)"
} >> "$OUT"
rc=0
for b in mma q4 tk3 s3 tk4 dense tk3m; do
  grep -q "^DONE	$b\$" "$OUT" && { echo "skip $b (done)"; continue; }
  echo "== $b $(date '+%T')"
  if [ "$b" = mma ]; then
    timeout -s TERM 240 "$BIN/mma" --out "$OUT" --dump "${OUT%.tsv}.mma4x.bin" || rc=$?
  else
    timeout -s TERM 240 "$BIN/gemv" --bench "$b" --out "$OUT" || rc=$?
  fi
done
echo "GPUEND	$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader 2> /dev/null)	$(date '+%F %T %z')" >> "$OUT"
exit $rc
