#!/bin/bash
# The ring's GPU checks and measurements, resumable: every step not yet marked DONE in the results file runs, so a
# preempted job (ring.sh exit 76) is resumed by starting it again. One job, through the DeepSeek engine's ring.sh,
# from the repo root:
#   OWNER=glm53-comm ds41f-4xgx10-switchless/engine/server/ring.sh glm53-4xgx10-switchless/engine/comm/run.sh [step...]
# Steps: ring (ring_test: correct race integrity latency graph), test (coll_tests test), bench (coll_tests bench
# step prefill). Binaries: BIN (default ~/dsgb10-data/glm53-comm/bin, installed by install.sh with the sources'
# hash); results: OUT (TSV; the logs beside it).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$HOME/dsgb10-data/glm53-comm/bin}
OUT=${OUT:-$HOME/dsgb10-data/glm53-comm/results.tsv}
LOG=${OUT%.tsv}.log
mkdir -p "$(dirname "$OUT")"
SRC=$(cat "$BIN/SOURCES" 2> /dev/null)
echo "RUN	$(date '+%F %T %z')	$SRC" >> "$OUT"
steps=("$@")
[ ${#steps[@]} -gt 0 ] || steps=(ring test bench)
rc=0
for s in "${steps[@]}"; do
  grep -q "^DONE	$s	$SRC\$" "$OUT" && { echo "skip $s (done)"; continue; }
  echo "== $s $(date '+%T')" | tee -a "$LOG"
  case $s in
    ring) cmd=("$BIN/ring_test" --integrity 300000 correct race integrity latency graph) ;;
    test) cmd=("$BIN/coll_tests" test) ;;
    bench) cmd=("$BIN/coll_tests" --out "$OUT" --tag "$(date '+%FT%T')" bench step prefill) ;;
    *) echo "unknown step $s"; rc=2; continue ;;
  esac
  "$HERE/run4.sh" "${cmd[@]}" 2>&1 | tee -a "$LOG"
  r=${PIPESTATUS[0]}
  if [ "$r" = 0 ]; then
    echo "DONE	$s	$SRC" >> "$OUT"
  else
    echo "FAILED	$s	rc $r	$(date '+%T')" >> "$OUT"
    rc=$r
  fi
done
exit $rc
