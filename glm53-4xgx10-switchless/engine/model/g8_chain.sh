#!/bin/bash
# G8's jobs back to back (docs/engine.md, "End to end on real weights"), each its own ring job through the DeepSeek
# engine's ring.sh (so the service can reclaim the ring between them): g8_job.sh's JOBS (default b c d), then the
# server's configurations CFGS (default plain k1 k2 k3 gate, tests/g8_job.sh). Logs in ~/dsgb10-data/glm53-g8/chain.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
RING=$HERE/../../../ds41f-4xgx10-switchless/engine/server/ring.sh
L=$HOME/dsgb10-data/glm53-g8/chain
mkdir -p "$L"
export OWNER=glm53-g8
for j in ${JOBS-b c d}; do
  echo "$(date +%T) job $j"
  if [ "$j" = b ]; then EXTRA="--drafts 1" "$RING" "$HERE/g8_job.sh" b > "$L/job-$j.log" 2>&1
  else "$RING" "$HERE/g8_job.sh" "$j" > "$L/job-$j.log" 2>&1; fi
  echo "$(date +%T) job $j exit $?"
done
for c in ${CFGS-plain k1 k2 k3 gate}; do
  echo "$(date +%T) server $c"
  "$RING" "$HERE/../server/tests/g8_job.sh" "$c" > "$L/server-$c.log" 2>&1
  echo "$(date +%T) server $c exit $?"
done
