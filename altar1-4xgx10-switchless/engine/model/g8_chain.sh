#!/bin/bash
# A8's jobs back to back (docs/engine.md § 9), each its own ring job through the DeepSeek engine's ring.sh (so the
# service and the other owners can take the ring between them): g8_job.sh's JOBS (default w a b c d m k p f).
# Logs in ~/dsgb10-data/altar1-g8/chain. Stop it by TERM to its process group.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
RING=$HERE/../../../ds41f-4xgx10-switchless/engine/server/ring.sh
L=$HOME/dsgb10-data/altar1-g8/chain
mkdir -p "$L"
export OWNER=altar1-g8
for j in ${JOBS-w a b c d m k p f}; do
  echo "$(date +%T) job $j"
  "$RING" "$HERE/g8_job.sh" "$j" > "$L/job-$j.$(date +%H%M%S).log" 2>&1
  echo "$(date +%T) job $j exit $?"
done
