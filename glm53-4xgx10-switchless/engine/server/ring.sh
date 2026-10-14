#!/bin/bash
# This configuration's copy of the DeepSeek engine's ring.sh (the same lock, ringlock.sh here).
# A command on the ring: takes the ring lock the way run4.sh does (ringlock.sh), so it borrows the ring
# from the service once the service has had no request for a while, and holds it until the command
# ends (the command inherits the lock). On a borrowed ring, requests that come for the service preempt
# it: the command's process group gets TERM, then KILL after PREEMPT_GRACE s (10), any ranks on the
# boxes are stopped, and ring.sh exits 76 (PREEMPT=0: never). For jobs that do not go through run4.sh:
# a bare flock(1) on the lock registers no want, and waits as long as the service serves
# (docs/engine.md, "Borrowing the ring"). Inside a job that holds the ring, it runs at once.
#   OWNER=me glm53-4xgx10-switchless/engine/server/ring.sh glm53-4xgx10-switchless/engine/server/tests/ring_job.sh
set -u
[ $# -gt 0 ] || {
  echo "usage: ring.sh <command> [args...]" >&2
  exit 2
}
. "$(dirname "$0")/ringlock.sh"
ring_take "ring.sh $*" || exit 75
[ "$RING_NESTED" = 1 ] || echo "$(date '+%F %T') pid $$ ${OWNER:-?}: ring.sh $*" > "$RING_OWNER"
ring_run "$@"
