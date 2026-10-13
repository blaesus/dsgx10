#!/bin/bash
# A job for the ring in parts, each its own ring.sh call (so each part takes a borrowed window of its own):
#   ringparts.sh DIR NAME=COMMAND...       COMMAND: a command line, split at spaces (no quoting)
# The parts run in turn. A part that a request preempted (ring.sh's exit 76) or whose wait for the ring was
# given up (75) runs again, up to TRIES times (12), SLEEP s apart (30). Any other exit finishes it:
# DIR/parts_done.txt gets its name and exit, and running the job again skips it (delete the line to run it
# again). Inside a part, steps.sh skips what a preempted run finished. Progress: DIR/parts.log.
# Between two parts it waits GAP s (5), so that another job waiting for the ring can take it first (a part's
# ring.sh would otherwise win the lock again before the other waiters poll).
# Env: OWNER (attention), PREEMPT (ring.sh's; 0: requests wait for the part instead), RING (ring.sh,
# default this tree's), TRIES, SLEEP, GAP. Start it detached, then poll DIR/parts.log:
#   setsid nohup engine/attn/tools/ringparts.sh $J a="$J/a.sh" b="$J/b.sh 16384" > $J/drive.log 2>&1 < /dev/null &
set -u
[ $# -ge 2 ] || {
  sed -n '2,12p' "$0" >&2
  exit 2
}
D=$1
shift
RING=${RING:-$(cd "$(dirname "$0")/../../server" && pwd)/ring.sh}
P=$D/parts.log
DONE=$D/parts_done.txt
log() { echo "$(date '+%F %T') $*" >> "$P"; }
mkdir -p "$D"
ran=0  # a part has run: the next one waits GAP s first
for spec in "$@"; do
  name=${spec%%=*}
  read -r -a cmd <<< "${spec#*=}"
  if grep -q "^$name " "$DONE" 2> /dev/null; then
    log "part $name: finished before (exit $(grep "^$name " "$DONE" | tail -1 | cut -d' ' -f2)), skipped"
    continue
  fi
  [ $ran = 1 ] && sleep "${GAP:-5}"
  ran=1
  rc=0
  for try in $(seq "${TRIES:-12}"); do
    log "part $name, try $try: ${cmd[*]}"
    OWNER=${OWNER:-attention} "$RING" "${cmd[@]}"
    rc=$?
    log "part $name: exit $rc"
    [ $rc = 76 ] || [ $rc = 75 ] || break
    sleep "${SLEEP:-30}"
  done
  [ $rc = 76 ] || [ $rc = 75 ] || echo "$name $rc" >> "$DONE"
done
log "all parts done"
