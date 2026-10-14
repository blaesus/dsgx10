# The ring lock for this configuration's scripts (ring.sh, service.sh; docs/serving.md): the DeepSeek engine's
# ringlock.sh (its docs/engine.md, "Borrowing the ring") on the same lock, the fleet's RING_LOCK, which the DeepSeek
# service, its jobs, this configuration's service and its jobs all take: one ring, one lock. What differs: the
# fleet file is this configuration's (the same boxes, its own directories), and a preempted job's ranks are
# stopped by both engines' names (dsgb10_rank, glm53_rank).
# Every job on the ring holds rank 0's ring lock (the fleet file's RING_LOCK; RING_LOCK in the
# environment: a test's own); ring.owner says whose it is. The service holds it while
# it serves and lends it to a job that waits for it, once no request has come for a while. A job that
# waits registers a want, ring.wants/<pid>, and holds that file flock'd while it waits: a want whose lock
# is free is stale. Jobs take the ring in the order of their wants: one whose want is younger than a
# live one waits for it, even when it finds the lock free (ring_queued). While requests wait during a
# lend, the service's supervisor holds ring.reclaim flock'd: no job takes the ring before the service
# has it back, and the job on it is preempted (ring_run). Requests first.
# A driver that runs ring.sh part after part holds a want of its own across them, ring.wants/<pid>.held
# (ring_want_hold): while the ring is lent the service does not take it back in the gaps between its
# parts. It neither makes the service lend nor gives the driver a place in the queue.
FLEET=${FLEET:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/fleet/homelab.conf}
. "$(dirname "${BASH_SOURCE[0]}")/../../../ds41f-4xgx10-switchless/engine/server/fleet.sh"  # the boxes, their directories
fleet_load "$FLEET" || exit 2
FLEET_LOCK=$(fleet_local "$FLEET_RING_LOCK")
LOCK=${RING_LOCK:-$FLEET_LOCK}
RING_OWNER=${LOCK%.lock}.owner RING_WANTS=${LOCK%.lock}.wants RING_RECLAIM=${LOCK%.lock}.reclaim
RING_BOXES=("${FLEET_WORKERS[@]}")  # ranks 1-3
RING_WHO=job RING_NESTED=0

ring_reclaimed() {  # the service wants the ring back before any job: its supervisor holds the flag's lock
  flock -n -s -E 99 "$RING_RECLAIM" true 2> /dev/null
  [ $? = 99 ]
}

# A want's place in the queue, its second line: when it was made (ns since the epoch), then the pid.
ring_key() { echo "$(date +%s%N) $$"; }
ring_queued() {  # [want]: true while a live want older than this one (with none: any live want) waits
  local f k kp mine= mp=0
  [ -n "${1:-}" ] && read -r mine mp < <(sed -n 2p "$1" 2> /dev/null)
  for f in "$RING_WANTS"/*; do
    case $f in *.new | *.held | "$RING_WANTS/*" | "${1:-/}") continue ;; esac
    read -r k kp < <(sed -n 2p "$f" 2> /dev/null)
    [[ "$k$kp" =~ ^[0-9]+$ ]] || continue  # an older ring.sh's want: no place in the queue
    [ -z "$mine" ] || ((k < mine || (k == mine && kp < mp))) || continue
    flock -n -s -E 99 "$f" true 2> /dev/null
    [ $? = 99 ] && return 0  # its job waits (a free lock: the job is gone)
  done
  return 1
}

ring_take() {  # what: the ring lock on fd 9, waiting up to LOCK_WAIT s (default 7200); 1 if it gave up
  local t0=$SECONDS want=$RING_WANTS/$$ who=${1%% *}
  RING_WHO=$who
  # a job inside a job (ring.sh around a script that runs run4.sh): the lock came with fd 9
  if [ -n "${RING_HOLDER:-}" ] && [ /proc/self/fd/9 -ef "$LOCK" ] && flock -n 9; then
    RING_NESTED=1
    return 0
  fi
  exec 9>> "$LOCK"
  if ! ring_reclaimed && ! ring_queued && flock -n 9; then return 0; fi
  if ring_queued; then
    echo "$who: jobs wait for the ring before this one ($(cat "$RING_OWNER" 2> /dev/null)); waiting up to ${LOCK_WAIT:-7200} s" >&2
  else
    echo "$who: the ring is busy ($(cat "$RING_OWNER" 2> /dev/null)); waiting up to ${LOCK_WAIT:-7200} s" >&2
  fi
  # the want: written under a temporary name, locked, then renamed, so that it is never seen unlocked
  if mkdir -p "$RING_WANTS" && exec 7> "$want.new" && flock 7 &&
    printf '%s pid %s %s: %s\n%s\n' "$(date '+%F %T')" $$ "${OWNER:-?}" "$1" "$(ring_key)" >&7 &&
    mv -f "$want.new" "$want"; then
    trap 'rm -f "$want" "$want.new"' EXIT
  else
    echo "$who: cannot register in $RING_WANTS, so the service will not lend the ring to this job" >&2
  fi
  while :; do
    if ! ring_reclaimed && ! ring_queued "$want" && flock -n 9; then
      ring_reclaimed || break
      flock -u 9  # the service began to reclaim the ring meanwhile: it goes first
    fi
    if ((SECONDS - t0 >= ${LOCK_WAIT:-7200})); then
      rm -f "$want" "$want.new"
      exec 7>&-
      echo "$who: gave up waiting for the ring" >&2
      return 1
    fi
    sleep 2
  done
  rm -f "$want" "$want.new"
  exec 7>&-
  trap - EXIT
}

# A driver of ring parts (a script that runs ring.sh part after part, sourcing this file): ring_want_hold
# "what" holds a want, ring.wants/<pid>.held, until ring_want_drop or the driver's end (killed too: a
# watcher removes it within 2 s). The service counts it only to stay lent: it does not take the ring back
# while the driver sleeps between parts. Each part still waits in turn with its own want, and a job that
# waits during the gap goes first (leave a gap of a few seconds between parts). Drop it before anything
# long that does not use the ring: while it is held, the ring stays lent, and idle, between parts.
ring_want_hold() {
  local want=$RING_WANTS/$$.held me=$$
  [ -z "${RING_HELD:-}" ] || return 0
  mkdir -p "$RING_WANTS" || return 1
  (
    exec 3>&- 4>&- 5>&- 6>&- 8>&- 9>&-  # none of the driver's locks: only the want's
    exec 7> "$want.new" && flock 7 || exit 1
    printf '%s pid %s %s: %s (held between its parts)\n%s\n' "$(date '+%F %T')" "$me" "${OWNER:-?}" "$1" "$(ring_key)" >&7
    mv -f "$want.new" "$want" || exit 1
    while kill -0 "$me" 2> /dev/null && [ -e "$want" ]; do sleep 2; done
    rm -f "$want"
  ) > /dev/null 2>&1 < /dev/null &
  RING_HELD=$!
  local i
  for ((i = 0; i < 50; i++)); do  # registered before this returns
    [ -e "$want" ] && return 0
    kill -0 "$RING_HELD" 2> /dev/null || break
    sleep 0.1
  done
  echo "ring_want_hold: cannot register in $RING_WANTS" >&2
  return 1
}
ring_want_drop() {
  rm -f "$RING_WANTS/$$.held"  # its watcher ends within 2 s, and its lock with it
  RING_HELD=
}

# Preemption: a job on a lent ring is stopped as soon as requests wait for the service (its supervisor
# holds ring.reclaim): TERM to the job's process group, its ranks stopped on every box, KILL after
# PREEMPT_GRACE s (10), and the job exits 76, to be run again later. PREEMPT=0: it runs to its end.
RING_RANKS='^(dsgb10_rank|glm53_rank)$'  # both engines' ranks
ring_reap() {  # shell text: wait up to $1 s for the ranks to exit by themselves, then pkill, 10 s more, then kill -9
  echo "for i in \$(seq $((2 * $1))); do pgrep '$RING_RANKS' > /dev/null || exit 0; sleep 0.5; done; pkill '$RING_RANKS'; for i in \$(seq 20); do pgrep '$RING_RANKS' > /dev/null || exit 0; sleep 0.5; done; pkill -9 '$RING_RANKS'"
}
ring_stop_ranks() {  # the order of run4.sh stop: rank 0 first, then whatever is left on every box
  local w
  pkill "$RING_RANKS"
  bash -c "$(ring_reap 10)" &
  for w in "${RING_BOXES[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$w" "$(ring_reap 5)" &
  done
  wait
}
ring_watch() {  # job pid, its process group: 76 once it stopped the job for the service; 0 when the job ended
  local i
  while kill -0 "$1" 2> /dev/null; do
    if ring_reclaimed; then
      echo "$(date '+%F %T') $RING_WHO: preempted: requests wait for the service; stopping the job (TERM, KILL after ${PREEMPT_GRACE:-10} s)" >&2
      kill -TERM -- "-$1" 2> /dev/null
      # while the ring is lent, every rank on the boxes is the job's; a test's own lock has none
      if [ "$LOCK" -ef "$FLEET_LOCK" ]; then ring_stop_ranks & fi
      for ((i = 0; i < 2 * ${PREEMPT_GRACE:-10}; i++)); do
        kill -0 -- "-$1" 2> /dev/null || break
        sleep 0.5
      done
      kill -KILL -- "-$1" 2> /dev/null
      wait  # the rank stop: see ring_run
      echo "$(date '+%F %T') $RING_WHO: stopped; exit 76 (preempted: run it again later)" >&2
      return 76
    fi
    sleep 1
  done
  return 0
}
ring_run() {  # command...: runs it holding the ring; its status, or 76 if the service took the ring back
  if [ "$RING_NESTED" = 1 ]; then  # the outer job's watcher stops this one with it
    "$@"
    return
  fi
  export RING_HOLDER=$$
  if [ "${PREEMPT:-1}" = 0 ]; then
    echo "$RING_WHO: PREEMPT=0: not preempted; requests that come meanwhile wait for its end" >&2
    "$@"
    return
  fi
  local job watch rc
  [ -t 0 ] && exec < /dev/null  # the command runs as a background process group: no terminal input
  set -m                        # so that it gets a process group of its own: the job, to stop
  "$@" &
  job=$!
  set +m
  ring_watch "$job" &
  watch=$!
  trap 'kill -INT -- -$job 2> /dev/null' INT
  trap 'kill -TERM -- -$job 2> /dev/null' TERM
  trap 'kill -HUP -- -$job 2> /dev/null' HUP
  until
    wait "$job"
    rc=$?
    ! kill -0 "$job" 2> /dev/null  # a signal ends the wait early; the job goes on, and so does the wait
  do :; done
  trap - INT TERM HUP
  # This shell and the watcher hold the lock (fd 9) until the watcher returns, and the watcher returns
  # only once its rank stop is over: so a preempted job's reaps (both engines' ranks on every box) are
  # done before the service can retake the ring, and never reach the engine it starts next. Keep it so.
  wait "$watch" || rc=$?
  return "$rc"
}
