#!/bin/bash
# Run the engine on all four boxes: ranks 1-3 detached on the workers, rank 0 here in the foreground
# (docs/engine.md). Every rank gets the same arguments. The boxes and their directories come from the
# fleet file (FLEET=FILE; default fleet/homelab.conf), which is installed beside the engine as fleet.conf
# on every box. Logs: LOG_DIR/dsgb10_rank.<rank>.log on each box (~/dsgb10/logs here); the workers' log
# tails are printed at the end.
#   engine/server/run4.sh --mode prefill --inject            # TIMEOUT=seconds (default 3600)
#   engine/server/run4.sh stop                               # stop every rank (rank 0 first)
#   WRAP="nsys profile ..." engine/server/run4.sh ...        # rank 0 under a wrapper (a profiler);
#                                                            # WRAP_RANK=r wraps rank r instead
# Runs take the ring lock (the fleet's RING_LOCK on rank 0, /data/dsgb10/ring.lock here; who holds it:
# ring.owner) and wait for it up to LOCK_WAIT seconds (default 7200): a second run would kill the first
# one's ranks and overwrite its binaries. Set OWNER to say who is running. The service lends the ring to
# a run that waits, once it has had no request for DSGB10_YIELD_IDLE_S (600 s by default; production
# 120 s) (ringlock.sh; docs/engine.md, "Borrowing the ring"), and takes it back as soon as requests
# wait: the run is preempted (its ranks stopped; exit 76) unless PREEMPT=0. Other big GPU jobs take the
# lock with ring.sh.
set -u
NAME=dsgb10_rank
BIN=${BIN:-$(dirname "$0")/../build/server/$NAME}
T=${TIMEOUT:-3600}
. "$(dirname "$0")/ringlock.sh"  # the fleet file (fleet.sh) and the ring lock
W1=$FLEET_RANK1_SSH W2=$FLEET_RANK2_SSH W3=$FLEET_RANK3_SSH  # ranks 1, 2, 3
# the binaries' and the logs' directories as written, for the workers' shells (a leading ~/ is theirs);
# here, expanded
BD=$FLEET_BIN_DIR LD=$FLEET_LOG_DIR
LBD=$(fleet_local "$BD") LLD=$(fleet_local "$LD")
reap() {  # wait up to $1 s for the ranks to exit by themselves, then pkill, 10 s more, then kill -9
  echo "for i in \$(seq $((2 * $1))); do pgrep -x $NAME > /dev/null || exit 0; sleep 0.5; done; pkill -x $NAME; for i in \$(seq 20); do pgrep -x $NAME > /dev/null || exit 0; sleep 0.5; done; pkill -9 -x $NAME"
}
if [ "${1:-}" = stop ]; then
  # the service's ranks are its supervisor's to stop (service.sh stop), not a job's
  if grep -q "^[0-9-]* [0-9:]* pid [0-9]* service: " "$RING_OWNER" 2> /dev/null && ! flock -n "$LOCK" true; then
    echo "run4.sh: the service holds the ring ($(cat "$RING_OWNER")): its ranks are not a job's to stop" >&2
    exit 1
  fi
  # Rank 0 first: a server on SIGTERM ends the request in flight and stops the other ranks (kCmdStop),
  # and all four leave the ring together. A worker killed first fails the others in the middle of that
  # ("socket read: peer closed", core dumps). Then whatever is left.
  pkill -x $NAME
  bash -c "$(reap 30)"
  ssh $W1 "$(reap 10)" &
  ssh $W2 "$(reap 10)" &
  ssh $W3 "$(reap 10)" &
  wait
  exit 0
fi
ring_take "run4.sh $*" || exit 75
[ "$RING_NESTED" = 1 ] || echo "$(date '+%F %T') pid $$ ${OWNER:-?}: run4.sh $*" > "$RING_OWNER"
ARGS="$*"
WR=${WRAP_RANK:-0}
wrap() { [ "$WR" = "$1" ] && echo "${WRAP:-}"; }
start() {  # host rank; a worker's timeout is a minute longer, so that rank 0's stops it (a server: cleanly)
  ssh "$1" "pkill -x $NAME; mkdir -p $LD; setsid nohup timeout $((T + 60)) $(wrap $2) $BD/$NAME --rank $2 $ARGS > $LD/$NAME.$2.log 2>&1 < /dev/null &"
}
run() {  # the run, under ring_run: stopped (exit 76) if the service takes the ring back
  # Stray ranks of an earlier job go first: workers outlive a failed rank 0 by up to their connect
  # timeout, and a copy onto their running binary fails ("text busy"), which left the workers running
  # the old binary with the new arguments.
  bash -c "$(reap 0)" &
  ssh $W1 "mkdir -p $BD $LD; $(reap 0)" &
  ssh $W2 "mkdir -p $BD $LD; $(reap 0)" &
  ssh $W3 "mkdir -p $BD $LD; $(reap 0)" &
  wait
  # the model files' page cache goes too: CUDA on GB10 allocates only from free pages (tools/evict_cache.py)
  EVICT=$(dirname "$0")/tools/evict_cache.py
  bash -c "python3 '$EVICT' $(fleet_model_files 0)" &
  ssh $W1 "python3 - $(fleet_model_files 1)" < "$EVICT" &
  ssh $W2 "python3 - $(fleet_model_files 2)" < "$EVICT" &
  ssh $W3 "python3 - $(fleet_model_files 3)" < "$EVICT" &
  wait
  # copies under a temporary name, then renamed (a rename works even onto a busy file); any failure
  # stops. The fleet file goes beside the binary on every box, where the engine reads it.
  mkdir -p "$LBD" "$LLD"
  cp "$BIN" "$LBD/$NAME.new" && cp "$FLEET" "$LBD/fleet.conf.new" &&
    mv -f "$LBD/$NAME.new" "$LBD/$NAME" && mv -f "$LBD/fleet.conf.new" "$LBD/fleet.conf" ||
    { echo "run4.sh: copying the binary failed here" >&2; exit 1; }
  for w in $W1 $W2 $W3; do
    scp -q "$BIN" "$w:$(fleet_rel "$BD")/$NAME.new" &&
      ssh $w "cat > $BD/fleet.conf.new && mv -f $BD/$NAME.new $BD/$NAME && mv -f $BD/fleet.conf.new $BD/fleet.conf" < "$FLEET" ||
      { echo "run4.sh: copying the binary to $w failed" >&2; exit 1; }
  done
  start $W1 1
  start $W2 2
  start $W3 3
  timeout "$T" $(wrap 0) "$LBD/$NAME" --rank 0 "$@" 2>&1 | tee "$LLD/$NAME.0.log"
  rc=${PIPESTATUS[0]}
  sleep 2
  echo "--- rank 1 (${FLEET_RANK1_NAME:-$W1})"; ssh $W1 "tail -n 3 $LD/$NAME.1.log"
  echo "--- rank 2 (${FLEET_RANK2_NAME:-$W2})"; ssh $W2 "tail -n 3 $LD/$NAME.2.log"
  echo "--- rank 3 (${FLEET_RANK3_NAME:-$W3})"; ssh $W3 "tail -n 3 $LD/$NAME.3.log"
  exit "$rc"
}
ring_run run "$@"
