#!/bin/bash
# The engine as a service (docs/engine.md, "Launch and stop"): the four ranks in serve mode and the
# front-end, dsfront, under a supervisor on rank 0 that holds the ring lock while it serves and restarts
# what fails, with bounded backoff. No systemd unit and no system configuration: the supervisor is a
# detached bash process, started and stopped by hand.
#   engine/server/service.sh start     # returns once it serves, or failed (START_WAIT s, default 900; 0: at once).
#                                      # dsfront answers at once. Behind a busy ring the start waits as during a
#                                      # lend: the engine starts once no job waits, or as soon as the job on the
#                                      # ring ends if a request waits (it is preempted unless PREEMPT=0)
#   engine/server/service.sh stop      # rank 0 first (it stops ranks 1-3), then the front-end; the ring is free
#   engine/server/service.sh status    # exit 0 serving, 1 starting / restarting / lent / not answering, 3 not running
#   engine/server/service.sh restart   # stop, then start (with this environment's settings, as start)
#   engine/server/service.sh forget    # the kept sessions gone on every box, and dsfront's saved table
#                                      # (not while it runs: stop --forget stops it first)
# Settings, from the environment of start (written to run/settings, which status and stop read):
#   DSGB10_ADDR, DSGB10_PORT  the front-end's addresses and port: 127.0.0.1 and 18893, a test port. One
#                             or more IPv4 addresses, separated by spaces or commas ("127.0.0.1 <private
#                             address>": loopback, and the box's address on a private network), each a
#                             specific one, never 0.0.0.0 (HTTP stays off public interfaces). Port 8888 is
#                             the production endpoint (M9, the owner's go-ahead on 2026-10-12): only with
#                             DSGB10_PROD=1
#   DSGB10_KEY_FILE           the API key, which dsfront reads from this file (mode 600):
#                             ~/dsgb10-data/dev-api-key. It is never on a command line
#   DSGB10_ENGINE_ARGS        the engine's arguments after --mode serve --socket <run/engine.sock>:
#                             --max-ctx 262144 (the rest are serve's defaults)
#   DSGB10_FRONT_ARGS         more of dsfront's arguments
#   DSGB10_NOTE               added to ring.owner: how long the service will hold the ring, say
#   DSGB10_BIN, DSGB10_FRONT_BIN  the binaries, copied at start (to the workers once the service has
#                             the ring), so a later build does not change a running service:
#                             engine/build/server/dsgb10_rank, frontend/target/release/dsfront
#   DSGB10_YIELD_IDLE_S (600) the ring is lent to a job that waits for it (run4.sh, ring.sh) once no
#                             request has come for this long, and comes back when the job is done
#                             (docs/engine.md, "Borrowing the ring")
#   DSGB10_HOLD_S (600)       while the engine is down (lent, or restarting) dsfront holds a request this
#                             long, then answers 503 with Retry-After
#   MIN_AVAIL_GIB (100)       a box with less MemAvailable than this after the page-cache eviction fails
#                             the engine's start (a rank needs ~90 GiB: an OOM can hang a box)
#   DSGB10_KEEP_GIB (4)       kept sessions (docs/engine.md): an engine stop (stop, restart, a lend)
#                             keeps the sessions for the next start: those in the disk tier, and those in
#                             device memory, the most recently used first, up to this many GiB written at
#                             the stop (0: nothing is kept). The engine runs with --disk-tag service, so
#                             that its files are the service's alone
# The boxes, from the environment of every command: FLEET, the fleet file (default fleet/homelab.conf;
# ringlock.sh): the workers (RANK<r>_SSH), the service's directory (SERVICE_DIR), each rank's files; it is
# installed beside the frozen engine as bin/fleet.conf on every box, where the engine reads it.
# Files: SERVICE_DIR/{bin,logs} (~/dsgb10/service here) on every box; on rank 0 also run/ (mode 700: the
# engine's socket, dsfront's admin socket, pids, state, settings). Logs: logs/service.log (the
# supervisor), rank.0.log and front.log on rank 0, rank.<r>.log on each worker; rotated at every (re)start
# and, on rank 0, once over LOG_MIB (64); KEEP (5). Kept sessions, mode 600 in directories of mode 700:
# rank<r>.service.spill and .index in the engine's spill directory on every box (the fleet's
# RANK<r>_SPILL: /data/dsgb10/spill on rank 0, ~/dsgb10/spill on the others), server.service.index (the
# sessions' token histories) beside rank 0's; and run/front.table, dsfront's session table (the
# conversations' text), saved at its stop; forget removes them.
# A test instance, from the environment of every command: DSGB10_MOCK=1 runs dsfront's mock engine in
# place of the four ranks, on rank 0 alone, and touches no rank, worker or page cache; it needs its own
# DSGB10_HOME (in place of SERVICE_DIR), DSGB10_PORT and RING_LOCK (tests/yield_test.sh).
set -u
NAME=dsgb10_rank
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
SRC=$(cd "$(dirname "$0")/../.." && pwd)
. "$(dirname "$SELF")/ringlock.sh"  # the fleet (FLEET_*); LOCK (RING_LOCK), RING_OWNER, RING_WANTS, RING_RECLAIM
H=${DSGB10_HOME:-$(fleet_local "$FLEET_SERVICE_DIR")}
RH=${H#"$HOME"/}  # here, and on the workers (relative to their home)
RUN=$H/run LOGS=$H/logs
SOCK=$RUN/engine.sock ADMIN=$RUN/front.admin
OWNERF=$RING_OWNER
WS=("${FLEET_WORKERS[@]}")  # ranks 1-3
MOCK=${DSGB10_MOCK:-0}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
KEEP=${KEEP:-5} LOG_MIB=${LOG_MIB:-64}
READY_S=${READY_S:-600}           # rank 0 must be listening this long after it starts (it takes ~60 s)
MAX_FAILS=${MAX_FAILS:-5}         # more failures than this within FAIL_WINDOW s: give up, stop everything
FAIL_WINDOW=${FAIL_WINDOW:-3600}
ENGINE_DELAYS=(10 30 60 120 300)  # seconds before the n-th restart in a row
FRONT_DELAYS=(2 5 10 30 60)

log() { echo "$(date '+%F %T') $*"; }
die() {
  echo "service.sh: $*" >&2
  exit 1
}
state() {  # state words...: run/state is "<state> <epoch> <what>"; starting|waiting|serving|restarting|lending|lent|stopping|stopped|failed
  echo "$1 $(date +%s) ${*:2}" > "$RUN/state.new" && mv -f "$RUN/state.new" "$RUN/state"
  log "state: $*"
}

settings() {
  ADDR=${DSGB10_ADDR:-127.0.0.1} PORT=${DSGB10_PORT:-18893}
  KEY=${DSGB10_KEY_FILE:-$HOME/dsgb10-data/dev-api-key}
  EARGS=${DSGB10_ENGINE_ARGS:---max-ctx 262144} FARGS=${DSGB10_FRONT_ARGS:-} NOTE=${DSGB10_NOTE:-}
  [ "$MOCK" = 1 ] && EARGS=${DSGB10_ENGINE_ARGS:---delay-ms 5}  # the mock engine's arguments
  BIN=${DSGB10_BIN:-$SRC/engine/build/server/$NAME} FBIN=${DSGB10_FRONT_BIN:-$SRC/frontend/target/release/dsfront}
  MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-100}
  YIELD_IDLE_S=${DSGB10_YIELD_IDLE_S:-600} HOLD_S=${DSGB10_HOLD_S:-600}
  KEEP_GIB=${DSGB10_KEEP_GIB:-4}
}

# the front-end's addresses, one per line; its URLs; every address's /health code; all of them 200
addrs() { tr ', ' '\n\n' <<< "$ADDR" | sed '/^$/d'; }
urls() { addrs | sed "s|.*|http://&:$PORT/v1|" | paste -sd' '; }
health() {
  local a
  for a in $(addrs); do curl -s -m 3 -o /dev/null -w '%{http_code} ' "http://$a:$PORT/health"; done
}
health_ok() {
  local c
  for c in $(health); do [ "$c" = 200 ] || return 1; done
}

check() {
  local a
  [ -n "$(addrs)" ] || die "DSGB10_ADDR is empty"
  for a in $(addrs); do
    case $a in
      0.0.0.0 | :: | '[::]' | '*') die "will not listen on '$a': specific addresses only, never 0.0.0.0 (HTTP stays off public interfaces)" ;;
    esac
    [[ $a =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "'$a' in DSGB10_ADDR is not an IPv4 address"
  done
  [ "$(addrs | sort | uniq -d)" = "" ] || die "DSGB10_ADDR names an address twice"
  [[ $PORT =~ ^[0-9]+$ ]] || die "DSGB10_PORT '$PORT' is not a port"
  [ "$PORT" != 8888 ] || [ "${DSGB10_PROD:-}" = 1 ] || die "port 8888 is the production endpoint: DSGB10_PROD=1"
  [ -r "$KEY" ] || die "no API key file $KEY"
  [ $((8#$(stat -c %a "$KEY") & 8#077)) = 0 ] || die "$KEY can be read by others: chmod 600 it"
  [[ $YIELD_IDLE_S =~ ^[0-9]+$ && $HOLD_S =~ ^[0-9]+$ ]] || die "DSGB10_YIELD_IDLE_S and DSGB10_HOLD_S are seconds"
  [[ $KEEP_GIB =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "DSGB10_KEEP_GIB '$KEEP_GIB' is not GiB"
  if [ "$MOCK" = 1 ]; then  # a test instance must share nothing with the service
    [ "$H" != "$(fleet_local "$FLEET_SERVICE_DIR")" ] || die "a mock instance needs its own DSGB10_HOME"
    [ "$LOCK" != "$FLEET_LOCK" ] || die "a mock instance needs its own RING_LOCK"
  else
    [ -x "$BIN" ] || die "no engine binary $BIN (cmake --build engine/build)"
    grep -qa -- --disk-tag "$BIN" || die "$BIN predates kept sessions (no --disk-tag), which this script passes: cmake --build engine/build"
  fi
  [ -x "$FBIN" ] || die "no front-end binary $FBIN (cargo build --release in frontend/)"
}

sup_pid() {  # the supervisor's pid, if one runs
  local p
  p=$(cat "$RUN/supervisor.pid" 2> /dev/null) && [ -n "$p" ] && kill -0 "$p" 2> /dev/null &&
    tr '\0' ' ' < "/proc/$p/cmdline" 2> /dev/null | grep -q "service.sh _supervise" && echo "$p"
}

rotate() {  # file: file -> file.1 -> ... -> file.$KEEP
  local f=$1 i
  [ -s "$f" ] || return 0
  for ((i = KEEP - 1; i >= 1; i--)); do [ -f "$f.$i" ] && mv -f "$f.$i" "$f.$((i + 1))"; done
  mv -f "$f" "$f.1"
}
rotate_live() {  # file: rotate a log that a running process appends to, once it passes LOG_MIB: copy, truncate
  local f=$1 i
  [ "$(stat -c %s "$f" 2> /dev/null || echo 0)" -gt $((LOG_MIB << 20)) ] || return 0
  for ((i = KEEP - 1; i >= 1; i--)); do [ -f "$f.$i" ] && mv -f "$f.$i" "$f.$((i + 1))"; done
  cp "$f" "$f.1" && : > "$f"
}

reap() {  # shell text: wait up to $1 s for the ranks to exit by themselves, then pkill, 10 s more, then kill -9
  echo "for i in \$(seq $((2 * $1))); do pgrep -x $NAME > /dev/null || exit 0; sleep 0.5; done; pkill -x $NAME; for i in \$(seq 20); do pgrep -x $NAME > /dev/null || exit 0; sleep 0.5; done; pkill -9 -x $NAME"
}
reap_all() {  # seconds: every box's ranks (only with the ring lock: then every dsgb10_rank is ours)
  local ps=() w
  [ "$MOCK" = 1 ] && return 0
  bash -c "$(reap "$1")" &
  ps+=($!)
  for w in "${WS[@]}"; do
    "${SSH[@]}" "$w" "$(reap "$1")" &
    ps+=($!)
  done
  wait "${ps[@]}"
}

freeze() {  # the binaries as they are now, here, and the fleet file beside the engine (the workers'
  if [ "$MOCK" = 1 ]; then  # copies: freeze_boxes)
    mkdir -p "$H/bin" && cp "$FBIN" "$H/bin/dsfront.new" && mv -f "$H/bin/dsfront.new" "$H/bin/dsfront"
    return
  fi
  mkdir -p "$H/bin" &&
    cp "$BIN" "$H/bin/$NAME.new" && mv -f "$H/bin/$NAME.new" "$H/bin/$NAME" &&
    cp "$FLEET" "$H/bin/fleet.conf.new" && mv -f "$H/bin/fleet.conf.new" "$H/bin/fleet.conf" &&
    cp "$FBIN" "$H/bin/dsfront.new" && mv -f "$H/bin/dsfront.new" "$H/bin/dsfront" &&
    cp "$SRC/engine/server/tools/evict_cache.py" "$H/bin/"
}
BOXES=0
freeze_boxes() {  # the frozen engine and fleet file on every worker, once, before the engine's first
  local w       # start: with the ring, since while a job holds it the boxes are the job's
  if [ "$MOCK" = 1 ] || [ "$BOXES" = 1 ]; then return 0; fi
  for w in "${WS[@]}"; do
    "${SSH[@]}" "$w" "mkdir -p $RH/bin $RH/logs" &&
      scp -q -o BatchMode=yes -o ConnectTimeout=10 "$H/bin/$NAME" "$w:$RH/bin/$NAME.new" &&
      "${SSH[@]}" "$w" "cat > $RH/bin/fleet.conf.new && mv -f $RH/bin/$NAME.new $RH/bin/$NAME && mv -f $RH/bin/fleet.conf.new $RH/bin/fleet.conf" < "$H/bin/fleet.conf" || {
      log "copying the binary to $w failed"
      return 1
    }
  done
  BOXES=1
}

avail_gib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1048576 }' /proc/meminfo; }

start_engine() {  # strays out, the model files' page cache dropped, memory checked; ranks 1-3, then rank 0
  local ps=() w r a low= old
  if [ "$MOCK" = 1 ]; then  # dsfront's mock engine in place of the four ranks
    rotate "$LOGS/rank.0.log"
    rm -f "$SOCK"
    "$H/bin/dsfront" mock-engine --socket "$SOCK" $EARGS >> "$LOGS/rank.0.log" 2>&1 < /dev/null 6>&- 8>&- 9>&- &
    R0=$! R0_T=$(date +%s) READY=0 KILL_AT=0
    echo "$R0" > "$RUN/rank0.pid"
    log "engine started: the mock, pid $R0, dsfront mock-engine $EARGS"
    return 0
  fi
  reap_all 0
  bash -c "python3 '$H/bin/evict_cache.py' $(fleet_model_files 0)" &
  ps+=($!)
  r=1
  for w in "${WS[@]}"; do
    "${SSH[@]}" "$w" "python3 - $(fleet_model_files $r)" < "$H/bin/evict_cache.py" &
    ps+=($!)
    r=$((r + 1))
  done
  wait "${ps[@]}"
  a=$(avail_gib)
  [ "${a:-0}" -ge "$MIN_AVAIL_GIB" ] || low="${FLEET_RANK0_NAME:-rank 0} $a GiB"
  for w in "${WS[@]}"; do
    a=$("${SSH[@]}" "$w" "awk '/^MemAvailable:/ { printf \"%d\", \$2 / 1048576 }' /proc/meminfo")
    [ "${a:-0}" -ge "$MIN_AVAIL_GIB" ] || low="$low${low:+, }$w ${a:-?} GiB"
  done
  if [ -n "$low" ]; then
    log "not starting: MemAvailable under $MIN_AVAIL_GIB GiB on $low (another model loaded?)"
    return 1
  fi
  ARGS="--mode serve --socket $SOCK --disk-tag service --keep-gib ${KEEP_GIB:-4} $EARGS"
  r=1
  for w in "${WS[@]}"; do
    old=$(for ((i = KEEP - 1; i >= 1; i--)); do echo -n "$i "; done)
    "${SSH[@]}" "$w" "cd $RH && f=logs/rank.$r.log && if [ -s \$f ]; then for i in $old; do [ -f \$f.\$i ] && mv -f \$f.\$i \$f.\$((i + 1)); done; mv -f \$f \$f.1; fi; setsid nohup bin/$NAME --rank $r $ARGS >> \$f 2>&1 < /dev/null &" ||
      {
        log "starting rank $r on $w failed"
        reap_all 0
        return 1
      }
    r=$((r + 1))
  done
  rotate "$LOGS/rank.0.log"
  rm -f "$SOCK"
  "$H/bin/$NAME" --rank 0 $ARGS >> "$LOGS/rank.0.log" 2>&1 < /dev/null 6>&- 8>&- 9>&- &
  R0=$! R0_T=$(date +%s) READY=0 KILL_AT=0
  echo "$R0" > "$RUN/rank0.pid"
  log "engine started: rank 0 pid $R0, $H/bin/$NAME $ARGS"
}

stop_engine() {  # rank 0 first: on SIGTERM a server ends the request in flight, keeps the sessions (DSGB10_KEEP_GIB) and stops ranks 1-3 (kCmdStop)
  local i
  if [ -n "$R0" ] && kill -0 "$R0" 2> /dev/null; then
    kill -TERM "$R0"
    for ((i = 0; i < 120; i++)); do
      kill -0 "$R0" 2> /dev/null || break
      sleep 0.5
    done
    kill -0 "$R0" 2> /dev/null && log "rank 0 still running 60 s after SIGTERM: kill -9 (nothing kept)" && kill -9 "$R0"
  fi
  if [ -n "$R0" ]; then
    wait "$R0" 2> /dev/null
    log "rank 0 exited (rc $?)"
  fi
  R0= READY=0
  rm -f "$RUN/rank0.pid"
  reap_all 10
}

start_front() {
  rotate "$LOGS/front.log"
  "$H/bin/dsfront" serve $(addrs | sed "s|.*|--listen &:$PORT|") --engine-socket "$SOCK" --api-key-file "$KEY" \
    --admin-socket "$ADMIN" --hold-s "$HOLD_S" --table-file "$RUN/front.table" $FARGS >> "$LOGS/front.log" 2>&1 < /dev/null 6>&- 8>&- 9>&- &
  FE=$! FE_T=$(date +%s)
  echo "$FE" > "$RUN/front.pid"
  log "front-end started: pid $FE, $(urls)"
}

stop_front() {
  local i
  if [ -n "$FE" ] && kill -0 "$FE" 2> /dev/null; then
    kill -TERM "$FE"
    for ((i = 0; i < 20; i++)); do
      kill -0 "$FE" 2> /dev/null || break
      sleep 0.5
    done
    kill -0 "$FE" 2> /dev/null && log "front-end still running 10 s after SIGTERM: kill -9" && kill -9 "$FE"
  fi
  [ -n "$FE" ] && wait "$FE" 2> /dev/null
  FE=
  rm -f "$RUN/front.pid"
}

# Lending the ring (docs/engine.md, "Borrowing the ring"): to a job that waits for it, once dsfront
# has had no request for YIELD_IDLE_S; dsfront holds requests meanwhile, and the ring comes back when
# the job is done, before any other job if requests wait.
admin() { "$H/bin/dsfront" admin --socket "$ADMIN" "$@" 2> /dev/null; }
wants() {  # [held]: the jobs waiting for the ring, a line each (held: drivers' held wants too, ringlock.sh);
  local f   # a stale want (its job gone: its lock free) goes
  for f in "$RING_WANTS"/*; do
    case $f in *.new | "$RING_WANTS/*") continue ;; *.held) [ "${1:-}" = held ] || continue ;; esac
    flock -n -s -E 99 "$f" true 2> /dev/null
    case $? in
      0) rm -f "$f" ;;
      99) head -n 1 "$f" 2> /dev/null ;;
    esac
  done
}
own() { echo "$(date '+%F %T') pid $$ service: service.sh, front-end $(urls)${NOTE:+ ($NOTE)}" > "$OWNERF"; }
lend() {  # jobs: dsfront holds requests (its hold succeeded); the engine stops and the ring lock goes
  log "lending the ring: no request for $YIELD_IDLE_S s; waiting for it: $(paste -sd'|' <<< "$1")"
  state lending "stopping the engine for a ring job"
  stop_engine
  flock -u 9
  LENT=1 LENT_T=$(date +%s) RECLAIMING=0
  state lent "to a ring job since $(date +%T); requests wait at the front-end, up to $HOLD_S s each"
}
retake() {  # the ring lock is ours again: dsfront lets requests through once the engine is back
  local why="the job is done"
  if [ "$RECLAIMING" = 1 ]; then
    why="the job is done; requests wait"
    flock -u 6
  fi
  own
  LENT=0 RECLAIMING=0
  if [ "$FIRST" = 1 ]; then
    log "the ring is ours after $(($(date +%s) - LENT_T)) s of waiting at the start ($why)"
  else
    log "the ring is back after $(($(date +%s) - LENT_T)) s ($why)"
  fi
  FIRST=0
  admin release > /dev/null || log "dsfront's admin socket did not answer: it is released again in a minute"
}

FAILS=() RESTARTS=0
failure() {  # what: count a failure; false once there were more than MAX_FAILS in FAIL_WINDOW s
  local now t keep=()
  now=$(date +%s)
  for t in "${FAILS[@]}"; do ((now - t < FAIL_WINDOW)) && keep+=("$t"); done
  FAILS=("${keep[@]}" "$now")
  RESTARTS=$((RESTARTS + 1))
  echo "$RESTARTS restarts; the last at $(date '+%F %T'): $1" > "$RUN/restarts"
  log "failure: $1 (${#FAILS[@]} in the last $FAIL_WINDOW s)"
  ((${#FAILS[@]} <= MAX_FAILS))
}

supervise() {
  exec >> "$LOGS/service.log" 2>&1 < /dev/null
  exec 8>> "$RUN/supervisor.lock"
  flock -w 5 8 || {  # a few seconds: a stopped supervisor's last children may still hold it
    log "another supervisor holds $RUN/supervisor.lock"
    exit 1
  }
  echo $$ > "$RUN/supervisor.pid"
  echo "0 restarts" > "$RUN/restarts"
  # sshd's OOM exemption (-1000) is inherited by everything started over SSH (so on rank 0): hand it
  # back, so that a rank that overruns memory is killed, not the box hung
  echo 0 > /proc/self/oom_score_adj 2> /dev/null
  STOP=
  trap 'STOP=1' TERM INT
  log "supervisor pid $$: front-end $(urls), engine $EARGS"
  exec 6>> "$RING_RECLAIM"  # flock'd while requests wait during a lend: no job takes the ring before us
  if ! freeze; then
    state failed "copying the binaries failed"
    rm -f "$RUN/supervisor.pid"
    exit 1
  fi
  R0= FE= READY=0 READY_T=0 KILL_AT=0 R0_T=0 FE_T=0 LENT=0 LENT_T=0 RECLAIMING=0 FIRST=0
  # The front-end first: until the engine is up, it holds requests (DSGB10_HOLD_S each). A job that
  # holds the ring keeps it as during a lend, with no bound on the wait: a request that waits takes it
  # back (the job is preempted unless PREEMPT=0); with none, the engine starts once no job waits.
  exec 9>> "$LOCK"
  start_front
  if flock -n 9; then
    own
  else
    LENT=1 LENT_T=$(date +%s) FIRST=1
    log "the ring is busy ($(cat "$OWNERF" 2> /dev/null)): the engine starts once no job waits; requests wait at the front-end meanwhile, and one that waits takes the ring back as during a lend"
    state waiting "for the ring; requests wait at the front-end, up to $HOLD_S s each"
  fi
  local now rc eb=0 fb=0 next_e=0 next_f=0 last_w=0 why= w r gone jobs seen= g n
  while [ -z "$STOP" ]; do
    now=$(date +%s)
    # the front-end: it reconnects to the engine's socket by itself, so it outlives engine restarts
    if [ -n "$FE" ] && ! kill -0 "$FE" 2> /dev/null; then
      wait "$FE"
      rc=$?
      log "front-end exited (rc $rc) after $((now - FE_T)) s: $(tail -n 2 "$LOGS/front.log" | tr '\n' '|')"
      FE=
      ((now - FE_T >= 600)) && fb=0
      failure "front-end exited (rc $rc)" || {
        why="too many failures; the last: the front-end exited (rc $rc)"
        break
      }
      next_f=$((now + FRONT_DELAYS[fb < 4 ? fb : 4]))
      fb=$((fb + 1))
    fi
    if [ -z "$FE" ] && ((now >= next_f)); then start_front; fi
    # the engine: rank 0 exits when a peer dies or a step hangs (the step watchdog)
    if [ -n "$R0" ] && ! kill -0 "$R0" 2> /dev/null; then
      wait "$R0"
      rc=$?
      log "rank 0 exited (rc $rc) after $((now - R0_T)) s: $(tail -n 3 "$LOGS/rank.0.log" | tr '\n' '|')"
      R0= READY=0
      rm -f "$RUN/rank0.pid"
      reap_all 10
      ((now - R0_T >= 600)) && eb=0
      failure "rank 0 exited (rc $rc)" || {
        why="too many failures; the last: rank 0 exited (rc $rc)"
        break
      }
      next_e=$((now + ENGINE_DELAYS[eb < 4 ? eb : 4]))
      eb=$((eb + 1))
      state restarting "rank 0 exited (rc $rc); the next start at $(date -d "@$next_e" +%T)"
    fi
    # the ring: lent to a job that waits, once idle; back when the job is done and no other waits, or
    # at once if requests wait (requests first)
    if [ "$LENT" = 0 ]; then
      jobs=$(wants)
      # not within YIELD_IDLE_S of the engine's last start either: what brought it back was a request
      if [ -n "$jobs" ] && [ -z "$STOP" ] && { [ -z "$R0" ] || { [ "$READY" = 1 ] && ((now - READY_T >= YIELD_IDLE_S)); }; } &&
        [ "$(admin hold "$YIELD_IDLE_S")" = ok ]; then
        lend "$jobs"
      elif [ -n "$jobs" ] && [ "$jobs" != "$seen" ]; then
        log "waiting for the ring: $(paste -sd'|' <<< "$jobs"); it is lent once no request has come for $YIELD_IDLE_S s"
      fi
      seen=$jobs
    else
      g=$(admin status)
      # waiting at the start: dsfront says the ring is lent (/health, a waiting stream's comment)
      [ "$FIRST" = 1 ] && [[ $g == "held 0 "* ]] && admin hold 0 > /dev/null
      n=$(sed -n 's/.* waiting \([0-9]*\) .*/\1/p' <<< "$g")
      if [ "${n:-0}" -gt 0 ] && [ "$RECLAIMING" = 0 ] && flock -n 6; then
        RECLAIMING=1
        log "requests wait ($n): the job is preempted (unless it runs with PREEMPT=0), and the ring comes back when it ends, before any other job"
        if [ "$FIRST" = 1 ]; then
          state waiting "for the ring; requests wait: the job on it is preempted (unless it runs with PREEMPT=0)"
        else
          state lent "to a ring job since $(date -d "@$LENT_T" +%T); requests wait: the job is preempted"
        fi
      fi
      if { [ "$RECLAIMING" = 1 ] || [ -z "$(wants held)" ]; } && flock -n 9; then  # a driver's gap: stays lent
        retake
        next_e=0 eb=0
      fi
    fi
    if [ -z "$R0" ] && [ "$LENT" = 0 ] && ((now >= next_e)) && [ -z "$STOP" ]; then
      state starting "the engine: page cache, memory check, four ranks"
      if ! freeze_boxes || ! start_engine; then
        failure "the engine did not start" || {
          why="too many failures; the last: the engine did not start"
          break
        }
        next_e=$(($(date +%s) + ENGINE_DELAYS[eb < 4 ? eb : 4]))
        eb=$((eb + 1))
        state restarting "the engine did not start; the next try at $(date -d "@$next_e" +%T)"
      fi
    fi
    if [ -n "$R0" ] && [ "$READY" = 0 ]; then
      if [ -S "$SOCK" ]; then
        READY=1 READY_T=$now
        log "engine ready after $((now - R0_T)) s"
        state serving "$(urls); engine up since $(date -d "@$R0_T" '+%F %T')"
      elif ((now - R0_T > READY_S && KILL_AT == 0)); then
        log "rank 0 not listening $READY_S s after its start: SIGTERM"
        kill -TERM "$R0"
        KILL_AT=$((now + 30))
      fi
    fi
    if [ -n "$R0" ] && ((KILL_AT > 0 && now >= KILL_AT)); then
      log "rank 0 still running 30 s after SIGTERM: kill -9"
      kill -9 "$R0"
      KILL_AT=$((now + 3600))
    fi
    # every minute: ranks 1-3 still running (a worker that died between requests fails rank 0 only at
    # the next request), and the logs
    if ((now - last_w >= 60)); then
      last_w=$now
      if [ "$READY" = 1 ] && ((KILL_AT == 0)); then
        gone= r=1
        for w in "${WS[@]}"; do
          "${SSH[@]}" "$w" "pgrep -x $NAME > /dev/null"
          [ $? = 1 ] && gone="$gone rank $r ($w)"
          r=$((r + 1))
        done
        if [ -n "$gone" ]; then
          log "not running:$gone; stopping rank 0 (SIGTERM) to restart the engine"
          kill -TERM "$R0"
          KILL_AT=$((now + 30))
        fi
      fi
      rotate_live "$LOGS/rank.0.log"
      rotate_live "$LOGS/front.log"
      rotate_live "$LOGS/service.log"
      # a release that did not arrive (dsfront restarted, or its socket did not answer) must not hold
      # requests while the ring is ours
      [ "$LENT" = 0 ] && [ -n "$FE" ] && admin release > /dev/null
    fi
    sleep 2 6>&- 8>&- 9>&- &  # holding no lock: a restart takes them at once
    wait $!
  done
  state stopping "${why:-by service.sh stop}"
  [ "$LENT" = 0 ] && stop_engine  # lent: the ranks are the job's
  stop_front
  rm -f "$RUN/supervisor.pid" "$SOCK" "$ADMIN"
  if [ -n "$why" ]; then state failed "$why"; else state stopped "by service.sh stop"; fi
  exit 0  # the ring lock goes with this process
}

cmd_start() {
  local p s last= t0=$SECONDS
  if p=$(sup_pid); then
    echo "service.sh: already running (supervisor pid $p)"
    cmd_status
    return
  fi
  mkdir -p "$RUN" "$LOGS" && chmod 700 "$RUN" || die "cannot make $RUN"
  declare -p ADDR PORT KEY EARGS FARGS NOTE BIN FBIN MIN_AVAIL_GIB YIELD_IDLE_S HOLD_S KEEP_GIB MOCK > "$RUN/settings"
  rm -f "$RUN/state"
  setsid nohup bash "$SELF" _supervise > /dev/null 2>&1 < /dev/null &
  p=$!
  echo "service.sh: supervisor started; log $LOGS/service.log"
  [ "${START_WAIT:-900}" -gt 0 ] || return 0
  while ((SECONDS - t0 < ${START_WAIT:-900})); do
    sleep 2
    if ! kill -0 "$p" 2> /dev/null && [ ! -s "$RUN/state" ]; then
      echo "service.sh: the supervisor exited before it began: $(tail -n 1 "$LOGS/service.log")" >&2
      return 1
    fi
    s=$(cat "$RUN/state" 2> /dev/null) || continue
    if [ "$s" != "$last" ]; then
      echo "  $(date +%T) ${s%% *}: $(cut -d' ' -f3- <<< "$s")"
      last=$s
    fi
    case ${s%% *} in
      serving)
        # the front-end connects to the engine a moment after its socket appears: until then /health is 503
        for _ in $(seq 1 15); do
          health_ok && break
          sleep 2
        done
        cmd_status
        return
        ;;
      failed | stopped) return 1 ;;
    esac
  done
  if [ "$(cut -d' ' -f1 "$RUN/state" 2> /dev/null)" = waiting ]; then
    echo "service.sh: still waiting for the ring after ${START_WAIT:-900} s; dsfront holds requests, and the engine starts once the ring is free (service.sh status)"
  else
    echo "service.sh: not serving after ${START_WAIT:-900} s; it goes on trying (service.sh status)"
  fi
  return 1
}

cmd_stop() {
  local p i
  # a test instance's cleanup must not reach the ranks; its other settings stay those of this command
  # (restart starts with this environment's)
  [ "$(source "$RUN/settings" 2> /dev/null; echo "$MOCK")" = 1 ] && MOCK=1 WS=()
  if p=$(sup_pid); then
    echo "service.sh: stopping (supervisor pid $p)"
    kill -TERM "$p"
    for ((i = 0; i < 240; i++)); do
      kill -0 "$p" 2> /dev/null || break
      sleep 0.5
    done
    if kill -0 "$p" 2> /dev/null; then
      echo "service.sh: the supervisor still runs after 120 s (pid $p)" >&2
      return 1
    fi
    echo "service.sh: stopped ($(cut -d' ' -f3- "$RUN/state" 2> /dev/null))"
    return 0
  fi
  # no supervisor: clean up after one that died, but only with the ring lock (else the ranks may be
  # another job's)
  if [ -f "$RUN/rank0.pid" ] || [ -f "$RUN/front.pid" ]; then
    exec 9>> "$LOCK"
    if flock -n 9; then
      p=$(cat "$RUN/front.pid" 2> /dev/null)
      [ -n "$p" ] && [ "$(cat "/proc/$p/comm" 2> /dev/null)" = dsfront ] && kill -TERM "$p"
      p=$(cat "$RUN/rank0.pid" 2> /dev/null)
      if [ "$MOCK" = 1 ]; then
        [ -n "$p" ] && [ "$(cat "/proc/$p/comm" 2> /dev/null)" = dsfront ] && kill -TERM "$p"
      elif [ -n "$p" ]; then
        pkill -x $NAME
      fi
      reap_all 10
      rm -f "$RUN/rank0.pid" "$RUN/front.pid" "$SOCK"
      echo "service.sh: not running; stopped what a dead supervisor left"
      return 0
    fi
    echo "service.sh: not running; the ring is held ($(cat "$OWNERF" 2> /dev/null)), so its processes are left alone"
    return 0
  fi
  echo "service.sh: not running"
}

# The kept sessions (the engine's files on every box, rank 0's histories) and dsfront's saved table:
# what conversations left on disk, removed. Not while the service runs, which would write them again.
cmd_forget() {
  local p f d dd r left=0 n
  [ "$(source "$RUN/settings" 2> /dev/null; echo "$MOCK")" = 1 ] && MOCK=1 WS=()
  if p=$(sup_pid); then
    echo "service.sh: running (supervisor pid $p): service.sh stop --forget stops it, then forgets" >&2
    return 1
  fi
  p=$(cat "$RUN/rank0.pid" 2> /dev/null)
  if [ -n "$p" ] && kill -0 "$p" 2> /dev/null; then
    echo "service.sh: rank 0 still runs (pid $p): service.sh stop first" >&2
    return 1
  fi
  rm -f "$RUN/front.table" "$RUN/front.table.new"
  if [ "$MOCK" = 0 ]; then
    # each box's spill directory: --disk-dir's, else the fleet's RANK<r>_SPILL ({rank}: r; its ~/ is
    # that box's)
    dd=$(source "$RUN/settings" 2> /dev/null; sed -n 's/.*--disk-dir \([^ ]*\).*/\1/p' <<< "${EARGS:-}")
    for r in 0 1 2 3; do
      d=$(fleet_rank "${dd:-$(fleet_get "RANK${r}_SPILL")}" "$r")
      [ -n "$d" ] || continue
      f="d=$d; rm -f \$d/rank[0-3].service.spill \$d/rank[0-3].service.index \$d/rank[0-3].service.index.new"
      f="$f \$d/server.service.index \$d/server.service.index.new; ls \$d 2> /dev/null | grep -c '[.]service[.]'"
      if [ "$r" = 0 ]; then
        n=$(bash -c "$f")
      else
        n=$("${SSH[@]}" "${WS[r - 1]}" "$f")
      fi
      left=$((left + ${n:-1}))
    done
  fi
  if [ "$left" != 0 ]; then
    echo "service.sh: $left kept files are left (a box unreachable?): run forget again" >&2
    return 1
  fi
  echo "service.sh: forgotten: the kept sessions on every box and dsfront's saved table"
}

cmd_status() {
  local p st since what code r w pids fp
  [ -f "$RUN/settings" ] && source "$RUN/settings"
  [ "$MOCK" = 1 ] && WS=()
  if ! p=$(sup_pid); then
    echo "service: not running"
    [ -f "$RUN/state" ] && read -r st since what < "$RUN/state" &&
      echo "  last: $st at $(date -d "@$since" '+%F %T'): $what"
    return 3
  fi
  read -r st since what < "$RUN/state"
  # waiting for the ring: who holds it now (jobs may follow one another while the start waits)
  [ "$st" = waiting ] && [[ $what == "for the ring"* ]] &&
    what="for the ring, held by: $(cut -c1-120 "$OWNERF" 2> /dev/null)${what#for the ring}"
  echo "service: $st since $(date -d "@$since" '+%F %T'): $what (supervisor pid $p)"
  echo "  $(cat "$RUN/restarts" 2> /dev/null)"
  echo "  ring: $(cat "$OWNERF" 2> /dev/null)"
  code=$(health)
  fp=$(cat "$RUN/front.pid" 2> /dev/null)
  echo "  front-end: $(urls), pid ${fp:-none}, /health ${code% } ($(health_ok && echo "the engine is connected" || echo "not serving"))"
  p=$(admin status) && echo "  gate: $p"
  p=$(wants)
  [ -n "$p" ] && echo "  waiting for the ring:" && sed 's/^/    /' <<< "$p"
  p=$(cat "$RUN/rank0.pid" 2> /dev/null)
  if [ -n "$p" ] && kill -0 "$p" 2> /dev/null; then
    echo "  rank 0 (${FLEET_RANK0_NAME:-here}): pid $p, up $(ps -o etime= -p "$p" | tr -d ' '), MemAvailable $(avail_gib) GiB"
  else
    echo "  rank 0 (${FLEET_RANK0_NAME:-here}): not running, MemAvailable $(avail_gib) GiB"
  fi
  r=1
  for w in "${WS[@]}"; do
    pids=$("${SSH[@]}" -o ConnectTimeout=5 "$w" "pgrep -x $NAME | tr '\n' ' '; awk '/^MemAvailable:/ { printf \"| MemAvailable %d GiB\", \$2 / 1048576 }' /proc/meminfo")
    echo "  rank $r ($w): ${pids:-unreachable}"
    r=$((r + 1))
  done
  echo "  logs: $LOGS/{service,rank.0,front}.log${WS[*]:+; workers ~/$RH/logs/rank.<r>.log}"
  [ "$st" = serving ] && health_ok
}

case ${1:-} in
  start)
    settings
    check
    cmd_start
    ;;
  _supervise)
    source "$RUN/settings"
    [ "$MOCK" = 1 ] && WS=()
    supervise
    ;;
  stop)
    [ $# -le 1 ] || [ "$2" = --forget ] || die "usage: service.sh stop [--forget]"
    cmd_stop || exit 1
    [ "${2:-}" != --forget ] || cmd_forget
    ;;
  forget) cmd_forget ;;
  status) cmd_status ;;
  restart)
    settings
    check
    cmd_stop && cmd_start
    ;;
  *) die "usage: service.sh start|stop [--forget]|status|restart|forget (settings: the header of $SELF)" ;;
esac
