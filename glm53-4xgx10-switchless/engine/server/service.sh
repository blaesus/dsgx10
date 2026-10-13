#!/bin/bash
# The GLM-5.3 engine as a service (docs/serving.md, "As a service"): the four ranks of glm53_rank in serve mode and
# the front-end, glmfront, under a supervisor on rank 0 that restarts what fails with bounded backoff: the DeepSeek
# engine's service.sh re-specialised. No systemd unit: the supervisor is a detached bash process, started and
# stopped by hand.
#
# The ring. Both services and every ring job take the one ring lock (the fleet's RING_LOCK, ringlock.sh), so the
# GLM engine never runs while the DeepSeek service (or a job) holds the ring, and the other way round. This service
# is the ring's borrower, not its owner (DESIGN decision 8: production keeps serving DeepSeek):
#   - it takes the ring only through the queue of wants (ring.wants/, as a job does): it registers a want while it
#     starts and while requests wait at its front-end, and the DeepSeek service lends it the ring as it lends it to a
#     job (once idle for its DSGB10_YIELD_IDLE_S); it never takes the reclaim flag, so it never preempts a job;
#   - it gives the ring back (its engine stopped, the sessions kept: GLM53_KEEP_GIB) as soon as anyone reclaims it
#     (the DeepSeek service's requests wait), and once it has had no request for GLM53_YIELD_IDLE_S;
#   - meanwhile glmfront holds its requests (GLM53_HOLD_S each), and they bring the engine back through a want.
#
#   engine/server/service.sh start     # returns once it serves, or failed (START_WAIT s, default 900; 0: at once)
#   engine/server/service.sh stop      # rank 0 first (it stops ranks 1-3), then the front-end
#   engine/server/service.sh status    # exit 0 serving, 1 waiting / starting / not answering, 3 not running
#   engine/server/service.sh restart   # stop, then start (with this environment's settings)
#   engine/server/service.sh forget    # the kept sessions gone on every box, and glmfront's saved table
# Settings, from the environment of start (written to run/settings, which status and stop read):
#   GLM53_ADDR, GLM53_PORT   the front-end's addresses (127.0.0.1; one or more specific IPv4 addresses, never
#                            0.0.0.0) and port (18894, the test port). Port 8888 is the DeepSeek service's: refused
#   GLM53_KEY_FILE           the API key file (mode 600): ~/dsgb10-data/dev-api-key
#   GLM53_ENGINE_ARGS        glm53_rank's arguments after --mode serve --socket <run/engine.sock>:
#                            --fill-experts (the rank files' unfilled expert pools synthetic; empty once filled)
#   GLM53_FRONT_ARGS         more of glmfront's arguments
#   GLM53_NOTE               added to ring.owner
#   GLM53_BIN, GLM53_FRONT_BIN  the binaries, copied at start: ~/dsgb10-data/glm53-server/bin/glm53_rank
#                            (install.sh), frontend/target/release/glmfront
#   GLM53_YIELD_IDLE_S (600) the ring goes back after this long without a request
#   GLM53_HOLD_S (900)       while the engine is down, glmfront holds a request this long, then 503
#   MIN_AVAIL_GIB (20)       a box with less MemAvailable than this fails the engine's start
#   GLM53_KEEP_GIB (8)       kept sessions: an engine stop keeps the sessions for the next start, up to this many GiB
# The boxes: FLEET, this configuration's fleet file (fleet/homelab.conf): the workers, SERVICE_DIR (~/glm53/service),
# the spill directories (RANK<r>_SPILL). Files: SERVICE_DIR/{bin,logs} on every box, run/ on rank 0 (mode 700: the
# engine's socket, glmfront's admin socket, pids, state, settings); kept sessions as rank<r>.service.{spill,index}
# and server.service.index in each box's spill directory, glmfront's table in run/front.table.
set -u
NAME=glm53_rank
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
SRC=$(cd "$(dirname "$0")/../.." && pwd)
. "$(dirname "$SELF")/ringlock.sh"  # the fleet (FLEET_*); LOCK, RING_OWNER, RING_WANTS, RING_RECLAIM, ring_*
H=${GLM53_HOME:-$(fleet_local "$FLEET_SERVICE_DIR")}
RH=${H#"$HOME"/}
RUN=$H/run LOGS=$H/logs
SOCK=$RUN/engine.sock ADMIN=$RUN/front.admin
WS=("${FLEET_WORKERS[@]}")
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
KEEP=${KEEP:-5} LOG_MIB=${LOG_MIB:-64}
READY_S=${READY_S:-600}
MAX_FAILS=${MAX_FAILS:-5}
FAIL_WINDOW=${FAIL_WINDOW:-3600}
ENGINE_DELAYS=(10 30 60 120 300)
FRONT_DELAYS=(2 5 10 30 60)

log() { echo "$(date '+%F %T') $*"; }
die() {
  echo "service.sh: $*" >&2
  exit 1
}
state() {  # waiting|starting|serving|yielding|restarting|stopping|stopped|failed
  echo "$1 $(date +%s) ${*:2}" > "$RUN/state.new" && mv -f "$RUN/state.new" "$RUN/state"
  log "state: $*"
}

settings() {
  ADDR=${GLM53_ADDR:-127.0.0.1} PORT=${GLM53_PORT:-18894}
  KEY=${GLM53_KEY_FILE:-$HOME/dsgb10-data/dev-api-key}
  EARGS=${GLM53_ENGINE_ARGS---fill-experts} FARGS=${GLM53_FRONT_ARGS:-} NOTE=${GLM53_NOTE:-}
  BIN=${GLM53_BIN:-$HOME/dsgb10-data/glm53-server/bin/$NAME} FBIN=${GLM53_FRONT_BIN:-$SRC/frontend/target/release/glmfront}
  MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-20}
  YIELD_IDLE_S=${GLM53_YIELD_IDLE_S:-600} HOLD_S=${GLM53_HOLD_S:-900}
  KEEP_GIB=${GLM53_KEEP_GIB:-8}
}

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
  [ -n "$(addrs)" ] || die "GLM53_ADDR is empty"
  for a in $(addrs); do
    case $a in
      0.0.0.0 | :: | '[::]' | '*') die "will not listen on '$a': specific addresses only, never 0.0.0.0" ;;
    esac
    [[ $a =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "'$a' in GLM53_ADDR is not an IPv4 address"
  done
  [ "$(addrs | sort | uniq -d)" = "" ] || die "GLM53_ADDR names an address twice"
  [[ $PORT =~ ^[0-9]+$ ]] || die "GLM53_PORT '$PORT' is not a port"
  [ "$PORT" != 8888 ] || die "port 8888 is the DeepSeek service's"
  [ -r "$KEY" ] || die "no API key file $KEY"
  [ $((8#$(stat -c %a "$KEY") & 8#077)) = 0 ] || die "$KEY can be read by others: chmod 600 it"
  [[ $YIELD_IDLE_S =~ ^[0-9]+$ && $HOLD_S =~ ^[0-9]+$ ]] || die "GLM53_YIELD_IDLE_S and GLM53_HOLD_S are seconds"
  [[ $KEEP_GIB =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "GLM53_KEEP_GIB '$KEEP_GIB' is not GiB"
  [ -x "$BIN" ] || die "no engine binary $BIN (engine/server/install.sh)"
  [ -x "$FBIN" ] || die "no front-end binary $FBIN (cargo build --release in frontend/)"
}

sup_pid() {
  local p
  p=$(cat "$RUN/supervisor.pid" 2> /dev/null) && [ -n "$p" ] && kill -0 "$p" 2> /dev/null &&
    tr '\0' ' ' < "/proc/$p/cmdline" 2> /dev/null | grep -q "service.sh _supervise" && echo "$p"
}

rotate() {
  local f=$1 i
  [ -s "$f" ] || return 0
  for ((i = KEEP - 1; i >= 1; i--)); do [ -f "$f.$i" ] && mv -f "$f.$i" "$f.$((i + 1))"; done
  mv -f "$f" "$f.1"
}
rotate_live() {
  local f=$1 i
  [ "$(stat -c %s "$f" 2> /dev/null || echo 0)" -gt $((LOG_MIB << 20)) ] || return 0
  for ((i = KEEP - 1; i >= 1; i--)); do [ -f "$f.$i" ] && mv -f "$f.$i" "$f.$((i + 1))"; done
  cp "$f" "$f.1" && : > "$f"
}

reap() {  # shell text: wait up to $1 s for this engine's ranks to exit, then pkill, 10 s more, then kill -9
  echo "for i in \$(seq $((2 * $1))); do pgrep -x $NAME > /dev/null || exit 0; sleep 0.5; done; pkill -x $NAME; for i in \$(seq 20); do pgrep -x $NAME > /dev/null || exit 0; sleep 0.5; done; pkill -9 -x $NAME"
}
reap_all() {  # only with the ring lock
  local ps=() w
  bash -c "$(reap "$1")" &
  ps+=($!)
  for w in "${WS[@]}"; do
    "${SSH[@]}" "$w" "$(reap "$1")" &
    ps+=($!)
  done
  wait "${ps[@]}"
}

freeze() {
  mkdir -p "$H/bin" &&
    cp "$BIN" "$H/bin/$NAME.new" && mv -f "$H/bin/$NAME.new" "$H/bin/$NAME" &&
    cp "$FLEET" "$H/bin/fleet.conf.new" && mv -f "$H/bin/fleet.conf.new" "$H/bin/fleet.conf" &&
    cp "$FBIN" "$H/bin/glmfront.new" && mv -f "$H/bin/glmfront.new" "$H/bin/glmfront"
}
BOXES=0
freeze_boxes() {  # the frozen engine on every worker, once, with the ring
  local w
  [ "$BOXES" = 1 ] && return 0
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

start_engine() {  # strays out, memory checked; ranks 1-3, then rank 0
  local w r a low= old
  reap_all 0
  a=$(avail_gib)
  [ "${a:-0}" -ge "$MIN_AVAIL_GIB" ] || low="${FLEET_RANK0_NAME:-rank 0} $a GiB"
  for w in "${WS[@]}"; do
    a=$("${SSH[@]}" "$w" "awk '/^MemAvailable:/ { printf \"%d\", \$2 / 1048576 }' /proc/meminfo")
    [ "${a:-0}" -ge "$MIN_AVAIL_GIB" ] || low="$low${low:+, }$w ${a:-?} GiB"
  done
  if [ -n "$low" ]; then
    log "not starting: MemAvailable under $MIN_AVAIL_GIB GiB on $low"
    return 1
  fi
  ARGS="--mode serve --socket $SOCK --disk-tag service --keep-gib ${KEEP_GIB:-8} $EARGS"
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
  "$H/bin/$NAME" --rank 0 $ARGS >> "$LOGS/rank.0.log" 2>&1 < /dev/null 6>&- 7>&- 8>&- 9>&- &
  R0=$! R0_T=$(date +%s) READY=0 KILL_AT=0
  echo "$R0" > "$RUN/rank0.pid"
  log "engine started: rank 0 pid $R0, $H/bin/$NAME $ARGS"
}

stop_engine() {  # rank 0 first: on SIGTERM it ends the requests in flight, keeps the sessions, stops ranks 1-3
  local i
  if [ -n "$R0" ] && kill -0 "$R0" 2> /dev/null; then
    kill -TERM "$R0"
    for ((i = 0; i < 240; i++)); do
      kill -0 "$R0" 2> /dev/null || break
      sleep 0.5
    done
    kill -0 "$R0" 2> /dev/null && log "rank 0 still running 120 s after SIGTERM: kill -9 (nothing kept)" && kill -9 "$R0"
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
  "$H/bin/glmfront" serve $(addrs | sed "s|.*|--listen &:$PORT|") --engine-socket "$SOCK" --api-key-file "$KEY" \
    --admin-socket "$ADMIN" --hold-s "$HOLD_S" --table-file "$RUN/front.table" $FARGS >> "$LOGS/front.log" 2>&1 < /dev/null 6>&- 7>&- 8>&- 9>&- &
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

admin() { "$H/bin/glmfront" admin --socket "$ADMIN" "$@" 2> /dev/null; }
own() { echo "$(date '+%F %T') pid $$ service: glm53 service.sh, front-end $(urls)${NOTE:+ ($NOTE)}" > "$RING_OWNER"; }

# The want (ringlock.sh's format, flock'd on fd 7 while it exists): the DeepSeek service lends the ring to it as to
# a job, and jobs queue behind it by its age.
WANT=$RING_WANTS/$$
want_on() {
  [ -e "$WANT" ] && return 0
  mkdir -p "$RING_WANTS" && exec 7> "$WANT.new" && flock 7 &&
    printf '%s pid %s glm53-service: the GLM-5.3 service (%s)\n%s\n' "$(date '+%F %T')" $$ "$1" "$(ring_key)" >&7 &&
    mv -f "$WANT.new" "$WANT" && log "want registered: $1"
}
want_off() {
  [ -e "$WANT" ] || [ -e "$WANT.new" ] || return 0
  rm -f "$WANT" "$WANT.new"
  exec 7>&-
}
waiting_requests() { sed -n 's/.* waiting \([0-9]*\) .*/\1/p' <<< "$(admin status)"; }

# The ring back to whoever wants it: glmfront holds requests, the engine stops (keeping the sessions), the lock goes.
give_back() {
  log "giving the ring back: $1"
  state yielding "$1"
  admin hold 0 > /dev/null
  stop_engine
  admin hold 0 > /dev/null
  flock -u 9
  HAVE=0
  state waiting "for the ring ($1); requests wait at the front-end, up to $HOLD_S s each"
}

FAILS=() RESTARTS=0
failure() {
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
  flock -w 5 8 || {
    log "another supervisor holds $RUN/supervisor.lock"
    exit 1
  }
  echo $$ > "$RUN/supervisor.pid"
  echo "0 restarts" > "$RUN/restarts"
  echo 0 > /proc/self/oom_score_adj 2> /dev/null
  STOP=
  trap 'STOP=1' TERM INT
  log "supervisor pid $$: front-end $(urls), engine $EARGS"
  if ! freeze; then
    state failed "copying the binaries failed"
    rm -f "$RUN/supervisor.pid"
    exit 1
  fi
  R0= FE= READY=0 READY_T=0 KILL_AT=0 R0_T=0 FE_T=0 HAVE=0 FIRST=1
  exec 9>> "$LOCK"
  start_front
  sleep 1
  admin hold 0 > /dev/null  # until the engine is up
  state waiting "for the ring; requests wait at the front-end, up to $HOLD_S s each"
  local now rc eb=0 fb=0 next_e=0 next_f=0 last_w=0 why= w r gone n idle g
  while [ -z "$STOP" ]; do
    now=$(date +%s)
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
    if [ -z "$FE" ] && ((now >= next_f)); then
      start_front
      [ "$HAVE" = 1 ] || { sleep 1; admin hold 0 > /dev/null; }
    fi
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
    if [ "$HAVE" = 0 ]; then
      # a want while starting or while requests wait; the ring when its turn comes
      n=$(waiting_requests)
      if [ "$FIRST" = 1 ] || [ "${n:-0}" -gt 0 ]; then
        want_on "$([ "$FIRST" = 1 ] && echo starting || echo "$n requests wait")"
      fi
      if [ -e "$WANT" ] && ! ring_reclaimed && ! ring_queued "$WANT" && flock -n 9 && { ! ring_reclaimed || ! flock -u 9; }; then
        want_off
        own
        HAVE=1 FIRST=0 next_e=0
        log "the ring is ours"
      fi
    else
      g=$(admin status)
      idle=$(sed -n 's/.* idle_s \([0-9]*\) .*/\1/p' <<< "$g")
      if ring_reclaimed; then
        give_back "another service's requests wait (reclaimed)"
      elif [ "$READY" = 1 ] && ((now - READY_T >= YIELD_IDLE_S)) && [ "$(admin hold "$YIELD_IDLE_S")" = ok ]; then
        give_back "no request for $YIELD_IDLE_S s"
      fi
    fi
    if [ "$HAVE" = 1 ] && [ -z "$R0" ] && ((now >= next_e)) && [ -z "$STOP" ]; then
      state starting "the engine: memory check, four ranks"
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
        admin release > /dev/null
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
    fi
    sleep 2 6>&- 7>&- 8>&- 9>&- &
    wait $!
  done
  state stopping "${why:-by service.sh stop}"
  [ "$HAVE" = 1 ] && stop_engine
  want_off
  stop_front
  rm -f "$RUN/supervisor.pid" "$SOCK" "$ADMIN"
  if [ -n "$why" ]; then state failed "$why"; else state stopped "by service.sh stop"; fi
  exit 0
}

cmd_start() {
  local p s last= t0=$SECONDS
  if p=$(sup_pid); then
    echo "service.sh: already running (supervisor pid $p)"
    cmd_status
    return
  fi
  mkdir -p "$RUN" "$LOGS" && chmod 700 "$RUN" || die "cannot make $RUN"
  declare -p ADDR PORT KEY EARGS FARGS NOTE BIN FBIN MIN_AVAIL_GIB YIELD_IDLE_S HOLD_S KEEP_GIB > "$RUN/settings"
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
  echo "service.sh: not serving after ${START_WAIT:-900} s; it goes on trying (service.sh status)"
  return 1
}

cmd_stop() {
  local p i
  if p=$(sup_pid); then
    echo "service.sh: stopping (supervisor pid $p)"
    kill -TERM "$p"
    for ((i = 0; i < 300; i++)); do
      kill -0 "$p" 2> /dev/null || break
      sleep 0.5
    done
    if kill -0 "$p" 2> /dev/null; then
      echo "service.sh: the supervisor still runs after 150 s (pid $p)" >&2
      return 1
    fi
    echo "service.sh: stopped ($(cut -d' ' -f3- "$RUN/state" 2> /dev/null))"
    return 0
  fi
  if [ -f "$RUN/rank0.pid" ] || [ -f "$RUN/front.pid" ]; then
    exec 9>> "$LOCK"
    p=$(cat "$RUN/front.pid" 2> /dev/null)
    [ -n "$p" ] && [ "$(cat "/proc/$p/comm" 2> /dev/null)" = glmfront ] && kill -TERM "$p"
    if flock -n 9; then  # the ranks only with the ring lock (else they may be another job's)
      pkill -x $NAME
      reap_all 10
      rm -f "$RUN/rank0.pid" "$SOCK"
      echo "service.sh: not running; stopped what a dead supervisor left"
      return 0
    fi
    rm -f "$RUN/front.pid"
    echo "service.sh: not running; the ring is held ($(cat "$RING_OWNER" 2> /dev/null)), so the ranks are left alone"
    return 0
  fi
  echo "service.sh: not running"
}

cmd_forget() {
  local p f d dd r left=0 n
  if p=$(sup_pid); then
    echo "service.sh: running (supervisor pid $p): service.sh stop --forget stops it, then forgets" >&2
    return 1
  fi
  rm -f "$RUN/front.table" "$RUN/front.table.new"
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
  if [ "$left" != 0 ]; then
    echo "service.sh: $left kept files are left (a box unreachable?): run forget again" >&2
    return 1
  fi
  echo "service.sh: forgotten: the kept sessions on every box and glmfront's saved table"
}

cmd_status() {
  local p st since what code r w pids fp
  [ -f "$RUN/settings" ] && source "$RUN/settings"
  if ! p=$(sup_pid); then
    echo "service: not running"
    [ -f "$RUN/state" ] && read -r st since what < "$RUN/state" &&
      echo "  last: $st at $(date -d "@$since" '+%F %T'): $what"
    return 3
  fi
  read -r st since what < "$RUN/state"
  echo "service: $st since $(date -d "@$since" '+%F %T'): $what (supervisor pid $p)"
  echo "  $(cat "$RUN/restarts" 2> /dev/null)"
  echo "  ring: $(cat "$RING_OWNER" 2> /dev/null)"
  code=$(health)
  fp=$(cat "$RUN/front.pid" 2> /dev/null)
  echo "  front-end: $(urls), pid ${fp:-none}, /health ${code% } ($(health_ok && echo "the engine is connected" || echo "not serving"))"
  p=$(admin status) && echo "  gate: $p"
  p=$(cat "$RUN/rank0.pid" 2> /dev/null)
  if [ -n "$p" ] && kill -0 "$p" 2> /dev/null; then
    echo "  rank 0 (${FLEET_RANK0_NAME:-here}): pid $p, up $(ps -o etime= -p "$p" | tr -d ' '), MemAvailable $(avail_gib) GiB"
  else
    echo "  rank 0 (${FLEET_RANK0_NAME:-here}): not running, MemAvailable $(avail_gib) GiB"
  fi
  r=1
  for w in "${WS[@]}"; do
    pids=$("${SSH[@]}" -o ConnectTimeout=5 "$w" "pgrep -x $NAME | tr '\n' ' '; awk '/^MemAvailable:/ { printf \"| MemAvailable %d GiB\", \$2 / 1048576 }' /proc/meminfo")
    echo "  rank $r: ${pids:-unreachable}"
    r=$((r + 1))
  done
  echo "  logs: $LOGS/{service,rank.0,front}.log; workers ~/$RH/logs/rank.<r>.log"
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
