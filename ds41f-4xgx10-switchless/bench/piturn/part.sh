#!/bin/bash
# pi's turnaround between turns, through pi itself (docs/engine.md, "pi's turnaround"): a dev engine at
# serve's defaults (--max-ctx 262144) built with the end-of-turn timers (runner-edit, aa27273), dsfront on
# 127.0.0.1:18897 (models.json's port) with --dump-dir and its stderr stamped line by line (stamp.py), then
# the edit bench's 12 tasks through pi (pidrive.py: read/edit/write only, an isolated config in OUT/agent,
# thinking off, pi's own sampling) in print mode, on a pseudo-terminal (the TUI) and in JSON mode; a
# planned stop; piturn.py's report in OUT/report.md. One ring part, under 10 minutes:
#   BIN=dsgb10_rank FE=dsfront OWNER=me PREEMPT=0 engine/server/ring.sh bash bench/piturn/part.sh OUT
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SV=$HERE/../../engine/server
D=$1
KEY=${KEY:-$HOME/dsgb10-data/dev-api-key}
PORT=18897
RUN=${RUN:-$HOME/dsgb10/editrun}
SOCK=$RUN/engine.sock
log() { echo "$(date '+%F %T') piturn: $*"; }
[ -n "${RING_HOLDER:-}" ] || {
  log "run me under ring.sh"
  exit 2
}
[ -x "${BIN:-}" ] && [ -x "${FE:-}" ] && command -v pi > /dev/null || {
  log "BIN, FE and pi are needed"
  exit 2
}
ss -ltn | grep -q "127.0.0.1:$PORT " && {
  log "port $PORT is in use"
  exit 2
}
mkdir -p "$D/dump" "$D/pi" "$D/agent" "$RUN" && chmod 700 "$RUN" "$D" "$D/dump" "$D/pi" "$D/agent"
cp "$HERE/models.json" "$D/agent/models.json"
rm -f "$SOCK"
L=$D/run4.log
log "the engine: $(sha256sum "$BIN" | cut -c1-16), dsfront: $(sha256sum "$FE" | cut -c1-16), pi $(pi --version 2> /dev/null)"
TIMEOUT=600 BIN=$BIN "$SV/run4.sh" --mode serve --socket "$SOCK" --max-ctx 262144 > "$L" 2>&1 &
R4=$!
FP=
cleanup() {  # dsfront, then a planned stop of the engine; no rank left behind
  local i
  if [ -n "$FP" ] && kill -0 "$FP" 2> /dev/null; then
    kill -TERM "$FP"
    wait "$FP" 2> /dev/null
  fi
  pkill -x dsgb10_rank
  for ((i = 0; i < 120; i++)); do
    kill -0 $R4 2> /dev/null || break
    sleep 0.5
  done
  if kill -0 $R4 2> /dev/null; then
    log "rank 0 still running 60 s after SIGTERM: run4.sh stop"
    "$SV/run4.sh" stop > "$D/run4.stop.log" 2>&1
  fi
  wait $R4
  log "stopped (run4.sh exit $?)"
  rm -f "$SOCK"
  pgrep -x dsgb10_rank > /dev/null && "$SV/run4.sh" stop > "$D/run4.stop2.log" 2>&1
}
for ((i = 0; i < 240; i++)); do
  [ -S "$SOCK" ] && break
  kill -0 $R4 2> /dev/null || break
  sleep 0.5
done
[ -S "$SOCK" ] || {
  log "the engine did not come up: $(grep -m3 -i "error\|cannot\|abort\|die\|unknown" "$L" | tr '\n' '|')"
  cleanup
  exit 1
}
log "the engine is up"
"$FE" serve --listen "127.0.0.1:$PORT" --engine-socket "$SOCK" --api-key-file "$KEY" --hold-s 120 --dump-dir "$D/dump" \
  > >(python3 "$HERE/stamp.py" > "$D/front.log") 2>&1 < /dev/null &
FP=$!
for ((i = 0; i < 60; i++)); do
  [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ] && break
  sleep 0.5
done
log "dsfront: pid $FP, /health $(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")"
read -r DSGB10_DEV_KEY < "$KEY"  # models.json reads it from the environment, never from a command line
export DSGB10_DEV_KEY PI_CODING_AGENT_DIR=$D/agent
for m in text:120 tui:120 json:90; do
  timeout 200 python3 "$HERE/pidrive.py" "$D/pi" --dump "$D/dump" --mode "${m%%:*}" --deadline-s "${m##*:}" --timeout 120 \
    > "$D/pidrive-${m%%:*}.log" 2>&1
  log "${m%%:*}: exit $?: $(grep -c "^pidrive .*exit" "$D/pidrive-${m%%:*}.log") tasks, $(grep -c "FAILED" "$D/pidrive-${m%%:*}.log") failed checks"
done
unset DSGB10_DEV_KEY
cleanup
log "the log: $(grep -c "serve: req [0-9]* session" "$L") requests, $(grep -c "last token to DONE" "$L") timed ends, $(grep -c ": end: " "$D/front.log") dsfront ends, $(grep -c -i "error\|abort\|watchdog" "$L") error lines"
python3 "$HERE/piturn.py" "$D/pi" "$D/dump" "$L" "$D/front.log" > "$D/report.md" 2>&1
log "report: exit $?"
