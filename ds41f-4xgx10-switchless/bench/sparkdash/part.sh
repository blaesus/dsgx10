#!/bin/bash
# sparkDash's own benchmark against the engine (docs/engine.md, "Against the kit with its own benchmark"):
# a dev engine at serve's defaults (--max-ctx 262144) and dsfront on 127.0.0.1:PORT with the dev key, then
# sdrun.mjs MODE (decode: the c1 bench and the prompt catalogs; prefill: real text and sparkDash's filler
# bench), then a planned stop. One ring part, under 10 minutes:
#   BIN=dsgb10_rank FE=dsfront SD_ROOT=<sparkDash at 754f40a, npm ci --omit=dev> CORPUS=corpus.txt \
#     OWNER=me PREEMPT=0 engine/server/ring.sh bash bench/sparkdash/part.sh decode|prefill OUT
# CORPUS (prefill only): corpus.py over `git archive` of a pinned commit.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SV=$HERE/../../engine/server
MODE=$1
D=$2
KEY=${KEY:-$HOME/dsgb10-data/dev-api-key}
PORT=${PORT:-18897}
RUN=${RUN:-$HOME/dsgb10/editrun}
SOCK=$RUN/engine.sock
log() { echo "$(date '+%F %T') sparkdash $MODE: $*"; }
[ -n "${RING_HOLDER:-}" ] || {
  log "run me under ring.sh"
  exit 2
}
[ -x "${BIN:-}" ] && [ -x "${FE:-}" ] && [ -d "${SD_ROOT:-}/node_modules/undici" ] || {
  log "BIN, FE and SD_ROOT (with its dependencies) are needed"
  exit 2
}
ss -ltn | grep -q "127.0.0.1:$PORT " && {
  log "port $PORT is in use"
  exit 2
}
mkdir -p "$D/$MODE" "$RUN" && chmod 700 "$RUN" "$D" "$D/$MODE"
rm -f "$SOCK"
L=$D/$MODE/run4.log
log "the engine: $(sha256sum "$BIN" | cut -c1-16), dsfront: $(sha256sum "$FE" | cut -c1-16)"
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
    "$SV/run4.sh" stop > "$D/$MODE/run4.stop.log" 2>&1
  fi
  wait $R4
  log "stopped (run4.sh exit $?)"
  rm -f "$SOCK"
  pgrep -x dsgb10_rank > /dev/null && "$SV/run4.sh" stop > "$D/$MODE/run4.stop2.log" 2>&1
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
"$FE" serve --listen "127.0.0.1:$PORT" --engine-socket "$SOCK" --api-key-file "$KEY" --hold-s 120 \
  > "$D/$MODE/front.log" 2>&1 < /dev/null &
FP=$!
for ((i = 0; i < 60; i++)); do
  [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ] && break
  sleep 0.5
done
log "dsfront: pid $FP, /health $(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")"
LIM=420
[ "$MODE" = prefill ] && LIM=360
SD_ROOT=$SD_ROOT BENCH_HISTORY_PATH=$D/$MODE/bench-history.json BENCH_ACTIVE_PATH=$D/$MODE/bench-active.json \
  PREFILL_BENCH_HISTORY_PATH=$D/$MODE/prefill-history.json PREFILL_BENCH_ACTIVE_PATH=$D/$MODE/prefill-active.json \
  timeout $LIM node "$HERE/sdrun.mjs" "$MODE" "$PORT" "$KEY" "$D/$MODE/sd.jsonl" "${CORPUS:-}" > "$D/$MODE/sdrun.log" 2>&1
log "sdrun: exit $?: $(wc -l < "$D/$MODE/sd.jsonl") records; $(tail -n 1 "$D/$MODE/sdrun.log")"
cleanup
log "the log: $(grep -c "serve: req [0-9]* session" "$L") requests, $(grep -c -i "error\|abort\|watchdog" "$L") error lines"
