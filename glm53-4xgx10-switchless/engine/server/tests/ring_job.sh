#!/bin/bash
# The server's tests on the ring, one job of a few minutes (docs/serving.md, "Tests"): the four ranks in serve mode
# on the fleet's rank files with synthetic routed experts (WEIGHTS: more glm53_rank options, default --fill-experts;
# "--synthetic" for no files), a pool small enough for the eviction test and a disk tier of its own (tag "test");
# serve_test.py, batch_test.py, glmfront on 127.0.0.1:PORT (18894) in front of the engine and front_test.py; then
# a planned stop (SIGTERM, kept sessions) and a start, and serve_test.py --kept; then the drafts' check (spec):
# spec_test.py's requests recorded from a server started with --no-spec, then compared, bit for bit, on one with
# --spec-force 3 --spec-oracle (every step verifies drafts; a second pass's drafts from the first pass's tokens)
# and on one with the gate. The other phases run with drafts on (the
# default). Through the DeepSeek engine's
# ring.sh, from the repo root (a preempted run, exit 76, is run again):
#   OWNER=glm53-server ds41f-4xgx10-switchless/engine/server/ring.sh glm53-4xgx10-switchless/engine/server/tests/ring_job.sh
# BIN: the engine (~/dsgb10-data/glm53-server/bin/glm53_rank, install.sh); FRONT: glmfront (frontend/target/release);
# OUT: results (~/dsgb10-data/glm53-server/results.tsv); PHASES: main batch front kept spec (all by default).
# SERVER_DATA: another data tree than ~/dsgb10-data/glm53-server (BIN, OUT, the state and key files).
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
T=$HERE/tests
. "$HERE/ranks.sh"
DATA=${SERVER_DATA:-$HOME/dsgb10-data/glm53-server}
BIN=${BIN:-$DATA/bin/$NAME}
FRONT=${FRONT:-$HERE/../../frontend/target/release/glmfront}
OUT=${OUT:-$DATA/results.tsv}
PORT=${PORT:-18894}
PHASES=${PHASES:-main batch front kept spec}
WEIGHTS=${WEIGHTS:---fill-experts}
SOCK=/tmp/glm53-test.$$.sock
STATE=$DATA/serve_state.json
KEY=$DATA/test-key
BASE_ARGS="--mode serve $WEIGHTS --socket $SOCK --kv-gib 1.2 --disk-gib 8 --disk-tag test --keep-gib 4 --slots 4 --slice 16 ${EXTRA:-}"
ARGS=$BASE_ARGS
mkdir -p "$DATA"
[ -s "$KEY" ] || (umask 077 && head -c 24 /dev/urandom | base64 > "$KEY")
SRC=$(cat "$DATA/bin/SOURCES" 2> /dev/null)
echo "RUN	$(date '+%F %T %z')	$SRC	$ARGS" >> "$OUT"
R0= FE=
cleanup() {
  [ -n "$FE" ] && kill "$FE" 2> /dev/null
  ranks_stop
  rm -f "$SOCK"
}
trap 'cleanup; exit 143' TERM INT HUP
start() {  # the four ranks; rank 0 in the background, until its socket is up
  local i
  rm -f "$SOCK"
  ranks_workers $ARGS
  "$LRD/$NAME" --rank 0 $ARGS >> "$LRD/logs/$NAME.0.log" 2>&1 < /dev/null &
  R0=$!
  for ((i = 0; i < 600; i++)); do
    [ -S "$SOCK" ] && return 0
    kill -0 "$R0" 2> /dev/null || break
    sleep 0.5
  done
  echo "rank 0 did not come up:"
  tail -n 20 "$LRD/logs/$NAME.0.log"
  ranks_tails 10
  return 1
}
stop() {  # a planned stop: SIGTERM to rank 0 (it keeps the sessions and stops the others)
  local i
  kill -TERM "$R0"
  for ((i = 0; i < 240; i++)); do kill -0 "$R0" 2> /dev/null || break; sleep 0.5; done
  wait "$R0"
  echo "rank 0 exited: rc $?"
  sleep 2
  ranks_stop
}
step() {  # name command...: run and record
  local name=$1 rc
  shift
  echo "== $name $(date '+%T')"
  "$@"
  rc=$?
  echo "$([ $rc = 0 ] && echo DONE || echo FAILED)	$name	rc $rc	$(date '+%T')	$SRC" >> "$OUT"
  RC=$((RC + (rc != 0)))
}
RC=0
ranks_ship "$BIN" || exit 3
: > "$LRD/logs/$NAME.0.log"
start || { cleanup; exit 1; }
grep -E "^(disk tier|cache pool|weights|serve: listening)" "$LRD/logs/$NAME.0.log"
for p in $PHASES; do
  case $p in
    main) step serve_test python3 "$T/serve_test.py" --socket "$SOCK" --state "$STATE" ;;
    batch) step batch_test python3 "$T/batch_test.py" --socket "$SOCK" ;;
    front)
      "$FRONT" serve --listen "127.0.0.1:$PORT" --engine-socket "$SOCK" --api-key-file "$KEY" > "$DATA/front.log" 2>&1 &
      FE=$!
      sleep 3
      step front_test python3 "$T/front_test.py" "http://127.0.0.1:$PORT" "$KEY"
      kill "$FE"
      wait "$FE" 2> /dev/null
      FE=
      ;;
    kept)
      stop
      grep -E "kept sessions" "$LRD/logs/$NAME.0.log" | tail -n 1
      start || { cleanup; exit 1; }
      grep -E "kept sessions|disk tier" "$LRD/logs/$NAME.0.log" | tail -n 3
      step kept_test python3 "$T/serve_test.py" --socket "$SOCK" --state "$STATE" --kept
      ;;
    spec)
      stop
      ARGS="$BASE_ARGS --no-spec"
      start || { cleanup; exit 1; }
      grep -E "^drafts" "$LRD/logs/$NAME.0.log" | tail -n 1
      step spec_record python3 "$T/spec_test.py" --socket "$SOCK" --record "$DATA/spec_ref.json"
      stop
      ARGS="$BASE_ARGS --spec-force 3 --spec-oracle"
      start || { cleanup; exit 1; }
      grep -E "^drafts" "$LRD/logs/$NAME.0.log" | tail -n 1
      step spec_forced python3 "$T/spec_test.py" --socket "$SOCK" --compare "$DATA/spec_ref.json" --repeat 2 \
        --save "$DATA/spec_forced.json"
      grep -E "^serve: req .*drafts" "$LRD/logs/$NAME.0.log" | tail -n 8
      stop
      ARGS=$BASE_ARGS
      start || { cleanup; exit 1; }
      grep -E "^drafts" "$LRD/logs/$NAME.0.log" | tail -n 1
      step spec_gate python3 "$T/spec_test.py" --socket "$SOCK" --compare "$DATA/spec_ref.json" --save "$DATA/spec_gate.json"
      ;;
  esac
done
stop
echo "--- rank 0 log (serve lines)"
grep -E "^serve: (req|run|session|pool|kept)" "$LRD/logs/$NAME.0.log" | tail -n 60
ranks_tails 3
exit $RC
