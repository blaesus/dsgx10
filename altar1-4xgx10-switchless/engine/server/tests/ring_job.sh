#!/bin/bash
# The server's tests on the ring, one job of a few minutes (docs/serving.md, "Tests"): the four ranks in serve mode
# (WEIGHTS: the weights' altar1_rank options, default --synthetic: every weight synthetic in the real formats, A4's
# pools; "" for the fleet's rank files, "--fill-experts" for rank files whose expert pools are not filled), a pool
# small enough for the eviction test and a disk tier of its own (tag "test", on rank 0 too: --disk-rank0-gib 8 with
# a free-space margin of 8 GiB); serve_test.py, batch_test.py, altarfront on 127.0.0.1:PORT (18895) in front of the
# engine and front_test.py; then a planned stop (SIGTERM, kept sessions) and a start, and serve_test.py --kept; then
# the drafts' check (spec): spec_test.py's requests recorded from a server started with --no-spec, then compared, bit
# for bit, on servers with --spec-drafts k --spec-force k --spec-oracle for every k of SPEC_KS (1 2 3: every step
# verifies k drafts; a second pass's drafts from the first pass's tokens) and on one with the gate (k up to 3). The
# other phases run with drafts on (the default). Prefill in chunks of 2,048 rows with 128 MiB of index scratch (the
# fleet's memory rule). Through the DeepSeek engine's ring.sh, from the repo root (a preempted run, exit 76, is run
# again):
#   OWNER=altar1-serve ds41f-4xgx10-switchless/engine/server/ring.sh altar1-4xgx10-switchless/engine/server/tests/ring_job.sh
# BIN: the engine (~/dsgb10-data/altar1-serve/bin/altar1_rank, install.sh); FRONT: altarfront
# (~/dsgb10-data/altar1-serve/bin/altarfront); OUT: results (~/dsgb10-data/altar1-serve/results.tsv); PHASES: main
# batch front kept spec (all by default; two jobs keep each under ten minutes: "main batch front kept", then "spec").
# SERVER_DATA: another data tree than ~/dsgb10-data/altar1-serve (BIN, OUT, the state and key files).
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
T=$HERE/tests
. "$HERE/ranks.sh"
DATA=${SERVER_DATA:-$HOME/dsgb10-data/altar1-serve}
BIN=${BIN:-$DATA/bin/$NAME}
FRONT=${FRONT:-$DATA/bin/altarfront}
OUT=${OUT:-$DATA/results.tsv}
PORT=${PORT:-18895}
PHASES=${PHASES:-main batch front kept spec}
SPEC_KS=${SPEC_KS:-1 2 3}
WEIGHTS=${WEIGHTS---synthetic}
SOCK=/tmp/altar1-test.$$.sock
STATE=$DATA/serve_state.json
KEY=$DATA/test-key
BASE_ARGS="--mode serve $WEIGHTS --socket $SOCK --kv-gib 1.2 --disk-gib 8 --disk-rank0-gib 8 --disk-margin-gib 8 --disk-tag test --keep-gib 4 --slots 4 --slice 16 --chunk 2048 --index-scratch-mib 128 ${EXTRA:-}"
ARGS=$BASE_ARGS
mkdir -p "$DATA"
[ -s "$KEY" ] || (umask 077 && head -c 24 /dev/urandom | base64 > "$KEY")
SRC=$(cat "$(dirname "$BIN")/SOURCES" 2> /dev/null)
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
      for k in $SPEC_KS; do
        stop
        ARGS="$BASE_ARGS --spec-drafts $k --spec-force $k --spec-oracle"
        start || { cleanup; exit 1; }
        grep -E "^drafts" "$LRD/logs/$NAME.0.log" | tail -n 1
        step spec_forced_k$k python3 "$T/spec_test.py" --socket "$SOCK" --compare "$DATA/spec_ref.json" --repeat 2 \
          --save "$DATA/spec_forced_k$k.json"
        grep -E "^serve: (req .*drafts|run of)" "$LRD/logs/$NAME.0.log" | tail -n 8
      done
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
cp "$LRD/logs/$NAME.0.log" "$DATA/rank0.$(date +%Y%m%d-%H%M%S).log" 2> /dev/null  # the whole job's rank 0 log
# rank 0's data disk has little room (DESIGN, rank 0's disk): the test tier's files go once the job is over
S0=$(fleet_local "$(fleet_rank "$(fleet_get RANK0_SPILL)" 0)")
[ -n "$S0" ] && rm -f "$S0"/rank0.test.spill "$S0"/rank0.test.index "$S0"/server.test.index
exit $RC
