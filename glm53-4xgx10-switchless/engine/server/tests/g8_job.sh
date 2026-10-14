#!/bin/bash
# G8 (docs/engine.md, "End to end on real weights"): the four ranks in serve mode on the fleet's rank files with
# every routed expert filled, glmfront on 127.0.0.1:PORT (18894) in front, and g8_front.py's requests (the four
# golden conversations, greedy and seeded, up to MAX new tokens; then two sessions at once). One server
# configuration a job (each start loads 101.4 GiB a rank, ~70 s):
#   OWNER=glm53-g8 ds41f-4xgx10-switchless/engine/server/ring.sh glm53-4xgx10-switchless/engine/server/tests/g8_job.sh CFG
# CFG: plain (--no-spec), k1, k2, k3 (--spec-force K: K drafts every step), gate (the default: drafts by the gate).
# Memory: the pool capped at KVGIB (2) with margins of 1 GiB, prefill chunks of CHUNK (2048) rows, no disk tier.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
T=$HERE/tests
. "$HERE/ranks.sh"
D=$HOME/dsgb10-data/glm53-g8
BIN=${BIN:-$D/server/bin/$NAME}
FRONT=${FRONT:-$HERE/../../frontend/target/release/glmfront}
REF=$HERE/../../ref/tokens
PORT=${PORT:-18894}
CFG=${1:?plain, k1, k2, k3 or gate}
case $CFG in
  plain) SPEC="--no-spec" ;;
  k1 | k2 | k3) SPEC="--spec-force ${CFG#k}" ;;
  gate) SPEC="" ;;
  *) echo "unknown configuration $CFG"; exit 2 ;;
esac
STAMP=$(date +%H%M%S)
SOCK=/tmp/glm53-g8.$$.sock
KEY=$D/front-key
DUMP=$D/front/$CFG.$STAMP
OUTJ=$D/front/$CFG.$STAMP.json
ARGS="--mode serve --socket $SOCK --max-ctx ${MAXCTX:-8192} --chunk ${CHUNK:-2048} --kv-gib ${KVGIB:-2} --margin-gib ${MARGIN:-0.25}
  --worker-margin-gib 1 --disk-gib 0 --slots 4 --index-scratch-mib 128 $SPEC ${EXTRA:-}"
export TIMEOUT=${TIMEOUT:-600}
[ -x "$FRONT" ] || { echo "no glmfront at $FRONT (FRONT=...: a worktree has no frontend build)"; exit 2; }
mkdir -p "$DUMP"
[ -s "$KEY" ] || (umask 077 && head -c 24 /dev/urandom | base64 > "$KEY")
echo "g8 $CFG $(date '+%F %T %z') $(cat "$D/server/bin/SOURCES" 2> /dev/null) args: $ARGS"
R0= FE=
cleanup() {
  [ -n "$FE" ] && kill "$FE" 2> /dev/null
  ranks_stop
  rm -f "$SOCK"
}
trap 'cleanup; exit 143' TERM INT HUP
ranks_ship "$BIN" || exit 3
: > "$LRD/logs/$NAME.0.log"
t0=$(date +%s.%N)
ranks_workers $ARGS
"$LRD/$NAME" --rank 0 $ARGS >> "$LRD/logs/$NAME.0.log" 2>&1 < /dev/null &
R0=$!
up=0
for ((i = 0; i < 600; i++)); do
  [ -S "$SOCK" ] && { up=1; break; }
  kill -0 "$R0" 2> /dev/null || break
  sleep 0.5
done
if [ $up = 0 ]; then
  echo "rank 0 did not come up:"
  tail -n 20 "$LRD/logs/$NAME.0.log"
  ranks_tails 10
  cleanup
  exit 1
fi
echo "serving after $(echo "$(date +%s.%N) - $t0" | bc) s"
grep -E "^(disk tier|cache pool|weights|drafts|serve: listening|memory)" "$LRD/logs/$NAME.0.log"
"$FRONT" serve --listen "127.0.0.1:$PORT" --engine-socket "$SOCK" --api-key-file "$KEY" --dump-dir "$DUMP" \
  --tokenize-threads 4 > "$D/front/$CFG.$STAMP.front.log" 2>&1 &
FE=$!
sleep 3
python3 "$T/g8_front.py" run "http://127.0.0.1:$PORT" "$KEY" "$REF" "$DUMP" "$OUTJ" --max "${MAX:-512}" \
  --phases "${PHASES:-single,pair}"
rc=$?
kill "$FE"
wait "$FE" 2> /dev/null
FE=
echo "--- glmfront"
grep -E "^req " "$D/front/$CFG.$STAMP.front.log" | tail -n 20
kill -TERM "$R0"
for ((i = 0; i < 120; i++)); do kill -0 "$R0" 2> /dev/null || break; sleep 0.5; done
sleep 1
ranks_stop
rm -f "$SOCK"
echo "--- rank 0 log (serve lines)"
grep -E "^serve: (req|run)" "$LRD/logs/$NAME.0.log" | tail -n 40
echo "results: $OUTJ"
exit $rc
