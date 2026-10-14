#!/bin/bash
# A7 (docs/serving.md, "Drafts on real text"; glm53's G8 pattern): the four ranks in serve mode on the fleet's rank files with
# every routed expert filled, altarfront on 127.0.0.1:PORT (18895) in front, and g8_front.py's requests (the four
# golden conversations, greedy and seeded, up to MAX new tokens; then two sessions at once). One server
# configuration a job (each start loads 76.6 GiB a rank):
#   OWNER=altar1-g8 ds41f-4xgx10-switchless/engine/server/ring.sh altar1-4xgx10-switchless/engine/server/tests/g8_job.sh CFG
# CFG: plain (--no-spec), k1, k2, k3 (--spec-drafts K --spec-force K: K drafts every step, K passes), gate (the
# default: up to 3 drafts by the gate; GATE_COST C1,DROW: its cost model).
# Memory: the pool capped at KVGIB (2), the engine's 8 GiB guard, prefill chunks of CHUNK (2048) rows, no disk tier.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
T=$HERE/tests
. "$HERE/ranks.sh"
D=$HOME/dsgb10-data/altar1-g8
BIN=${BIN:-$HOME/dsgb10-data/altar1-serve/bin/$NAME}
FRONT=${FRONT:-$HOME/dsgb10-data/altar1-serve/bin/altarfront}
REF=$HERE/../../ref/tokens
PORT=${PORT:-18895}
CFG=${1:?plain, k1, k2, k3 or gate}
case $CFG in
  plain) SPEC="--no-spec" ;;
  k1 | k2 | k3) SPEC="--spec-drafts ${CFG#k} --spec-force ${CFG#k}" ;;
  gate) SPEC="${GATE_COST:+--spec-cost $GATE_COST}" ;;
  *) echo "unknown configuration $CFG"; exit 2 ;;
esac
STAMP=$(date +%H%M%S)
SOCK=/tmp/altar1-g8.$$.sock
KEY=$D/front-key
DUMP=$D/front/$CFG.$STAMP
OUTJ=$D/front/$CFG.$STAMP.json
ARGS="--mode serve --socket $SOCK --max-ctx ${MAXCTX:-8192} --chunk ${CHUNK:-2048} --kv-gib ${KVGIB:-2}
  --disk-gib 0 --slots 4 --index-scratch-mib 128 $SPEC ${EXTRA:-}"
export TIMEOUT=${TIMEOUT:-600}
[ -x "$FRONT" ] || { echo "no altarfront at $FRONT (FRONT=...: a worktree has no frontend build)"; exit 2; }
mkdir -p "$DUMP"
[ -s "$KEY" ] || (umask 077 && head -c 24 /dev/urandom | base64 > "$KEY")
echo "g8 $CFG $(date '+%F %T %z') $(cat "$(dirname "$BIN")/SOURCES" 2> /dev/null) args: $ARGS"
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
echo "--- altarfront"
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
