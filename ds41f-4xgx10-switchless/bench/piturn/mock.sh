#!/bin/bash
# part.sh's harness against dsfront's mock engine (no ring, no GPU): pidrive.py's three modes on two tasks
# whose prompt asks the mock for a read call, its result, an answer; then piturn.py on the dump (gaps only:
# a mock has no tails). dsfront on 127.0.0.1:18897 (models.json's port).
#   FE=dsfront bash bench/piturn/mock.sh OUT
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
D=$1
KEY=${KEY:-$HOME/dsgb10-data/dev-api-key}
PORT=18897
SOCK=${RUN:-$HOME/dsgb10/editrun}/mock-pi.sock  # a short path: a socket's path has a limit
[ -x "${FE:-}" ] && command -v pi > /dev/null || {
  echo "FE and pi are needed"
  exit 2
}
ss -ltn | grep -q "127.0.0.1:$PORT " && {
  echo "port $PORT is in use"
  exit 2
}
mkdir -p "$D/dump" "$D/agent" "$(dirname "$SOCK")" && chmod 700 "$D" "$D/dump" "$D/agent"
cp "$HERE/models.json" "$D/agent/models.json"
rm -f "$SOCK"
"$FE" mock-engine --socket "$SOCK" --delay-ms 2 --max-context 262144 > "$D/mock.log" 2>&1 &
MP=$!
for ((i = 0; i < 100; i++)); do
  [ -S "$SOCK" ] && break
  sleep 0.2
done
"$FE" serve --listen "127.0.0.1:$PORT" --engine-socket "$SOCK" --api-key-file "$KEY" --hold-s 120 --dump-dir "$D/dump" \
  > >(python3 "$HERE/stamp.py" > "$D/front.log") 2>&1 < /dev/null &
FP=$!
for ((i = 0; i < 60; i++)); do
  [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ] && break
  sleep 0.5
done
echo "mock: health $(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")"
read -r DSGB10_DEV_KEY < "$KEY"
export DSGB10_DEV_KEY PI_CODING_AGENT_DIR=$D/agent
for m in text json tui; do
  timeout 120 python3 "$HERE/pidrive.py" "$D/pi" --dump "$D/dump" --mode $m --tasks fix_comment,rename_helper --timeout 60 \
    --prompt 'MOCK_TOOL read {"path": "engine/server/npy.h"}' > "$D/pidrive-$m.log" 2>&1
  echo "mock $m: exit $?; $(tr '\n' '|' < "$D/pidrive-$m.log" | cut -c1-400)"
done
unset DSGB10_DEV_KEY
kill -TERM "$FP" "$MP"
wait "$FP" "$MP" 2> /dev/null
rm -f "$SOCK"
touch "$D/run4.log"
python3 "$HERE/piturn.py" "$D/pi" "$D/dump" "$D/run4.log" "$D/front.log"
