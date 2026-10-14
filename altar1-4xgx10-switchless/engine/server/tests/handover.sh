#!/bin/bash
# The service's first start on the ring beside the DeepSeek service (A6b; glm53's G6b protocol, docs/serving.md § 7):
# service.sh start; one altar-1 request (64 tokens); one DeepSeek request while altar-1 holds the ring (32 tokens: the
# DeepSeek service reclaims it); one altar-1 request while the DeepSeek service holds the ring (it waits for the
# DeepSeek service's idle time, then the lend); service.sh stop. Every event timestamped; about 4-6 min of ring time.
# Run from rank 0, not through ring.sh (the service borrows the ring itself):
#   DS_KEY_FILE=FILE altar1-4xgx10-switchless/engine/server/tests/handover.sh
# DS_KEY_FILE: the DeepSeek service's API key file (its run/settings name it); altar-1's key: ALTAR1_KEY_FILE
# (~/dsgb10-data/dev-api-key). DS_URL (http://127.0.0.1:8888), ALTAR1 service settings from the environment.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
SVC=$HERE/service.sh
. "$HERE/../../../ds41f-4xgx10-switchless/engine/server/fleet.sh"
fleet_load "${FLEET:-$HERE/../../fleet/homelab.conf}" || exit 2
OWNER_F=$(dirname "$(fleet_local "$FLEET_RING_LOCK")")/ring.owner
SLOG=${ALTAR1_HOME:-$(fleet_local "$FLEET_SERVICE_DIR")}/logs/service.log
AKEY=${ALTAR1_KEY_FILE:-$HOME/dsgb10-data/dev-api-key}
DKEY=${DS_KEY_FILE:?the DeepSeek service\'s API key file}
AURL=http://127.0.0.1:${ALTAR1_PORT:-18895}
DURL=${DS_URL:-http://127.0.0.1:8888}
ts() { date '+%T.%3N'; }
say() { echo "$(ts) $*"; }
req() {  # url key model max_tokens text: one chat completion, its timing and usage
  local t0 t1 out
  t0=$(date +%s.%N)
  out=$(curl -s -m 900 "$1/v1/chat/completions" -H "Authorization: Bearer $(cat "$2")" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$3\",\"max_tokens\":$4,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"$5\"}]}")
  t1=$(date +%s.%N)
  echo "$(ts) $3: $(echo "$t1 - $t0" | bc) s; $(echo "$out" | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin); u = d.get("usage", {}); c = d["choices"][0]["message"]
    print("usage", u, "text", repr(((c.get("reasoning_content") or "") + (c.get("content") or ""))[:120]))
except Exception as e:
    print("no answer:", e)')"
}
owner() { say "ring.owner: $(cut -c1-140 "$OWNER_F")"; }
[ -r "$DKEY" ] || { echo "no DeepSeek key file $DKEY"; exit 2; }
say "DeepSeek service: $(~/dsgb10/prod/ds41f-4xgx10-switchless/engine/server/service.sh status | head -n 1)"
owner
mkdir -p "$(dirname "$SLOG")"
{ wc -l < "$SLOG" || echo 0; } > "$SLOG.n0" 2> /dev/null
say "== service.sh start"
"$SVC" start
say "start returned (rc $?)"
owner
say "== an altar-1 request (64 tokens)"
req "$AURL" "$AKEY" altar-1 64 "Write a haiku about a ring of four small computers."
say "== a DeepSeek request while altar-1 holds the ring (32 tokens)"
req "$DURL" "$DKEY" deepseek-v4.1-flash-uncensored 32 "Say hello in five words." &
D=$!
n0=$(cat "$SLOG.n0" 2> /dev/null || echo 0)
for i in $(seq 120); do tail -n +$((n0 + 1)) "$SLOG" 2> /dev/null | grep -q 'giving the ring back' && break; sleep 0.5; done
wait $D
owner
say "== an altar-1 request while the DeepSeek service holds the ring (64 tokens)"
req "$AURL" "$AKEY" altar-1 64 "Name three prime numbers and say why they are prime."
owner
say "== service.sh stop"
"$SVC" stop
say "stop returned (rc $?)"
owner
say "altar1_rank left: rank 0 $(pgrep -xc altar1_rank)"
say "--- the service's log"
tail -n +$(($(cat "$SLOG.n0") + 1)) "$SLOG" | grep -E "state:|want|ring|engine (started|ready)|rank 0 exited|giving"
