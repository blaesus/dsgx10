#!/bin/bash
# Agentic turns through a test server (the engine on the four boxes, its socket /tmp/dsgb10-engine.sock):
#   serve_turns.sh LABEL OUTDIR [cold prompt sizes...]        (turns.py's; default 16384 65536 131072 250000)
# Starts run4.sh --mode serve at --max-ctx MAXCTX (266240), runs turns.py through it, then with CEDTEST=1
# serve_test.py --ced against the same server; stops it and joins rank 0's turn lines with the turns
# (tjoin.py). Out, in OUTDIR: srv_LABEL.log (rank 0), turns_LABEL.log, tj_LABEL.txt (the table for
# turns_cmp.py), st_LABEL.log; progress lines in serve_turns.log.
# Env: BIN, OWNER, TIMEOUT as run4.sh takes them (OWNER attention, TIMEOUT 1800); SETENV="NAME=value"
# (spaces escaped for the workers, as run_hash.sh); TOKENS (turns.py's token file). Under ring.sh (a part
# of ringparts.sh) the server starts at once; on its own, run4.sh waits for the ring.
set -u
[ $# -ge 2 ] || {
  sed -n '2,10p' "$0" >&2
  exit 2
}
LABEL=$1
OUT=$(realpath -m "$2")
shift 2
T=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
P=$OUT/serve_turns.log
L=$OUT/srv_$LABEL.log
say() { echo "$(date '+%F %T') $LABEL: $*" >> "$P"; }
SE=()
[ -n "${SETENV:-}" ] && SE=(--setenv "${SETENV// /\\ }")
TA=()
[ -n "${TOKENS:-}" ] && TA=(--tokens "$TOKENS")
cd "$T/../../.." || exit 2
say "serve${SETENV:+, $SETENV}"
OWNER=${OWNER:-attention} TIMEOUT=${TIMEOUT:-1800} engine/server/run4.sh --mode serve --max-ctx "${MAXCTX:-266240}" \
  "${SE[@]}" > "$L" 2>&1 < /dev/null &
SP=$!
for _ in $(seq 300); do
  grep -q "serve: listening" "$L" && break
  kill -0 $SP 2> /dev/null || break
  sleep 2
done
rc=1
if grep -q "serve: listening" "$L"; then
  say "serve up"
  timeout 1800 python3 "$T/turns.py" "${TA[@]}" "$@" > "$OUT/turns_$LABEL.log" 2>&1
  rc=$?
  say "turns exit $rc"
  if [ "${CEDTEST:-0}" = 1 ]; then
    timeout 900 python3 engine/server/tests/serve_test.py --ced > "$OUT/st_$LABEL.log" 2>&1
    st=$?
    say "serve_test --ced exit $st: $(tail -1 "$OUT/st_$LABEL.log")"
    [ $st = 0 ] || rc=$st
  fi
  engine/server/run4.sh stop
else
  say "the server did not come up"
fi
wait $SP
python3 "$T/tjoin.py" "$OUT/turns_$LABEL.log" "$L" > "$OUT/tj_$LABEL.txt" 2>&1
say "done"
exit $rc
