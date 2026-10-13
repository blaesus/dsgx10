#!/bin/bash
# One ring part of the time-to-first-token study (docs/engine.md, "A small turn's time to first token"): the
# engine BIN as a test server (run4.sh --mode serve on the test socket), ttft_turns.py through it, then stop:
#   OWNER=runner PREEMPT=0 engine/server/ring.sh bash ttft_part.sh LABEL OUTDIR BIN [engine args...]
# Engine args go after --mode serve --max-ctx 262144 (production's), e.g. --profile-from-req 1 for nsys.
# Env: WRAP / WRAP_RANK as run4.sh takes them (nsys on that rank; a worker's report, NSYS_REMOTE on that box,
# is fetched into OUTDIR once its nsys has finished), ROUTES (a directory: rank 0 dumps every pass's routes
# there, a build with -DDSGB10_NVTX), TURNS_ARGS (ttft_turns.py's), TIMEOUT (run4.sh's, 900).
# Out, in OUTDIR: srv_LABEL.log (rank 0's log), turns_LABEL.log (the driver's rows); progress lines in
# OUTDIR/parts.log.
set -u
[ $# -ge 3 ] || {
  sed -n '2,10p' "$0" >&2
  exit 2
}
LABEL=$1
OUT=$(realpath -m "$2")
BIN=$(realpath "$3")
shift 3
T=$(cd "$(dirname "$0")" && pwd)
SV=$T/..
mkdir -p "$OUT"
P=$OUT/parts.log
L=$OUT/srv_$LABEL.log
say() { echo "$(date '+%F %T') $LABEL: $*" >> "$P"; }
[ -n "${RING_HOLDER:-}" ] || {
  echo "ttft_part.sh: run me under ring.sh" >&2
  exit 2
}
say "start: $(sha256sum "$BIN" | cut -c1-16) ${WRAP:+(WRAP rank ${WRAP_RANK:-0})}${ROUTES:+(routes to $ROUTES)} $*"
EV=()
[ -n "${ROUTES:-}" ] && mkdir -p "$ROUTES" && EV=(DSGB10_ROUTE_DUMP="$ROUTES")
env "${EV[@]}" BIN="$BIN" TIMEOUT="${TIMEOUT:-900}" "$SV/run4.sh" --mode serve --max-ctx 262144 "$@" > "$L" 2>&1 < /dev/null &
SP=$!
for _ in $(seq 300); do
  grep -q "serve: listening" "$L" && break
  kill -0 $SP 2> /dev/null || break
  sleep 2
done
rc=1
if grep -q "serve: listening" "$L"; then
  say "serve up"
  # shellcheck disable=SC2086
  timeout 600 python3 "$T/ttft_turns.py" ${TURNS_ARGS:-} > "$OUT/turns_$LABEL.log" 2>&1
  rc=$?
  say "turns exit $rc: $(grep -c '^row' "$OUT/turns_$LABEL.log") rows"
  "$SV/run4.sh" stop > /dev/null 2>&1
else
  say "the server did not come up"
  "$SV/run4.sh" stop > /dev/null 2>&1
fi
wait $SP
if [ -n "${WRAP:-}" ] && [ "${WRAP_RANK:-0}" != 0 ] && [ -n "${NSYS_REMOTE:-}" ]; then
  W=(x 10.0.0.4 10.0.0.3 10.0.0.2)
  h=${W[$WRAP_RANK]}
  for _ in $(seq 90); do
    ssh "$h" "pgrep -x nsys > /dev/null" || break
    sleep 2
  done
  scp -q "$h:$NSYS_REMOTE.nsys-rep" "$OUT/" && ssh "$h" "rm -f $NSYS_REMOTE.nsys-rep" &&
    say "fetched $(basename "$NSYS_REMOTE").nsys-rep from rank $WRAP_RANK" || say "no report from rank $WRAP_RANK"
fi
say "done (exit $rc)"
exit $rc
