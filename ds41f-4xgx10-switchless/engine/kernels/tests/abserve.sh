#!/bin/bash
# abserve.sh <binary> <label> <spec|k1> <out dir> [server args...]: one server session of an engine A/B, run
# under the ring (ring.sh or a job of parts, PREEMPT=0; at most ~9 minutes): <binary> serving on its own socket
# (SOCK, default /tmp/dsgb10-kernels.sock) with production's arguments (--max-ctx 262144; serve's defaults: CED,
# drafts on the FP8 drafter head) but no disk tier, and --no-spec for k1, plus the server args (e.g. --no-l2pf,
# or --setenv NAME=VALUE); abload.py's workload through the socket; then a planned stop. Writes
# <out>/ab_<label>_server.log, ab_<label>.log, ab_<label>.json (abcmp.py's input).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$HERE/../../.." && pwd)
SOCK=${SOCK:-/tmp/dsgb10-kernels.sock}
B=$1
LB=$2
M=$3
D=$4
shift 4
case $M in
  spec) X="" ;;
  k1) X="--no-spec" ;;
  *) echo "spec or k1, not $M"; exit 2 ;;
esac
SL=$D/ab_${LB}_server.log
cd "$R" || exit 2
t0=$SECONDS
echo "$(date '+%F %T') $LB: $B ($(sha256sum "$B" | cut -c1-16)) $M $*"
BIN=$B OWNER=${OWNER:-kernels} TIMEOUT=560 setsid engine/server/run4.sh --mode serve --socket $SOCK \
  --max-ctx 262144 --disk-gib 0 $X "$@" > "$SL" 2>&1 < /dev/null &
pid=$!
until grep -q "serve: listening" "$SL" 2> /dev/null; do
  if ! kill -0 $pid 2> /dev/null || ((SECONDS - t0 > 150)); then
    echo "the server is not listening after $((SECONDS - t0)) s"
    engine/server/run4.sh stop
    wait $pid
    exit 1
  fi
  sleep 2
done
echo "the server listens after $((SECONDS - t0)) s"
timeout 360 python3 "$HERE/abload.py" --socket $SOCK --out "$D/ab_$LB.json" --label "$LB $(basename "$B") $M${*:+ $*}" \
  > "$D/ab_$LB.log" 2>&1
rc=$?
engine/server/run4.sh stop
wait $pid
echo "abload.py exit $rc after $((SECONDS - t0)) s: $(tail -n 1 "$D/ab_$LB.log")"
exit $rc
