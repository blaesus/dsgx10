#!/bin/bash
# Run altar1_rank on all four boxes (docs/serving.md): ranks 1-3 detached on the workers, rank 0 here in the
# foreground, every rank with the same arguments; the binary (BIN, default ~/dsgb10-data/altar1-serve/bin/altar1_rank,
# installed by install.sh) and the fleet file (FLEET) go to ~/dsgb10-data/altar1-serve/run/ on every box, logs in
# run/logs/altar1_rank.<rank>.log there. TIMEOUT s (900) bounds every rank. Only on the ring, through the DeepSeek
# engine's ring.sh (it lends the ring while its service is idle and preempts the job for a request: the TERM trap
# here then stops the workers' ranks too):
#   OWNER=altar1-serve ds41f-4xgx10-switchless/engine/server/ring.sh altar1-4xgx10-switchless/engine/server/run4.sh \
#       --mode serve --fill-experts --socket /tmp/altar1-test.sock
#   engine/server/run4.sh stop        # every rank of this script's binary stopped, rank 0 first
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/ranks.sh"
if [ "${1:-}" = stop ]; then
  ranks_stop
  exit 0
fi
BIN=${BIN:-$HOME/dsgb10-data/altar1-serve/bin/$NAME}
trap 'ranks_stop; exit 143' TERM INT HUP
ranks_ship "$BIN" || { echo "run4.sh: copying the binary failed" >&2; exit 3; }
ranks_workers "$@"
timeout "${TIMEOUT:-900}" "$LRD/$NAME" --rank 0 "$@" 2>&1 | tee "$LRD/logs/$NAME.0.log" &
wait $!
rc=$?
sleep 2
ranks_tails 3
ranks_stop
exit "$rc"
