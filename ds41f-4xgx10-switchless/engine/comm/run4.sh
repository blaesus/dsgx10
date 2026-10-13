#!/bin/bash
# Run one ring binary on all four boxes: ranks 1-3 detached on the workers, rank 0 here in the
# foreground. The boxes come from the fleet file (FLEET=FILE; default fleet/homelab.conf), which goes
# beside the binary as fleet.conf on every box. Logs: LOG_DIR/<name>.<rank>.log on each box
# (~/dsgb10/logs here); worker log tails are printed after.
# usage: engine/comm/run4.sh <binary> [args...]
set -u
. "$(dirname "$0")/../server/fleet.sh"
fleet_load "$FLEET" || exit 2
BIN=$1
shift
NAME=$(basename "$BIN")
BD=$FLEET_BIN_DIR LD=$FLEET_LOG_DIR  # as written, for the workers' shells (a leading ~/ is theirs)
LBD=$(fleet_local "$BD") LLD=$(fleet_local "$LD")
mkdir -p "$LBD" "$LLD"
cp "$BIN" "$LBD/$NAME"
cp "$FLEET" "$LBD/fleet.conf"
for w in "${FLEET_WORKERS[@]}"; do scp -q "$BIN" "$w:$(fleet_rel "$BD")/$NAME"; done
ARGS="$*"
r=1
for w in "${FLEET_WORKERS[@]}"; do
  ssh "$w" "cat > $BD/fleet.conf; pkill -x $NAME; mkdir -p $LD; setsid nohup timeout 900 $BD/$NAME --rank $r $ARGS > $LD/$NAME.$r.log 2>&1 < /dev/null &" < "$FLEET"
  r=$((r + 1))
done
timeout 900 "$LBD/$NAME" --rank 0 "$@" 2>&1 | tee "$LLD/$NAME.0.log"
rc=${PIPESTATUS[0]}
sleep 2
r=1
for w in "${FLEET_WORKERS[@]}"; do
  echo "--- rank $r ($(fleet_get "RANK${r}_NAME"))"; ssh "$w" "tail -n 4 $LD/$NAME.$r.log"
  r=$((r + 1))
done
exit "$rc"
