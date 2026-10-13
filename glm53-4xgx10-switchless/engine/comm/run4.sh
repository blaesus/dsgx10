#!/bin/bash
# Run one ring binary on all four boxes (the DeepSeek engine's engine/comm/run4.sh pattern): ranks 1-3 detached on
# the workers, rank 0 here in the foreground. The binary and the fleet file (FLEET=FILE; default the DeepSeek
# engine's fleet/homelab.conf, the same boxes) go to ~/dsgb10-data/glm53-comm/run/ on every box, the fleet file
# as fleet.conf beside the binary; logs in run/logs/<name>.<rank>.log there; worker log tails are printed after.
# GPU and ring code only through the DeepSeek engine's ring.sh (it lends the ring while the service is idle and
# preempts the job for a request: then this script's TERM trap stops the workers' ranks too):
#   OWNER=glm53-comm ds41f-4xgx10-switchless/engine/server/ring.sh glm53-4xgx10-switchless/engine/comm/run4.sh <binary> [args...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../../../ds41f-4xgx10-switchless/engine/server/fleet.sh"
fleet_load "$FLEET" || exit 2
BIN=$1
shift
NAME=$(basename "$BIN")
RD='~/dsgb10-data/glm53-comm/run'  # as written, for the workers' shells
LRD=$(fleet_local "$RD")
PAT="[d]sgb10-data/glm53-comm/run/$NAME"  # [d]: never matches the shell running pkill
mkdir -p "$LRD/logs"
cp "$BIN" "$LRD/$NAME"
cp "$FLEET" "$LRD/fleet.conf"
stop_workers() {
  local w
  for w in "${FLEET_WORKERS[@]}"; do ssh -o BatchMode=yes -o ConnectTimeout=10 "$w" "pkill -f '$PAT'" & done
  wait
}
trap 'stop_workers; exit 143' TERM INT HUP
for w in "${FLEET_WORKERS[@]}"; do
  ssh -o BatchMode=yes "$w" "mkdir -p $RD/logs" && scp -q "$BIN" "$w:$(fleet_rel "$RD")/$NAME" || exit 3
done
r=1
for w in "${FLEET_WORKERS[@]}"; do
  # pkill in a call of its own: the launch's command line names the binary's path, which the pattern matches
  ssh -o BatchMode=yes "$w" "pkill -f '$PAT'"
  ssh -o BatchMode=yes "$w" "cat > $RD/fleet.conf; setsid nohup timeout 600 $RD/$NAME --rank $r $* > $RD/logs/$NAME.$r.log 2>&1 < /dev/null &" < "$FLEET"
  r=$((r + 1))
done
timeout 600 "$LRD/$NAME" --rank 0 "$@" 2>&1 | tee "$LRD/logs/$NAME.0.log" &
wait $!
rc0=$(tail -n 1 "$LRD/logs/$NAME.0.log" | grep -q ': passed;' && echo 0 || echo 1)
sleep 2
r=1
for w in "${FLEET_WORKERS[@]}"; do
  echo "--- rank $r"
  ssh -o BatchMode=yes "$w" "tail -n 3 $RD/logs/$NAME.$r.log"
  ssh -o BatchMode=yes "$w" "pkill -f '$PAT'"
  r=$((r + 1))
done
exit "$rc0"
