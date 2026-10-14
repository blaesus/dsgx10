#!/bin/bash
# Run one ring binary on all four boxes (comm/run4.sh's copy for the model's driver): ranks 1-3 detached on
# the workers, rank 0 here in the foreground. The binary and the fleet file (FLEET=FILE; default this configuration's
# fleet/homelab.conf, the same boxes as the DeepSeek engine's) go to ~/dsgb10-data/$DATA_NAME/run/ on every box (DATA_NAME:
# default glm53-model), the fleet file as fleet.conf beside the binary; logs in run/logs/<name>.<rank>.log there; worker log
# tails are printed after. NSYS=1: every rank runs under nsys (CUDA trace, graph nodes, the cudaProfilerApi range), the
# reports collected into run/nsys/<name>.<rank>.nsys-rep on this box.
# GPU and ring code only through the DeepSeek engine's ring.sh (it lends the ring while the service is idle and
# preempts the job for a request: then this script's TERM trap stops the workers' ranks too):
#   OWNER=glm53-model ds41f-4xgx10-switchless/engine/server/ring.sh glm53-4xgx10-switchless/engine/model/run4.sh <binary> [args...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
FLEET=${FLEET:-$HERE/../../fleet/homelab.conf}
. "$HERE/../../../ds41f-4xgx10-switchless/engine/server/fleet.sh"
fleet_load "$FLEET" || exit 2
BIN=$1
shift
NAME=$(basename "$BIN")
DATA_NAME=${DATA_NAME:-glm53-model}
RD="~/dsgb10-data/$DATA_NAME/run"  # as written, for the workers' shells
LRD=$(fleet_local "$RD")
PAT="[d]sgb10-data/$DATA_NAME/run/$NAME"  # [d]: never matches the shell running pkill
NS=""
[ "${NSYS:-0}" = 1 ] && NS="/usr/local/cuda-13.0/bin/nsys profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o"
pre() { [ -n "$NS" ] && echo "$NS $2/nsys/$NAME.$1"; }
mkdir -p "$LRD/logs" "$LRD/nsys"
cp "$BIN" "$LRD/$NAME"
cp "$FLEET" "$LRD/fleet.conf"
stop_workers() {
  local w
  for w in "${FLEET_WORKERS[@]}"; do ssh -o BatchMode=yes -o ConnectTimeout=10 "$w" "pkill -f '$PAT'" & done
  wait
}
trap 'stop_workers; exit 143' TERM INT HUP
for w in "${FLEET_WORKERS[@]}"; do
  ssh -o BatchMode=yes "$w" "mkdir -p $RD/logs $RD/nsys" && scp -q "$BIN" "$w:$(fleet_rel "$RD")/$NAME" || exit 3
done
r=1
for w in "${FLEET_WORKERS[@]}"; do
  # pkill in a call of its own: the launch's command line names the binary's path, which the pattern matches
  ssh -o BatchMode=yes "$w" "pkill -f '$PAT'"
  ssh -o BatchMode=yes "$w" "cat > $RD/fleet.conf; setsid nohup timeout 600 $(pre $r "$RD") $RD/$NAME --rank $r $* > $RD/logs/$NAME.$r.log 2>&1 < /dev/null &" < "$FLEET"
  r=$((r + 1))
done
timeout 600 $(pre 0 "$LRD") "$LRD/$NAME" --rank 0 "$@" 2>&1 | tee "$LRD/logs/$NAME.0.log" &
wait $!
rc0=$(tail -n 12 "$LRD/logs/$NAME.0.log" | grep -q '^rank 0: passed;' && echo 0 || echo 1)
sleep 2
r=1
for w in "${FLEET_WORKERS[@]}"; do
  echo "--- rank $r"
  ssh -o BatchMode=yes "$w" "tail -n 3 $RD/logs/$NAME.$r.log"
  # under nsys: let the report be written before anything is stopped
  [ -n "$NS" ] && ssh -o BatchMode=yes "$w" "for i in \$(seq 90); do pgrep -f '$PAT' > /dev/null || break; sleep 1; done"
  ssh -o BatchMode=yes "$w" "pkill -f '$PAT'"
  [ -n "$NS" ] && scp -q "$w:$(fleet_rel "$RD")/nsys/$NAME.$r.nsys-rep" "$LRD/nsys/"
  r=$((r + 1))
done
exit "$rc0"
