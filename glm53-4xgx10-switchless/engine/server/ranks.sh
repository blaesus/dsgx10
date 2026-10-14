# The four ranks of glm53_rank for jobs and tests (docs/serving.md), sourced by run4.sh and tests/ring_job.sh: the
# binary and the fleet file shipped to RD (~/dsgb10-data/glm53-server/run) on every box (engine/comm/run4.sh's
# pattern), ranks 1-3 started detached on the workers, rank 0 here. Only inside a ring job (the DeepSeek engine's
# ring.sh): the ranks need the ring.
HERE=${HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
FLEET=${FLEET:-$HERE/../../fleet/homelab.conf}
. "$HERE/../../../ds41f-4xgx10-switchless/engine/server/fleet.sh"
fleet_load "$FLEET" || exit 2
# a ring job is preemptible: under memory pressure (earlyoom on every box) it goes first, not the long CPU passes
# beside it (the reference, the converter); the workers' ranks get the same below
echo 1000 > /proc/self/oom_score_adj
NAME=glm53_rank
RD='~/dsgb10-data/glm53-server/run'  # as written, for the workers' shells
LRD=$(fleet_local "$RD")
PAT="[d]sgb10-data/glm53-server/run/$NAME"  # [d]: never matches the shell running pkill
SSHO=(-o BatchMode=yes -o ConnectTimeout=10)

ranks_ship() {  # BIN: the binary to every box (under a temporary name, then renamed), the fleet file beside it
  local w
  mkdir -p "$LRD/logs"
  cp "$1" "$LRD/$NAME.new" && mv -f "$LRD/$NAME.new" "$LRD/$NAME" && cp "$FLEET" "$LRD/fleet.conf" || return 3
  for w in "${FLEET_WORKERS[@]}"; do
    ssh "${SSHO[@]}" "$w" "mkdir -p $RD/logs; pkill -f '$PAT'"
    scp -q "$1" "$w:$(fleet_rel "$RD")/$NAME.new" &&
      ssh "${SSHO[@]}" "$w" "cat > $RD/fleet.conf && mv -f $RD/$NAME.new $RD/$NAME" < "$FLEET" || return 3
  done
}
ranks_workers() {  # args...: ranks 1-3, detached (a timeout of TIMEOUT s, default 900)
  local w r=1
  for w in "${FLEET_WORKERS[@]}"; do
    ssh "${SSHO[@]}" "$w" "echo 1000 > /proc/self/oom_score_adj; setsid nohup timeout ${TIMEOUT:-900} $RD/$NAME --rank $r $* > $RD/logs/$NAME.$r.log 2>&1 < /dev/null &"
    r=$((r + 1))
  done
}
ranks_stop() {  # rank 0 first (a server stops the others on SIGTERM), then whatever is left on every box
  local w i
  pkill -TERM -f "$PAT"
  for ((i = 0; i < 120; i++)); do pgrep -f "$PAT" > /dev/null || break; sleep 0.5; done
  pkill -KILL -f "$PAT"
  for w in "${FLEET_WORKERS[@]}"; do
    ssh "${SSHO[@]}" "$w" "for i in \$(seq 20); do pgrep -f '$PAT' > /dev/null || exit 0; sleep 0.5; done; pkill -KILL -f '$PAT'" &
  done
  wait
}
ranks_tails() {
  local w r=1
  for w in "${FLEET_WORKERS[@]}"; do
    echo "--- rank $r"
    ssh "${SSHO[@]}" "$w" "tail -n ${1:-3} $RD/logs/$NAME.$r.log"
    r=$((r + 1))
  done
}
