#!/bin/bash
# One engine run on the four boxes, checked for a tokens hash:
#   run_hash.sh LOG HASH run4.sh-arguments...
# Rank 0's output goes to LOG; exit 0 if the run exited 0 and LOG has HASH, else 1 (the run's own exit if it
# was killed: 137 or 143 tell steps.sh to run it again). Env: BIN, OWNER, TIMEOUT as run4.sh takes them (OWNER
# attention, TIMEOUT 900); SETENV="NAME=value with spaces": NAME set on every rank (run4.sh hands the workers
# its arguments as one string that their shell splits, so the spaces are escaped here).
#   run_hash.sh $J/tf.log 842278b28d87dad0 --mode tf --big --golden /data/dsgb10/golden/long --prompt 4096 \
#     --limit 4608 --prefill-rows 4096
set -u
[ $# -ge 3 ] || {
  sed -n '2,9p' "$0" >&2
  exit 2
}
LOG=$(realpath -m "$1") HASH=$2
shift 2
SE=()
[ -n "${SETENV:-}" ] && SE=(--setenv "${SETENV// /\\ }")
cd "$(dirname "$0")/../../.." || exit 2
OWNER=${OWNER:-attention} TIMEOUT=${TIMEOUT:-900} engine/server/run4.sh "${SE[@]}" "$@" > "$LOG" 2>&1 < /dev/null
rc=$?
((rc == 143 || rc == 137)) && exit $rc
[ $rc = 0 ] && grep -q "$HASH" "$LOG" && exit 0
exit 1
