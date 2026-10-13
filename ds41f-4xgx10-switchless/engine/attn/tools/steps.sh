# Resumable steps inside a ring part (ringparts.sh), to source:
#   STEPS_DIR=$J . engine/attn/tools/steps.sh
#   step KEY command...   the command, unless KEY finished in an earlier run (STEPS_DIR/steps_done.txt); a
#                         step killed by a signal (exit 137 or 143: its part was preempted) runs again next time
#   say message...        a timestamped line in STEPS_DIR/steps.log
# step returns the command's exit, so `step k cmd || ...` sees a failure; a failed step is still finished.
STEPS_DIR=${STEPS_DIR:-.}
mkdir -p "$STEPS_DIR"
say() { echo "$(date '+%F %T') $*" >> "$STEPS_DIR/steps.log"; }
step() {
  local key=$1 rc
  shift
  if grep -qx "$key" "$STEPS_DIR/steps_done.txt" 2> /dev/null; then
    say "$key: finished before, skipped"
    return 0
  fi
  say "$key: $*"
  "$@"
  rc=$?
  say "$key: exit $rc"
  ((rc == 143 || rc == 137)) || echo "$key" >> "$STEPS_DIR/steps_done.txt"
  return $rc
}
