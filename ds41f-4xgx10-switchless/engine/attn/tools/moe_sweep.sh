#!/bin/bash
# The prefill MoE and mHC boundary knobs (DSGB10_MOE_TUNE) on rank 0's GPU, ~7 min: the bit tests (test_kernels
# p_moe, test_pattn mhc), bench_kernels moereal on real routes (the down kernel's column groups at rings 3 and
# 9; up's prefetch and the tile order), bench_pattn's boundary table, then pick.py's tuning:
#   moe_sweep.sh OUTDIR KBUILD          KBUILD: a build's kernels directory (engine/build/kernels)
# Out, in OUTDIR: tk.log, tp.log, bk_d3.log, bk_d9.log, bk_ut.log, bp_mhcr.log, tune.txt (the DSGB10_MOE_TUNE
# line), pick.log (why). A GPU job: run it under ring.sh, e.g. as a part of ringparts.sh; a preempted run
# skips the steps it finished (steps.sh). Then check the tune's bits on the ring (run_hash.sh with
# SETENV=DSGB10_MOE_TUNE=...) and its turns against the defaults (serve_turns.sh, turns_cmp.py).
# Env: ROUTES (dumped routes, /data/dsgb10/kernels/routes/long).
set -u
[ $# = 2 ] || {
  sed -n '2,10p' "$0" >&2
  exit 2
}
OUT=$(realpath -m "$1")
X=$(realpath "$2")
T=$(cd "$(dirname "$0")" && pwd)
STEPS_DIR=$OUT
. "$T/steps.sh"
ROUTES=${ROUTES:-/data/dsgb10/kernels/routes/long}
bk() {  # bk NAME VAR=value...: bench_kernels moereal on the routes, at the turn sizes
  local name=$1
  shift
  env PF_ROUTES="$ROUTES" PF_M="512 2000 8000 8192" PF_ROUNDS=3 "$@" "$X/bench_kernels" moereal > "$OUT/bk_$name.log" 2>&1
}
step tk sh -c "'$X/test_kernels' '' p_moe > '$OUT/tk.log' 2>&1"
step tp sh -c "'$X/test_pattn' '' mhc > '$OUT/tp.log' 2>&1"
step bk_d3 bk d3 PF_DGRP="0 1 2 4 5 10 20 40" PF_DRING=3
step bk_d9 bk d9 PF_DGRP="0 2 4 5 10" PF_DRING=9
step bk_ut bk ut PF_DGRP="0" PF_UPF="0 1 2 3" PF_TORD="0 1 2 3 5"
step bp sh -c "'$X/bench_pattn' --only mhcr --iters 10 > '$OUT/bp_mhcr.log' 2>&1"
python3 "$T/pick.py" "$OUT" > "$OUT/tune.txt" 2> "$OUT/pick.log"  # pick.log also has --later's candidate
say "bit tests: $(grep -c FAIL "$OUT/tk.log" "$OUT/tp.log" | tr '\n' ' ')FAIL lines; sweeps with differing bits:" \
  "$(grep -c DIFFER "$OUT"/bk_*.log | tr '\n' ' ')"
say "pick: $(cat "$OUT/tune.txt")"
