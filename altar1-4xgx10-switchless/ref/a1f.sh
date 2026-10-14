#!/bin/bash
# A1f's analyses on rank 2 (docs/reference.md § 10), from the ref directory, logs to ~/altar1-ref/logs/a1f-*.txt:
#
#   ./a1f.sh check    once passes-abs.log says "pass short-f32-abs exit 0": batch invariance of the absorbed form
#                     (short alone against short in a1a-f32-abs, bit for bit), the form's effect on the a1a pair,
#                     the engine's A9 layer-0 dumps against a1a-f32-abs; numpy, one thread, beside the running pass
#   ./a1f.sh finish   once passes-abs.log says "pass all-f32-abs exit 0": summary, SHA256SUMS, batch invariance
#                     inside all-f32-abs, the form's effect and the new order-variance pair on the four sequences,
#                     the MTP acceptance, gold_prep's raw copies, and A8's comparisons (engine.md § 10.3) against
#                     all-f32-abs; run alone (8 threads)
#
# Every step at nice 19, idle I/O, cores 10-19, oom_score_adj 1000; nothing here writes into the earlier golden runs.
set -u
cd ~/altar1-ref/code/altar1-4xgx10-switchless/ref || exit 2
echo 1000 > /proc/self/oom_score_adj
G=~/altar1-ref/golden
K=~/altar1-ref/checks
L=~/altar1-ref/logs
E=~/dsgb10-data/altar1-g8
PY=~/altar1-ref/venv/bin/python
T=${T:-8}
run() {  # run NAME CMD...: CMD's output to $L/a1f-NAME.txt, its time and peak RSS to $L/a1f-NAME.time
  local name=$1; shift
  echo "$(date -Is) $name"
  OMP_NUM_THREADS=$T taskset -c 10-19 nice -n 19 ionice -c3 /usr/bin/time -v "$@" > "$L/a1f-$name.txt" 2> "$L/a1f-$name.time"
  echo "$(date -Is) $name exit $?"
}
case ${1:-} in
  check)
    T=1
    run bits-short-alone $PY -I cmp_bits.py $K/short-f32-abs $G/a1a-f32-abs --seq short
    for s in short prose2400; do
      run cmp-a1a-$s $PY -I cmp_runs.py $G/a1a-f32-abs $G/a1a-f32 --seq $s
    done
    run summarize-a1a-f32-abs $PY -I summarize.py $G/a1a-f32-abs --tokenizer ~/altar-1/tokenizer.json --json $G/a1a-f32-abs/summary.json
    # A8's prefill comparison of short (the fixed engine) against the a1a pair in both attention forms
    run g8-a1a-short $PY -I ~/altar1-ref/code/compare_golden.py g8 $G $E/g8/prefill-fix --runs a1a-f32-abs,a1a-f32 --seqs short --mtp-runs a1a-f32-abs,a1a-f32 --json $L/a1f-g8-a1a-short.json
    run sha-a1a-f32-abs bash -c "cd $G/a1a-f32-abs && find . -path ./state -prune -o -type f ! -name SHA256SUMS -print | sort | xargs sha256sum > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS && sha256sum SHA256SUMS"
    ;;
  finish)
    run summarize-all-f32-abs $PY -I summarize.py $G/all-f32-abs --tokenizer ~/altar-1/tokenizer.json --json $G/all-f32-abs/summary.json
    run sha-all-f32-abs bash -c "cd $G/all-f32-abs && find . -path ./state -prune -o -type f ! -name SHA256SUMS -print | sort | xargs sha256sum > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS && sha256sum SHA256SUMS final/argmax.npy mtp/argmax.npy summary.json"
    run bits-all-short $PY -I cmp_bits.py $G/all-f32-abs $G/a1a-f32-abs --seq short
    run bits-all-prose $PY -I cmp_bits.py $G/all-f32-abs $G/a1a-f32-abs --seq prose --seq-b prose2400
    for s in short prose python agentic; do
      run cmp-form-$s $PY -I cmp_runs.py $G/all-f32-abs $G/all-f32 --seq $s        # the attention form's effect
      run cmp-orders-$s $PY -I cmp_runs.py $G/all $G/all-f32-abs --seq $s          # the new order-variance pair
      run cmp-orders-rev-$s $PY -I cmp_runs.py $G/all-f32-abs $G/all --seq $s
    done
    run mtp $PY -I mtp_accept.py $G all-f32-abs all-f32-abs,all-f32,all
    run gold-prep $PY -I ~/altar1-ref/code/gold_prep.py $G/all-f32-abs ~/altar1-ref/gold/all-f32-abs --l78
    run gold-sha bash -c "cd ~/altar1-ref/gold && sha256sum all-f32-abs.*"
    # A8's comparisons (engine.md § 10.3) against the new set, the engine's dumps of the fixed binaries
    T=4
    run g8-prefill-fix $PY -I ~/altar1-ref/code/compare_golden.py g8 $G $E/g8/prefill-fix --runs all-f32-abs,all --mtp-runs all-f32-abs,all-f32 --json $L/a1f-g8-prefill-fix-abs.json
    run g8-decode-fix $PY -I ~/altar1-ref/code/compare_golden.py g8 $G $E/g8/decode-fix --runs all-f32-abs,all --seqs short,python --mtp-runs all-f32-abs,all-f32 --json $L/a1f-g8-decode-fix-abs.json
    ;;
  *) sed -n 2,12p "$0"; exit 2 ;;
esac
echo "$(date -Is) a1f $1 done"
