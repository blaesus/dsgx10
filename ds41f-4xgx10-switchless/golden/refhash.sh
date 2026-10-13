#!/bin/bash
# The reference hashes (golden/README.md, "Checking a build"): each run below through engine/server/run4.sh,
# a few runs to a ring job (engine/server/ring.sh, which borrows the ring from a serving engine the way
# run4.sh does), then a table: each run's hash, the hash it gives on dealignai's variant (the weights every
# documented hash was made with), and its time. Run it from the configuration directory:
#
#   golden/refhash.sh                         # the eight runs: tf k1 nol2pf fp8 bf16 temp ced16 ced28
#   golden/refhash.sh ced28 ced28 exact28     # any of them, repeats too
#   golden/refhash.sh k1 fp8 -- --weights P   # arguments after -- go to every run
#
#   tf       --mode tf --big --golden G --prompt 4096 --limit 4608 --prefill-rows 4096
#   k1       --mode decode --golden G --ctx 1024 --steps 256 --big
#   nol2pf   the same and --no-l2pf
#   fp8      the same and --spec --draft-fp8
#   bf16     the same and --spec --draft-bf16
#   temp     --mode decode --golden G --ctx 1024 --steps 128 --big --temp 0.8 --top-p 0.95 --top-k 40 --seed 7
#   ced16    --mode decode --golden G --ctx 16384 --steps 256 --big --ced
#   ced28    --mode tf --big --golden C --prompt 28489 --limit 28489 --prefill-rows 8192 --ced
#   exact28  ced28 without CED (--prefill-rows 4096): not a reference, a control that must print another hash
#
# Environment:
#   G         the long transcript's golden directory (default /data/dsgb10/golden/long)
#   C         its CED check directory (default /data/dsgb10/kernels/cedcheck/long28k-ced128). These runs read
#             only the transcript's tokens (tf also opens G's top logits, whatever they hold), so the directory
#             `python3 golden/make_cedcheck.py --tokens-only --out DIR` makes from this repository alone,
#             DIR/long-tokens, does for both
#   BIN       the engine binary (run4.sh's default, engine/build/server/dsgb10_rank)
#   OUT       the logs (default ~/dsgb10/refhash/<date>-<time>, made 0700)
#   OWNER     who holds the ring (default refhash); PREEMPT=0: a serving engine's requests wait for the job's
#             end (by default they stop it, exit 76, and the runs it had not finished go again: 3 tries a job)
#   EXPECT    1 (default): the hashes of dealignai's variant, the weights every documented hash was made with;
#             official: deepseek-ai's checkpoint's (docs/engine.md, "The official checkpoint's weights"; pass its
#             files with -- --weights PATH; exact28 has no value there); 0: none (other weights). The table always
#             checks the hashes that must be equal (k1 = nol2pf = fp8 = bf16, every repeat of a run) and that
#             exact28 differs from ced28
#   JOB_RUNS  runs per ring job (default 4: about 4-5 minutes); TIMEOUT per run in s (default 240)
# Exit 0 when every run printed a hash and every check holds.
set -u
cd "$(dirname "$0")/.." || exit 2
SELF=$PWD/golden/refhash.sh
G=${G:-/data/dsgb10/golden/long}
C=${C:-/data/dsgb10/kernels/cedcheck/long28k-ced128}
NAMES=(tf k1 nol2pf fp8 bf16 temp ced16 ced28 exact28)

args_of() {  # name: the run's arguments into A
  case $1 in
    tf) A=(--mode tf --big --golden "$G" --prompt 4096 --limit 4608 --prefill-rows 4096) ;;
    k1) A=(--mode decode --golden "$G" --ctx 1024 --steps 256 --big) ;;
    nol2pf) A=(--mode decode --golden "$G" --ctx 1024 --steps 256 --big --no-l2pf) ;;
    fp8) A=(--mode decode --golden "$G" --ctx 1024 --steps 256 --big --spec --draft-fp8) ;;
    bf16) A=(--mode decode --golden "$G" --ctx 1024 --steps 256 --big --spec --draft-bf16) ;;
    temp) A=(--mode decode --golden "$G" --ctx 1024 --steps 128 --big --temp 0.8 --top-p 0.95 --top-k 40 --seed 7) ;;
    ced16) A=(--mode decode --golden "$G" --ctx 16384 --steps 256 --big --ced) ;;
    ced28) A=(--mode tf --big --golden "$C" --prompt 28489 --limit 28489 --prefill-rows 8192 --ced) ;;
    exact28) A=(--mode tf --big --golden "$C" --prompt 28489 --limit 28489 --prefill-rows 4096) ;;
    *) return 1 ;;
  esac
}
expected_of() {  # name: the hash on dealignai's variant (release candidate 5; ced28 and exact28 since b47c510), or
  # with EXPECT=official on deepseek-ai's checkpoint (2026-10-13, release candidate 5's code; ced28 b47c510's)
  if [ "${EXPECT:-1}" = official ]; then
    case $1 in
      tf) echo c7605e0b6ba8d9d2 ;;
      k1 | nol2pf | fp8 | bf16) echo 7941647b162df1a8 ;;
      temp) echo 02ef3e0858d7eb45 ;;
      ced16) echo 8790ee9388f4abd8 ;;
      ced28) echo de99bfce6ea96ba8 ;;
    esac
    return
  fi
  case $1 in
    tf) echo 842278b28d87dad0 ;;
    k1 | nol2pf | fp8 | bf16) echo febc7dc54ee40a39 ;;
    temp) echo a8f7a08f8cd1c5e8 ;;
    ced16) echo 81f772ab79a950e4 ;;
    ced28) echo 8f08a153d415c5c8 ;;
    exact28) echo 87f42fa9f94f13de ;;
  esac
}
SEED=14650fb0739d0383  # FNV's seed: what ced28 and exact28 printed before b47c510 (they hashed nothing)
log() { echo "$(date '+%F %T') refhash: $*"; }

# ---- a ring job: refhash.sh --job OUT NN:name... [-- engine args]; under ring.sh, each run through run4.sh
if [ "${1:-}" = --job ]; then
  OUT=$2
  shift 2
  RUNS=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do RUNS+=("$1") && shift; done
  [ $# -gt 0 ] && shift
  [ -n "${RING_HOLDER:-}" ] || { log "a job runs under engine/server/ring.sh"; exit 2; }
  for r in "${RUNS[@]}"; do
    i=${r%%:*} name=${r#*:}
    args_of "$name" || exit 2
    t0=$SECONDS
    TIMEOUT=${TIMEOUT:-240} engine/server/run4.sh "${A[@]}" "$@" > "$OUT/$i.$name.log" 2>&1 < /dev/null
    rc=$?
    h=$(grep -o 'hash [0-9a-f]\{16\}' "$OUT/$i.$name.log" | tail -n 1 | cut -d' ' -f2)
    echo "$i $name $rc $((SECONDS - t0)) ${h:--}" >> "$OUT/results"
    log "$i $name: exit $rc after $((SECONDS - t0)) s, hash ${h:-none}"
  done
  exit 0
fi

# ---- the driver
case ${1:-} in -h | --help) sed -n '2,/^# Exit 0/p' "$SELF"; exit 0 ;; esac
LIST=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do
  args_of "$1" || { echo "refhash.sh: no run '$1' (runs: ${NAMES[*]})" >&2; exit 2; }
  LIST+=("$1")
  shift
done
[ $# -gt 0 ] && shift
EXTRA=("$@")
[ ${#LIST[@]} -gt 0 ] || LIST=(tf k1 nol2pf fp8 bf16 temp ced16 ced28)
OUT=${OUT:-$HOME/dsgb10/refhash/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT" && chmod 700 "$OUT" || exit 2
: > "$OUT/results"
export OWNER=${OWNER:-refhash} OUT
B=${BIN:-engine/build/server/dsgb10_rank}
[ -f "$B" ] || { echo "refhash.sh: no engine binary $B (BIN)" >&2; exit 2; }
log "binary $B (sha256 $(sha256sum "$B" | cut -c1-16)); expected hashes: $(case ${EXPECT:-1} in 0) echo none ;; official) echo "deepseek-ai's checkpoint" ;; *) echo "dealignai's variant" ;; esac); G $G; C $C; ${EXTRA[*]:+engine arguments ${EXTRA[*]}; }logs in $OUT"
TAGS=()
for ((k = 0; k < ${#LIST[@]}; k++)); do TAGS+=("$(printf '%02d' $((k + 1))):${LIST[k]}"); done
. engine/server/ringlock.sh
ring_want_hold "refhash: ${LIST[*]}" || log "no want held between the jobs"
trap ring_want_drop EXIT
per=${JOB_RUNS:-4}
for ((j = 0; j < ${#TAGS[@]}; j += per)); do
  [ $j = 0 ] || sleep 5  # a job waiting for the ring may go between ours
  JOB=("${TAGS[@]:j:per}")
  for attempt in 1 2 3; do
    todo=()
    for t in "${JOB[@]}"; do grep -q "^${t%%:*} " "$OUT/results" || todo+=("$t"); done
    [ ${#todo[@]} -gt 0 ] || break
    t0=$SECONDS
    engine/server/ring.sh bash "$SELF" --job "$OUT" "${todo[@]}" -- "${EXTRA[@]}" >> "$OUT/jobs.log" 2>&1
    rc=$?
    log "job ${todo[*]}: ring.sh exit $rc after $((SECONDS - t0)) s (the wait for the ring included)"
    [ $rc = 76 ] || break
    sleep 30  # preempted: the runs it did not finish go again
  done
done
ring_want_drop

# ---- the table
bad=0
declare -A FIRST
printf '\n%-3s %-8s %-17s %-17s %-9s %s\n' "" run hash expected check seconds
for t in "${TAGS[@]}"; do
  i=${t%%:*} name=${t#*:} rc= s= h=
  read -r _ _ rc s h < <(grep "^$i " "$OUT/results")
  want=$(expected_of "$name")
  [ "${EXPECT:-1}" = 0 ] && want=
  if [ -z "${rc:-}" ]; then check="not run" s=- h=-
  elif [ "$h" = - ]; then check="exit $rc"
  elif [ "$h" = "$SEED" ] && [ "${name%28}" != "$name" ]; then check="the seed"
  elif [ -z "$want" ]; then check=-
  elif [ "$h" = "$want" ]; then check=ok
  else check=DIFFERS; fi
  case $check in ok | -) ;; *) bad=$((bad + 1)) ;; esac
  printf '%-3s %-8s %-17s %-17s %-9s %s\n' "$i" "$name" "$h" "${want:--}" "$check" "$s"
  [ "$h" = - ] && continue
  group=$name
  case $name in k1 | nol2pf | fp8 | bf16) group=k1 ;; esac
  if [ -z "${FIRST[$group]:-}" ]; then FIRST[$group]=$h
  elif [ "${FIRST[$group]}" != "$h" ]; then log "$name: $h, not ${FIRST[$group]} as before: they must be equal"; bad=$((bad + 1)); fi
done
if [ -n "${FIRST[ced28]:-}" ] && [ "${FIRST[ced28]}" = "${FIRST[exact28]:-}" ]; then
  log "ced28 and exact28 printed the same hash: it does not depend on the computation"
  bad=$((bad + 1))
fi
[ $bad = 0 ] && log "every run printed its hash; every check holds" || log "$bad check(s) failed (logs in $OUT)"
[ $bad = 0 ]
