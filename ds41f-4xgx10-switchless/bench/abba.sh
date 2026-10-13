#!/bin/bash
# A/B/B/A of two engine binaries on the 1k golden run (golden/README.md: --mode decode --golden G --ctx 1024
# --steps 256 --big), at k = 1 and with drafts (--spec --draft-fp8): each run through engine/server/run4.sh
# under engine/server/ring.sh (OWNER set; preemptible, a run preempted (exit 76) goes again, never
# PREEMPT=0). Takes each run's "ms per token" figure (the line "then X ms per token", over the run's last
# tokens) and its hash, prints every run's figure, then the medians per binary and variant and whether
# the hashes agree (they must: both binaries give febc7dc54ee40a39 on dealignai's variant).
#
#   OWNER=me bench/abba.sh A_BIN B_BIN              # A B B A per variant, k1 then fp8
#   ROUNDS=2 VARIANTS=k1 bench/abba.sh A_BIN B_BIN   # ROUNDS ABBA rounds; VARIANTS: k1, fp8 (default both)
#   bench/abba.sh A_BIN B_BIN -- --weights P        # arguments after -- go to every run
# Environment: G (the golden directory, default /data/dsgb10/golden/long), TIMEOUT per run in s (240),
# OUT (the logs, default ~/dsgb10/abba/<date>-<time>, made 0700).
# Exit 0 when every run printed its figure and the hashes agree.
set -u
cd "$(dirname "$0")/.." || exit 2
case ${1:-} in -h | --help | '') sed -n '2,/^# Exit 0/p' "$0"; exit 2 ;; esac
[ $# -ge 2 ] || { sed -n '2,/^# Exit 0/p' "$0"; exit 2; }
A_BIN=$1 B_BIN=$2
shift 2
[ "${1:-}" = -- ] && shift
EXTRA=("$@")
for b in "$A_BIN" "$B_BIN"; do
  [ -x "$b" ] || { echo "abba.sh: no engine binary $b" >&2; exit 2; }
done
G=${G:-/data/dsgb10/golden/long}
OUT=${OUT:-$HOME/dsgb10/abba/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT" && chmod 700 "$OUT" || exit 2
export OWNER=${OWNER:-abba}
ROUNDS=${ROUNDS:-1}
VARIANTS=${VARIANTS:-k1 fp8}
log() { echo "$(date '+%F %T') abba: $*"; }
args_of() {  # variant: the run's arguments into ARGS
  case $1 in
    k1) ARGS=(--mode decode --golden "$G" --ctx 1024 --steps 256 --big) ;;
    fp8) ARGS=(--mode decode --golden "$G" --ctx 1024 --steps 256 --big --spec --draft-fp8) ;;
    *) echo "abba.sh: no variant '$1' (k1, fp8)" >&2; exit 2 ;;
  esac
}
: > "$OUT/results"
log "A $A_BIN (sha256 $(sha256sum "$A_BIN" | cut -c1-16)); B $B_BIN (sha256 $(sha256sum "$B_BIN" | cut -c1-16)); variants $VARIANTS, $ROUNDS round(s); ${EXTRA[*]:+engine arguments ${EXTRA[*]}; }logs in $OUT"
n=0
for v in $VARIANTS; do
  args_of "$v"
  for ((r = 0; r < ROUNDS; r++)); do
    for side in A B B A; do
      n=$((n + 1))
      bin=$A_BIN
      [ $side = B ] && bin=$B_BIN
      f=$OUT/$(printf '%02d' $n).$v.$side.log
      rc=1
      for attempt in 1 2 3; do
        t0=$SECONDS
        BIN=$bin TIMEOUT=${TIMEOUT:-240} engine/server/ring.sh engine/server/run4.sh "${ARGS[@]}" "${EXTRA[@]}" > "$f" 2>&1 < /dev/null
        rc=$?
        [ $rc = 76 ] || break
        log "$n $v $side: preempted after $((SECONDS - t0)) s (try $attempt); again in 30 s"
        sleep 30
      done
      ms=$(grep -o 'then [0-9.]* ms per token' "$f" | tail -n 1 | cut -d' ' -f2)
      h=$(grep -o 'hash [0-9a-f]\{16\}' "$f" | tail -n 1 | cut -d' ' -f2)
      echo "$n $v $side $rc ${ms:--} ${h:--}" >> "$OUT/results"
      log "$n $v $side: exit $rc after $((SECONDS - t0)) s, ${ms:-no} ms per token, hash ${h:-none}"
      sleep 5  # a job waiting for the ring may go between ours
    done
  done
done
# the table: each run, then the medians per binary and variant, and the hashes
printf '\n%-3s %-4s %-3s %-10s %s\n' "" var bin ms/token hash
while read -r i v side rc ms h; do
  printf '%-3s %-4s %-3s %-10s %s%s\n' "$i" "$v" "$side" "$ms" "$h" "$([ "$rc" = 0 ] || echo " (exit $rc)")"
done < "$OUT/results"
python3 - "$OUT/results" <<'PY'
import statistics, sys
rows = [l.split() for l in open(sys.argv[1]) if l.strip()]
bad = 0
for v in dict.fromkeys(r[1] for r in rows):
    med = {}
    for side in 'AB':
        xs = [float(r[4]) for r in rows if r[1] == v and r[2] == side and r[4] != '-']
        if xs:
            med[side] = statistics.median(xs)
            print('%s %s: median %.3f ms per token over %d run(s) (min %.3f, max %.3f)' % (v, side, med[side], len(xs), min(xs), max(xs)))
        else:
            print('%s %s: no run printed its figure' % (v, side))
            bad += 1
    if len(med) == 2:
        print('%s: B over A %+.2f %%' % (v, 100.0 * (med['B'] / med['A'] - 1)))
    hs = {r[5] for r in rows if r[1] == v and r[5] != '-'}
    if len(hs) == 1:
        print('%s: every run hashed %s' % (v, hs.pop()))
    else:
        print('%s: the hashes %s' % (v, 'DIFFER: ' + ' '.join(sorted(hs)) if hs else 'are missing'))
        bad += 1
sys.exit(1 if bad else 0)
PY
