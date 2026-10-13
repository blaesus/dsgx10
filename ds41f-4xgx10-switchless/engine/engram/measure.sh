#!/bin/bash
# Engram measurements on one box (docs/engram.md):
#   measure.sh <rank> <shard-dir> <checkpoint-dir> [read-batch=1]
# Binaries in ~/dsgb10/bin/engram, hash map and token streams in ~/dsgb10/engram. Run detached:
#   setsid bash measure.sh 3 ~/dsv41-4x-ring/engram ~/DeepSeek-V4.1-Flash-UNCENSORED-FP8 8 \
#     > ~/dsgb10/logs/engram-measure.log 2>&1 < /dev/null &
r=$1
S=$2
CK=$3
b=${4:-1}
B=~/dsgb10/bin/engram
D=~/dsgb10/engram
echo "== $(hostname) rank $r read batch $b $(date -Is)"
cat /sys/class/nvme/nvme0/model
for f in apst_primary_timeout_ms apst_secondary_timeout_ms default_ps_max_latency_us; do
  echo "$f=$(cat /sys/module/nvme_core/parameters/$f)"
done
d=$(readlink -f /sys/class/nvme/nvme0/device)
echo "l1_aspm=$(cat $d/link/l1_aspm 2>&1) link=$(cat $d/current_link_speed) x$(cat $d/current_link_width)"
echo "== iobench"
taskset -c 7 $B/engram_iobench $S/engram-l1-r${r}of4.bin 3
echo "== iobench gap"
taskset -c 7 $B/engram_iobench $S/engram-l1-r${r}of4.bin gap 0,1000,5000,9000,11000,15000,50000 200
taskset -c 7 $B/engram_iobench $S/engram-l1-r${r}of4.bin gap 120000,2500000 20
taskset -c 7 $B/engram_iobench $S/engram-l1-r${r}of4.bin gap 1000 300 8
echo "== rowcheck"
$B/engram_rowcheck $S/engram-l1-r${r}of4.bin $CK 3000
$B/engram_rowcheck $S/engram-l14-r${r}of4.bin $CK 3000
echo "== storecheck"
$B/engram_storecheck $D/engram_hashmap.bin rank $r $S/engram-l1-r${r}of4.bin $S/engram-l14-r${r}of4.bin 200
for s in code prose; do
  echo "== e2e $s"
  $B/engram_e2e $D/engram_hashmap.bin $r $S/engram-l1-r${r}of4.bin $S/engram-l14-r${r}of4.bin $D/streams/$s.i32 \
    4096 1000000 7 5 $b
done
echo "== done $(date -Is)"
