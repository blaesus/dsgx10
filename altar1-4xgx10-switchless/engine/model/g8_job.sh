#!/bin/bash
# A8's ring jobs (docs/engine.md § 9, "End to end on real weights"; glm53's G8 the template): run.sh's steps in the
# data tree altar1-g8 (binaries from install.sh with DATA_NAME=altar1-g8, the commit in bin/COMMIT), the fleet's four
# installed rank files, every routed expert from the file. One job of at most 10 minutes through the DeepSeek
# engine's ring.sh, from the repo root:
#   OWNER=altar1-g8 ds41f-4xgx10-switchless/engine/server/ring.sh altar1-4xgx10-switchless/engine/model/g8_job.sh JOB
# JOB: w (the loads with every CRC on every rank's GPU, glw_kcheck's host binding on rank 0),
# a (tf via prefill on short and python, layers 0-77 dumped; refhash graphs, graphs, eager),
# b (tf via decode with one draft pass on short and python, layers 0, 2, 3, 4, 6, 76, 77, 78 dumped: the MTP's argmax),
# c (tf via prefill on prose and agentic's first TFMAX tokens), d (refhash again: a second process, for the pin),
# m (mtpgold: the MTP layer teacher-forced on the golden final hidden, all four sequences),
# k (check and check6 on the real files), p (bench at 1, 2, 4, 6, 16 rows: the step and its profile),
# f (pfab: prefill at 2,600 / 8,192 / 8,192 @ 24,576 tokens in chunks of 8,192)
# GOLD: run.sh's default, ~/dsgb10-data/altar1-model/gold/all on every box: gold_prep.py's raw copies of the all-f32
# golden set under the name the scripts expect (docs/engine.md § 9).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
export DATA_NAME=altar1-g8
D=$HOME/dsgb10-data/altar1-g8
job=${1:?job w, a, b, c, d, m, k, p or f}
export OUT=$D/results-$job.$(date +%H%M%S).tsv
ALL=$(seq -s, 0 77)
case $job in
  w) "$HERE/run.sh" weights ;;
  a) G8CTX=6400 G8TAG=prefill DUMP=$ALL G8MODES="tf refhash refhash eager refhash" "$HERE/run.sh" g8 ;;
  b) G8CTX=6400 G8TAG=decode VIA=decode DUMP=0,2,3,4,6,76,77,78 EXTRA="--drafts 1" G8MODES="tf" "$HERE/run.sh" g8 ;;
  c) G8TAG=prefill SEQS=prose,agentic TFMAX=${TFMAX:-8000} DUMP=77 G8MODES="tf" "$HERE/run.sh" g8 ;;
  d) G8CTX=6400 G8TAG=prefill DUMP=77 G8MODES="refhash" "$HERE/run.sh" g8 ;;
  m) WEIGHTS=fleet EXTRA="--chunk 2048 --index-scratch-mib 128" "$HERE/run.sh" mtpgold ;;
  k) GW="--weights fleet" "$HERE/run.sh" check check6 ;;
  p) GW="--weights fleet" EXTRA="--rows 1,2,4,6,16" "$HERE/run.sh" bench ;;
  f) GW="--weights fleet" "$HERE/run.sh" pfab ;;
  *) echo "unknown job $job"; exit 2 ;;
esac
