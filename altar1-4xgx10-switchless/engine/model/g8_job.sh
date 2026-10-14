#!/bin/bash
# G8's ring jobs (docs/engine.md, "End to end on real weights"): run.sh's g8 step in the data tree glm53-g8
# (binaries from install.sh with DATA_NAME=glm53-g8), the fleet's rank files with every routed expert filled. One
# job of at most 10 minutes through the DeepSeek engine's ring.sh, from the repo root:
#   OWNER=glm53-g8 ds41f-4xgx10-switchless/engine/server/ring.sh glm53-4xgx10-switchless/engine/model/g8_job.sh JOB
# JOB: a (tf via prefill on short and python, all layers dumped; refhash graphs, graphs, eager), b (tf via decode with
# one draft on short and python: the MTP's argmax), c (tf via prefill on prose and agentic's first TFMAX tokens),
# d (refhash again: a second process, for the pin)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
export DATA_NAME=glm53-g8
D=$HOME/dsgb10-data/glm53-g8
job=${1:?job a, b, c or d}
export OUT=$D/results-$job.$(date +%H%M%S).tsv
ALL=$(seq -s, 0 77)
case $job in
  a) G8CTX=6400 G8TAG=prefill DUMP=$ALL G8MODES="tf refhash refhash eager refhash" "$HERE/run.sh" g8 ;;
  b) G8CTX=6400 G8TAG=decode VIA=decode DUMP=77 EXTRA="--drafts 1" G8MODES="tf" "$HERE/run.sh" g8 ;;
  c) G8TAG=prefill SEQS=prose,agentic TFMAX=${TFMAX:-8000} DUMP=77 G8MODES="tf" "$HERE/run.sh" g8 ;;
  d) G8CTX=6400 G8TAG=prefill DUMP=77 G8MODES="refhash" "$HERE/run.sh" g8 ;;
  *) echo "unknown job $job"; exit 2 ;;
esac
