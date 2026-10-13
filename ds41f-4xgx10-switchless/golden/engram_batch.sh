#!/bin/bash
# The long-transcript collapse (golden/README.md, "Collapse"): every run behind that section, on the
# first 5,120 tokens of long-cut (onset ~4,527) or on 2,048-token plain documents. Each run holds ~10 GiB
# and takes ~5-8 min alone; run them one or two at a time, since other lines load ~72 GiB engine
# ranks on the workers, and drop the page cache between runs (tools/evict_cache.py).
#   golden/engram_batch.sh probe | ablate | orig | dense | exact | docs | conv
cd ~/dsgb10/golden-harness || exit 1
G=~/dsgb10-data/golden
T=golden/long-cut5120.tokens.json
MIN="--positions sparse:100000:1 --record none --full-logits 1"
ORIG=$HOME/DeepSeek-V4.1-Flash  # deepseek-ai's weights; the default CKPT is the uncensored set
R=golden/run_ref.sh
case "${1:-}" in
  probe) $R py /work/golden/engram_probe.py --tokens /work/$T --out /out/probe5120.npz > $G/probe5120.log 2>&1 ;;
  ablate)
    $R run off-onset $T $MIN --engram-off 4480-4560 > $G/off-onset.log 2>&1 & sleep 45
    $R run off-before $T $MIN --engram-off 0-4480 > $G/off-before.log 2>&1 ;;
  orig)
    CKPT=$ORIG $R run orig-base $T $MIN > $G/orig-base.log 2>&1 & sleep 45
    CKPT=$ORIG $R run orig-noengram $T $MIN --ablate engram > $G/orig-noengram.log 2>&1 ;;
  dense)
    $R run dense $T $MIN --set index_topk=8192 > $G/dense.log 2>&1 & sleep 45
    $R run off-l01 $T $MIN --engram-off 0-5120 --engram-off-layers 1 > $G/off-l01.log 2>&1 & sleep 45
    $R run off-l14 $T $MIN --engram-off 0-5120 --engram-off-layers 14 > $G/off-l14.log 2>&1 ;;
  exact) TORCHREF_ACT=exact $R run exact $T $MIN > $G/exact.log 2>&1 ;;
  docs)  # BOS + textwrap.py and BOS + docs/findings.md (never published), layer-1 Engram on and off
    for d in textwrap novel; do
      $R run doc-$d golden/doc-$d.tokens.json $MIN > $G/doc-$d.log 2>&1 & sleep 30
      $R run doc-$d-offl1 golden/doc-$d.tokens.json $MIN --engram-off 0-2048 --engram-off-layers 1 \
        > $G/doc-$d-offl1.log 2>&1
      wait
    done ;;
  conv) $R run off-l01-conv $T $MIN --engram-off 0-1163 --engram-off-layers 1 > $G/off-l01-conv.log 2>&1 ;;
  *) echo "usage: $0 probe | ablate | orig | dense | exact | docs | conv" >&2; exit 2 ;;
esac
wait
echo "$1 done" >> $G/engram-batch-done.txt
