#!/bin/bash
# SHA-256 of checkpoint shards, read with O_DIRECT (no page-cache churn on a box serving), at nice 19, checked
# against the Hub manifest (lines "name<TAB>sha256<TAB>bytes").
#   tools/shard_sha.sh MANIFEST DIR SHARD... > out.tsv     (SHARD: 00001 or model-00001.safetensors)
set -u
man=$1 dir=$2
shift 2
rc=0
for s in "$@"; do
  f=$s
  [[ $f == model-* ]] || f=model-$s.safetensors
  want=$(awk -F'\t' -v f="$f" '$1 == f {print $2 "\t" $3}' "$man")
  [ -n "$want" ] || { echo -e "$f\tnot in the manifest"; rc=1; continue; }
  size=$(stat -c %s "$dir/$f")
  got=$(nice -n 19 dd if="$dir/$f" bs=16M iflag=direct status=none | nice -n 19 sha256sum | cut -d' ' -f1)
  if [ "$got	$size" = "$want" ]; then v=OK; else v=MISMATCH; rc=1; fi
  echo -e "$f\t$got\t$size\t$v"
done
exit $rc
