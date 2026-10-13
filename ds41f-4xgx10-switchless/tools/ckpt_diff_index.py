"""Compare two safetensors index files: shard count, where tensors live, and which local shards hold
the Engram tensors.

  python3 tools/ckpt_diff_index.py <local model.safetensors.index.json> <original index.json>
"""
import collections
import json
import sys

loc = json.load(open(sys.argv[1]))["weight_map"]
org = json.load(open(sys.argv[2]))["weight_map"]
print(f"tensors: local {len(loc)}, original {len(org)}; only local {len(set(loc) - set(org))}, "
      f"only original {len(set(org) - set(loc))}")
print("shards: local", len(set(loc.values())), "original", len(set(org.values())))
same = sum(1 for n in loc if org.get(n) == loc[n])
print(f"tensors in the same-named shard: {same}/{len(loc)}")
for n in sorted(loc):
    if ".engram." in n:
        print(f"  {n:45s} local {loc[n]}  original {org.get(n)}")
by_shard = collections.defaultdict(set)
for n, f in loc.items():
    layer = n.split(".")[1] if n.startswith(("layers.", "mtp.")) else n.split(".")[0]
    by_shard[f].add(n.split(".")[0] + "." + layer if n.startswith(("layers.", "mtp.")) else layer)
for f in sorted(by_shard):
    print(f"  {f}: {sorted(by_shard[f], key=lambda s: (len(s), s))[:8]}{' ...' if len(by_shard[f]) > 8 else ''}")
