"""Summarise a golden run's indexer records: how many compressed positions each index layer picked
at each record position, how many blocks layer 20's candidate pool kept, and whether the decoder's
reindex layers stayed inside that pool.

  golden/run_ref.sh py /work/golden/check_records.py /out/long
"""
import glob
import json
import os
import sys

import numpy as np

run = sys.argv[1]
man = json.load(open(f"{run}/manifest.json"))
cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "ref", "inference", "config.json")))
R = np.load(f"{run}/record_positions.npy")
topk, block, n_blocks = cfg["index_topk"], cfg["candidate_block_size"], cfg["candidate_topk_blocks"]
src = cfg["candidate_source_layer"]
print(f"L={man['L']} record positions {R.tolist()}")

pool = None
if os.path.exists(f"{run}/L{src:02d}.pool_blocks.npy"):
    pool = np.load(f"{run}/L{src:02d}.pool_blocks.npy").astype(bool)  # [R, n_blocks_total]
    kept = pool.sum(1)
    reach = [-(-(p + 1) // block) for p in R]  # ratio 1 at the candidate source
    want = [min(n_blocks, r) for r in reach]
    print(f"L{src:02d} pool: kept blocks {kept.tolist()}")
    print(f"        reachable blocks {reach}; expected kept {want}: {'OK' if kept.tolist() == want else 'MISMATCH'}")

for f in sorted(glob.glob(f"{run}/L*.index_topk.npy")):
    layer = int(os.path.basename(f)[1:3])
    ratio = cfg["compress_ratios"][layer]
    t = np.load(f)
    valid = (t >= 0).sum(1)
    want = [min(topk, (p + 1) // ratio) for p in R]
    line = f"L{layer:02d} ratio {ratio}: picked {valid.tolist()} {'OK' if valid.tolist() == want else 'MISMATCH ' + str(want)}"
    if pool is not None and layer > src:
        inside = all(pool[i, t[i][t[i] >= 0] // block].all() for i in range(len(R)))
        line += f"; inside L{src:02d}'s pool: {inside}"
    print(line)
