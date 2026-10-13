"""Engram layout facts per layer, without a GPU: the 24 primes against the table size in config.json,
the hash multipliers, and the hash seeds. A mismatch between the primes and the table would shift
every sub-table offset.

  golden/run_ref.sh py /work/golden/engram_layout_check.py
"""
import json
import os
import sys

sys.path.insert(0, "/work/ref/inference")
from engram import EngramLayout, compute_hash_multipliers  # noqa: E402

cfg = json.load(open(os.path.join("/work/ref/inference", "config.json")))


class A:
    pass


args = A()
for k, v in cfg.items():
    setattr(args, k, v)
lay = EngramLayout.from_args(args)
mult = compute_hash_multipliers(lay.layer_ids, lay.max_ngram_size, cfg["engram_compressed_vocab_size"])
for i, layer in enumerate(lay.layer_ids):
    flat = [p for per in lay.primes[i] for p in per]
    print(f"layer {layer}: sum of primes {sum(flat):,} vs table {lay.num_embeddings[i]:,} "
          f"({'same' if sum(flat) == lay.num_embeddings[i] else 'DIFFERENT'}); primes {flat[0]:,}..{flat[-1]:,}; "
          f"multipliers {mult[i].tolist()}")
