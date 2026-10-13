"""Engram writes at and right after special tokens (role and tool markers), against the rest, from
engram_probe.py's npz.

  golden/run_ref.sh py /work/golden/engram_specials.py /out/probe5120.npz
"""
import sys

import numpy as np
from transformers import AutoTokenizer

d = np.load(sys.argv[1])
toks = d["tokens"]
tok = AutoTokenizer.from_pretrained("/models")
names = [tok.decode([int(t)]) for t in toks]
is_sp = np.array([n.startswith("<｜") for n in names])
after = np.zeros_like(is_sp)
for k in (1, 2, 3):
    after[k:] |= is_sp[:-k]
after &= ~is_sp
for layer in (1, 14):
    k = f"L{layer:02d}."
    dn, gate = d[k + "delta_norm"], d[k + "gate"]
    med = np.median(dn)
    print(f"== layer {layer}: |delta|/median: special tokens {np.median(dn[is_sp]) / med:.2f} (max "
          f"{dn[is_sp].max() / med:.1f}), 1-3 after one {np.median(dn[after]) / med:.2f}, others "
          f"{np.median(dn[~is_sp & ~after]) / med:.2f}; gate max-copy mean: special {gate[is_sp].max(1).mean():.3f}, "
          f"after {gate[after].max(1).mean():.3f}, others {gate[~is_sp & ~after].max(1).mean():.3f}")
    for p in np.nonzero(is_sp)[0][:40]:
        print(f"   {p:5d} {names[p]:28s} |d|/med {dn[p] / med:5.1f} gate {np.round(gate[p], 2)}")
