"""Next-token accuracy / mean logprob per window, several runs side by side.

  golden/run_ref.sh py /work/golden/cmp_windows.py 4096 5120 256 /out/pre-base /out/pre-noengram
"""
import sys

import numpy as np

a0, a1, w = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
runs = sys.argv[4:]
data = []
for r in runs:
    toks = np.load(f"{r}/tokens.npy")
    lp = np.load(f"{r}/next_logprob.npy")
    topi = np.load(f"{r}/logits_topi.npy")
    data.append((topi[:-1, 0] == toks[1:], lp))
print("window      " + "  ".join(f"{r.split('/')[-1]:>16s}" for r in runs))
for a in range(a0, a1, w):
    cells = []
    for hit, lp in data:
        b = min(len(lp), a + w)
        cells.append(f"{hit[a:b].mean():.2f} {lp[a:b].mean():7.2f}" if b > a else " " * 12)
    print(f"{a:6d}+{w:<4d} " + "  ".join(f"{c:>16s}" for c in cells))
