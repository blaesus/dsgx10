"""How far a variant run drifts from a reference run of the same tokens along the sequence: the relative
error of each block's output at the saved positions, by position band, then top-1 agreement and clear
flips before and after an onset. A clear flip: the argmaxes differ and each run puts its own pick more
than 1 nat above the other's (an id missing from a run's top-32 counts at its 32nd logit, which makes
that margin a lower bound). Same rule as cmp_runs.py, which needs full logits.

  golden/run_ref.sh py /work/golden/cmp_drift.py /out/long /out/long-splitk --onset 16864
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a", help="reference run (golden)")
ap.add_argument("b", help="variant run")
ap.add_argument("--onset", type=int, default=16864)
ap.add_argument("--tensor", default="out", help="per-layer tensor: out, mid, attn_out, ffn_out, ...")
a = ap.parse_args()
L = json.load(open(f"{a.a}/manifest.json"))["L"]
pos = np.load(f"{a.a}/positions.npy")
assert (pos == np.load(f"{a.b}/positions.npy")).all(), "the runs saved different positions"
bands = [("0", 0, 1), ("256-1024", 256, 1025), ("1280-4096", 1025, 4097), (f"4352-{a.onset - 1}", 4097, a.onset),
         (f"{a.onset}-{L - 513}", a.onset, L - 512), ("last 512", L - 512, L)]
sel = [(name, (pos >= lo) & (pos < hi)) for name, lo, hi in bands]


def rel(x, y):
    x, y = x.reshape(len(x), -1).astype(np.float64), y.reshape(len(y), -1).astype(np.float64)
    return np.linalg.norm(y - x, axis=1) / np.maximum(np.linalg.norm(x, axis=1), 1e-30)


print(f"relative error of L*.{a.tensor} at the saved positions, median (max) per band; "
      f"positions per band: " + ", ".join(f"{n} {int(m.sum())}" for n, m in sel))
print("| layer | " + " | ".join(n for n, _ in sel) + " |")
print("|---" * (len(sel) + 1) + "|")
names = [f"L{i:02d}.{a.tensor}" for i in range(40)] + ["final_hidden"]
worst = []
for nm in names:
    r = rel(np.load(f"{a.a}/{nm}.npy"), np.load(f"{a.b}/{nm}.npy"))
    print(f"| {nm} | " + " | ".join(f"{np.median(r[m]):.3f} ({r[m].max():.3f})" if m.any() else "-" for _, m in sel) + " |")
    worst.append((r.max(), nm, int(pos[r.argmax()])))
print("largest: " + ", ".join(f"{nm} {v:.3g} at {p}" for v, nm, p in sorted(worst, reverse=True)[:5]))

toks = np.load(f"{a.a}/tokens.npy")
ta, tb = np.load(f"{a.a}/logits_topi.npy"), np.load(f"{a.b}/logits_topi.npy")
va, vb = np.load(f"{a.a}/logits_topv.npy").astype(np.float64), np.load(f"{a.b}/logits_topv.npy").astype(np.float64)


def logit_of(ids, vals, p, t):
    hit = np.nonzero(ids[p] == t)[0]
    return vals[p, hit[0]] if len(hit) else vals[p, -1]


flip = ta[:, 0] != tb[:, 0]
clear = np.zeros(L, bool)
for p in np.nonzero(flip)[0]:
    ma_ = va[p, 0] - logit_of(ta, va, p, tb[p, 0])
    mb_ = vb[p, 0] - logit_of(tb, vb, p, ta[p, 0])
    clear[p] = ma_ > 1.0 and mb_ > 1.0
hit_a = np.append(ta[:-1, 0] == toks[1:], False)
hit_b = np.append(tb[:-1, 0] == toks[1:], False)
print("\n| positions | top-1 agreement | flips | clear flips | ref right | variant right |")
print("|---|---|---|---|---|---|")
for name, lo, hi in (("first 512", 0, 512), (f"0-{a.onset - 1} (before onset)", 0, a.onset),
                     (f"{a.onset}-{L - 1} (after)", a.onset, L), ("all", 0, L)):
    s = slice(lo, hi)
    print(f"| {name} | {1 - flip[s].mean():.3f} | {int(flip[s].sum())} | {int(clear[s].sum())} | "
          f"{int(hit_a[s].sum())} | {int(hit_b[s].sum())} |")
