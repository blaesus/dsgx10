"""Where two runs of the same tokens differ most: positions ranked by next-token logprob gain of run B over
run A, with the text before each, plus a breakdown of the gain by the kind of token being predicted.

  golden/run_ref.sh py /work/golden/cmp_positions.py /out/pre-base /out/off-l01 [--upto 4480] [--top 25]
"""
import argparse

import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--upto", type=int, default=0)
ap.add_argument("--top", type=int, default=25)
a = ap.parse_args()
toks = np.load(f"{a.a}/tokens.npy")
la, lb = np.load(f"{a.a}/next_logprob.npy"), np.load(f"{a.b}/next_logprob.npy")
ta, tb = np.load(f"{a.a}/logits_topi.npy"), np.load(f"{a.b}/logits_topi.npy")
n = min(len(la), len(lb), a.upto or 1 << 30)
la, lb = la[:n], lb[:n]
gain = lb - la
tok = AutoTokenizer.from_pretrained("/models")
print(f"positions 0..{n - 1}: mean logprob A {la.mean():.3f} B {lb.mean():.3f}; B better by >1 nat at "
      f"{int((gain > 1).sum())}, A better by >1 nat at {int((gain < -1).sum())}")
nxt = toks[1:n + 1]
kinds = {"special": [], "whitespace-only": [], "other": []}
special = set(tok.all_special_ids)
for p in range(n):
    t = int(nxt[p])
    s = tok.decode([t])
    k = "special" if t in special or s.startswith("<｜") else "whitespace-only" if not s.strip() else "other"
    kinds[k].append(p)
for k, ps in kinds.items():
    if ps:
        print(f"  next token {k:16s}: {len(ps):5d} positions, mean gain {gain[ps].mean():+.3f}, "
              f"share of total gain {gain[ps].sum() / gain.sum():+.2f}")
print(f"top {a.top} positions where B gains:")
for p in np.argsort(-gain)[: a.top]:
    ctx = tok.decode(toks[max(0, p - 6): p + 1].tolist())
    print(f"  {p:6d} gain {gain[p]:+6.2f} (A {la[p]:6.2f} B {lb[p]:6.2f}) {ctx[-36:]!r} -> {tok.decode([int(nxt[p])])!r}"
          f"  A says {[tok.decode([int(t)]) for t in ta[p, :3]]}")
