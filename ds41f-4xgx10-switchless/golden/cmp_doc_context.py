"""The same text in two contexts: for the positions of run A (a transcript) where run B gains most, find
the identical 12-token stretch in run D (the file as a plain document) and show D's logprob there.

  golden/run_ref.sh py /work/golden/cmp_doc_context.py /out/orig-base /out/off-l01 /out/doc-textwrap [--top 12]
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("doc")
ap.add_argument("--top", type=int, default=12)
ap.add_argument("--span", type=int, default=12)
a = ap.parse_args()
ta, td = np.load(f"{a.a}/tokens.npy"), np.load(f"{a.doc}/tokens.npy")
la, lb, ld = (np.load(f"{r}/next_logprob.npy") for r in (a.a, a.b, a.doc))
n = min(len(la), len(lb))
gain = lb[:n] - la[:n]
index = {}
for q in range(a.span - 1, len(td) - 1):
    index.setdefault(td[q - a.span + 1: q + 2].tobytes(), q)  # span tokens ending at q, plus the next one
found = 0
print(f"{'pos A':>6s} {'A':>7s} {'B':>7s} | {'pos D':>6s} {'D':>7s}")
for p in np.argsort(-gain):
    if p < a.span - 1:
        continue
    q = index.get(ta[p - a.span + 1: p + 2].tobytes())
    if q is None:
        continue
    print(f"{p:6d} {la[p]:7.2f} {lb[p]:7.2f} | {q:6d} {ld[q]:7.2f}")
    found += 1
    if found == a.top:
        break
