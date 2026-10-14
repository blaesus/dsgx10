"""Batch invariance, bit for bit: one sequence's saved outputs in two glmref.py runs (e.g. `short` alone and inside
a batch; or `prose2400` in a1a-f32-abs and the first 2,400 positions of `prose` in all-f32-abs, which attend to the
same keys). Compares the bytes of every row both runs saved at the same position: each layer's hidden state
(layers/LNN/hidden.npy, layer 78 by its MTP rows), the routed sets and weights, the index sets, the final and MTP
logits at the sample rows, and the per-row argmax, log p(next), logsumexp and top-20 over every position.

  python3 -I cmp_bits.py RUN_A RUN_B --seq short [--seq-b NAME_IN_B] [--max-pos P]

Prints one line per file kind and "ALL BIT-EQUAL" or the kinds that differ; exits 1 on any difference.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


class Run:
    def __init__(self, d, name):
        self.d = d
        man = json.load(open(os.path.join(d, "manifest.json")))
        s = [q for q in man["sequences"] if q["name"] == name]
        assert s, f"{name} not in {d}"
        self.off, self.n = s[0]["off"], s[0]["n"]
        self.sample = np.load(os.path.join(d, "sample_rows.npy")).astype(np.int64)
        # MTP rows: n - 1 a sequence, in sequence order (glmref.py run_mtp)
        self.moff = sum(q["n"] - 1 for q in man["sequences"] if q["off"] < self.off)
        rm = os.path.join(d, "mtp", "rows_main.npy")
        self.rows_main = np.load(rm).astype(np.int64) if os.path.exists(rm) else None

    def load(self, rel):
        p = os.path.join(self.d, rel)
        return np.load(p, mmap_mode="r") if os.path.exists(p) else None

    def pos_of_sample(self):
        """index into the saved sample rows -> position in the sequence (-1: another sequence)."""
        p = self.sample - self.off
        return np.where((p >= 0) & (p < self.n), p, -1)

    def pos_of_mtp_saved(self):
        """layer 78's and mtp/logits' saved rows: the MTP rows whose main row is a sample row, in MTP-row order."""
        keep = set(self.sample.tolist())
        mrows = [i for i, r in enumerate(self.rows_main.tolist()) if r in keep]
        main = self.rows_main[mrows]
        p = main - self.off
        return np.where((p >= 0) & (p < self.n - 1), p, -1)


def pairs(pa, pb, max_pos):
    """indices (ia, ib) of saved rows at the same position in both."""
    ib_of = {int(p): i for i, p in enumerate(pb) if p >= 0 and p < max_pos}
    ia, ib = [], []
    for i, p in enumerate(pa):
        if p >= 0 and int(p) in ib_of:
            ia.append(i)
            ib.append(ib_of[int(p)])
    return np.array(ia, np.int64), np.array(ib, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--seq-b", default=None)
    ap.add_argument("--max-pos", type=int, default=None)
    x = ap.parse_args()
    A, B = Run(x.a, x.seq), Run(x.b, x.seq_b or x.seq)
    P = min(A.n, B.n) if x.max_pos is None else x.max_pos
    bad, kinds = [], 0

    def same(kind, a, b):
        nonlocal kinds
        kinds += 1
        eq = a.shape == b.shape and a.tobytes() == b.tobytes()
        if not eq:
            bad.append(kind)
        return eq

    # hidden states and per-row layer files at the sample rows
    ia, ib = pairs(A.pos_of_sample(), B.pos_of_sample(), P)
    lay = {"hidden": 0, "route": 0, "topk": 0}
    nlay = {"hidden": 0, "route": 0, "topk": 0}
    for la in sorted(glob.glob(os.path.join(x.a, "layers", "L*"))):
        L = os.path.basename(la)
        if L == "L78":
            continue
        ha, hb = A.load(f"layers/{L}/hidden.npy"), B.load(f"layers/{L}/hidden.npy")
        if ha is None or hb is None:
            continue
        nlay["hidden"] += 1
        lay["hidden"] += same(f"{L}/hidden", ha[ia], hb[ib])
        ra = A.load(f"layers/{L}/route_idx.npy")
        if ra is not None:
            nlay["route"] += 1
            ok = True
            for f in ("route_idx", "route_w", "route_margin"):
                fa, fb = A.load(f"layers/{L}/{f}.npy"), B.load(f"layers/{L}/{f}.npy")
                ok &= same(f"{L}/{f}", fa[A.off:A.off + P], fb[B.off:B.off + P])
            lay["route"] += ok
        ta = A.load(f"layers/{L}/topk_{x.seq}.npy")
        if ta is not None:
            tb = B.load(f"layers/{L}/topk_{x.seq_b or x.seq}.npy")
            k = max(0, P - 2048)
            nlay["topk"] += 1
            lay["topk"] += same(f"{L}/topk", ta[:k], tb[:k]) & same(
                f"{L}/topk_margin", A.load(f"layers/{L}/topk_margin_{x.seq}.npy")[:k],
                B.load(f"layers/{L}/topk_margin_{x.seq_b or x.seq}.npy")[:k])
    print(f"positions 0..{P - 1}; {len(ia)} sample rows in both")
    for k in lay:
        print(f"layers 0-77 {k:7s} bit-equal on {lay[k]} of {nlay[k]} layers")
    # final: full logits at the sample rows, per-row outputs everywhere
    la, lb = A.load("final/logits_rows.npy"), B.load("final/logits_rows.npy")
    fa, fb = pairs(la.astype(np.int64) - A.off, lb.astype(np.int64) - B.off, P)
    print("final logits", len(fa), "rows:", "bit-equal" if same("final/logits", A.load("final/logits.npy")[fa],
                                                               B.load("final/logits.npy")[fb]) else "DIFFER")
    for f in ("argmax", "next_logprob", "lse", "top20_ids", "top20_logprobs"):
        a, b = A.load(f"final/{f}.npy")[A.off:A.off + P], B.load(f"final/{f}.npy")[B.off:B.off + P]
        if f == "next_logprob" and P == min(A.n, B.n) and A.n != B.n:
            a, b = a[:P - 1], b[:P - 1]   # the shorter run's last row has no next token
        print(f"final {f:15s}", "bit-equal" if same(f"final/{f}", a, b) else "DIFFER")
    # the MTP layer and its head
    if A.rows_main is not None and B.rows_main is not None:
        Q = P - 1
        ma, mb = pairs(A.pos_of_mtp_saved(), B.pos_of_mtp_saved(), Q)
        print("layer 78 hidden", len(ma), "rows:", "bit-equal" if same(
            "L78/hidden", A.load("layers/L78/hidden.npy")[ma], B.load("layers/L78/hidden.npy")[mb]) else "DIFFER")
        print("mtp logits", len(ma), "rows:", "bit-equal" if same(
            "mtp/logits", A.load("mtp/logits.npy")[ma], B.load("mtp/logits.npy")[mb]) else "DIFFER")
        for f in ("argmax", "lse", "top20_ids", "top20_logprobs"):
            a = A.load(f"mtp/{f}.npy")[A.moff:A.moff + Q]
            b = B.load(f"mtp/{f}.npy")[B.moff:B.moff + Q]
            print(f"mtp {f:15s}", "bit-equal" if same(f"mtp/{f}", a, b) else "DIFFER")
        ra, rb = A.load("layers/L78/route_idx.npy"), B.load("layers/L78/route_idx.npy")
        print("layer 78 routes", "bit-equal" if same("L78/route_idx", ra[A.moff:A.moff + Q], rb[B.moff:B.moff + Q])
              else "DIFFER")
    print(f"{kinds} comparisons:", "ALL BIT-EQUAL" if not bad else f"{len(bad)} DIFFER: {bad[:20]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
