"""Compare two glmref.py runs on a sequence they share (e.g. `short` run alone and inside the 4-sequence
batch, whose GEMMs then block differently): the reference's own order variance.

  python3 cmp_runs.py RUN_A RUN_B [--seq short]

Prints the argmax agreement, |Δ log p| of the next token and over A's top-20, the KL(A‖B) estimated on
the top-20 of the sampled rows' full logits, per-layer routing agreement, and the hidden-state relative
error at a few layers.
"""
import argparse
import glob
import json
import os

import numpy as np


def seq_rows(run, name):
    man = json.load(open(os.path.join(run, "manifest.json")))
    s = [s for s in man["sequences"] if s["name"] == name][0]
    return s["off"], s["n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--seq", default="short")
    x = ap.parse_args()
    oa, n = seq_rows(x.a, x.seq)
    ob, nb = seq_rows(x.b, x.seq)
    assert n == nb
    ld = lambda run, f, o: np.load(os.path.join(run, f))[o:o + n]
    am_a, am_b = ld(x.a, "final/argmax.npy", oa), ld(x.b, "final/argmax.npy", ob)
    print(f"{x.seq}: {n} rows; argmax equal {np.mean(am_a == am_b):.4f} ({int((am_a != am_b).sum())} differ)")
    la, lb = ld(x.a, "final/next_logprob.npy", oa)[:-1], ld(x.b, "final/next_logprob.npy", ob)[:-1]
    d = np.abs(la - lb)
    print(f"|Δ log p(next)|: mean {d.mean():.2e}, max {d.max():.2e}; bit-equal {np.mean(la == lb):.4f}")
    # full logits at rows both runs sampled
    ra, rb = np.load(os.path.join(x.a, "final/logits_rows.npy")), np.load(os.path.join(x.b, "final/logits_rows.npy"))
    ia = {int(r) - oa: j for j, r in enumerate(ra) if oa <= r < oa + n}
    ib = {int(r) - ob: j for j, r in enumerate(rb) if ob <= r < ob + n}
    common = sorted(set(ia) & set(ib))
    if common:
        A = np.load(os.path.join(x.a, "final/logits.npy"), mmap_mode="r")
        B = np.load(os.path.join(x.b, "final/logits.npy"), mmap_mode="r")
        kl, mx = [], 0.0
        for p in common:
            a = np.asarray(A[ia[p]], np.float64)
            b = np.asarray(B[ib[p]], np.float64)
            pa = a - np.logaddexp.reduce(a)
            pb = b - np.logaddexp.reduce(b)
            kl.append(float(np.sum(np.exp(pa) * (pa - pb))))
            mx = max(mx, float(np.abs(a - b).max()))
        kl = np.array(kl)
        print(f"full logits at {len(common)} rows: KL(A‖B) mean {kl.mean():.2e}, max {kl.max():.2e}; max |Δ logit| {mx:.3f}")
    # routing and hidden states per layer
    sa, sb = np.load(os.path.join(x.a, "sample_rows.npy")), np.load(os.path.join(x.b, "sample_rows.npy"))
    ha = {int(r) - oa: j for j, r in enumerate(sa) if oa <= r < oa + n}
    hb = {int(r) - ob: j for j, r in enumerate(sb) if ob <= r < ob + n}
    hc = sorted(set(ha) & set(hb))
    print("layer  route-sets-equal  hidden rel-l2 (median over sampled rows)")
    for d_a in sorted(glob.glob(os.path.join(x.a, "layers", "L*"))):
        L = os.path.basename(d_a)
        d_b = os.path.join(x.b, "layers", L)
        if L == "L78" or not os.path.exists(d_b):
            continue
        line = f"{L}  "
        if os.path.exists(os.path.join(d_a, "route_idx.npy")):
            ra_ = np.sort(np.load(os.path.join(d_a, "route_idx.npy"))[oa:oa + n], axis=1)
            rb_ = np.sort(np.load(os.path.join(d_b, "route_idx.npy"))[ob:ob + n], axis=1)
            line += f"{np.mean((ra_ == rb_).all(1)):.4f}"
        else:
            line += "  -   "
        if hc:
            HA = np.load(os.path.join(d_a, "hidden.npy"))[[ha[p] for p in hc]]
            HB = np.load(os.path.join(d_b, "hidden.npy"))[[hb[p] for p in hc]]
            rel = np.linalg.norm(HA - HB, axis=1) / np.linalg.norm(HA, axis=1)
            line += f"   {np.median(rel):.2e}"
        print(line)


if __name__ == "__main__":
    main()
