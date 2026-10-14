"""Level 3 of DESIGN decision 2: altar-1's golden logits against the parent GLM-5.3's, on the same token sequences.

  python3 -I level3.py --altar ALTAR_GOLDEN_ROOT --glm GLM53_GOLDEN_ROOT [--runs all,all-f32] [--json OUT]

ALTAR_GOLDEN_ROOT/<run> and GLM53_GOLDEN_ROOT/<run> are glmref.py runs (each needs manifest.json, tokens.npy,
sample_rows.npy and final/: argmax, top20_ids, next_logprob, logits, logits_rows). For every sequence:
- pairs: altar-1 against GLM-5.3 in the same order (all vs all, all-f32 vs all-f32), and the noise floors: each
  model's two orders against each other (GLM-5.3 all vs all-f32: glm53 reference § 10.2; altar-1 the same);
- argmax agreement over every row; the share of rows where the argmax differs but lies in the other's top-5;
- KL(first ‖ second) over the full vocabulary (the 154,856 used ids) at the sample rows (every row under 1,024
  tokens, else every 256th, the first and last 32 and 2047-2049): mean, median, p99, max;
- each model's top-1 against the next token and mean log p(next).
"""
import argparse
import json
import os

import numpy as np

VOCAB_USED = 154856


class R:
    def __init__(self, root, run):
        self.d = os.path.join(root, run)
        self.man = json.load(open(os.path.join(self.d, "manifest.json")))
        self.tok = np.load(os.path.join(self.d, "tokens.npy"))
        self.am = np.load(os.path.join(self.d, "final/argmax.npy"))
        self.top = np.load(os.path.join(self.d, "final/top20_ids.npy"), mmap_mode="r")
        self.nlp = np.load(os.path.join(self.d, "final/next_logprob.npy"))
        self.lr = np.load(os.path.join(self.d, "final/logits_rows.npy"))
        self.lg = np.load(os.path.join(self.d, "final/logits.npy"), mmap_mode="r")
        self.seq = {s["name"]: (s["off"], s["n"]) for s in self.man["sequences"]}


def lsm(x):
    x = np.asarray(x, np.float64)[:VOCAB_USED]
    return x - np.logaddexp.reduce(x)


def pair(A, B, name):
    oa, n = A.seq[name]
    ob, nb = B.seq[name]
    assert n == nb and np.array_equal(A.tok[oa:oa + n], B.tok[ob:ob + n]), f"{name}: tokens differ"
    a, b = A.am[oa:oa + n], B.am[ob:ob + n]
    diff = a != b
    ta, tb = np.asarray(A.top[oa:oa + n, :5]), np.asarray(B.top[ob:ob + n, :5])
    in_b5 = np.array([a[i] in tb[i] for i in range(n)])
    in_a5 = np.array([b[i] in ta[i] for i in range(n)])
    ia = {int(r) - oa: j for j, r in enumerate(A.lr) if oa <= r < oa + n}
    ib = {int(r) - ob: j for j, r in enumerate(B.lr) if ob <= r < ob + n}
    common = sorted(set(ia) & set(ib))
    kl = []
    for p in common:
        pa, pb = lsm(A.lg[ia[p]]), lsm(B.lg[ib[p]])
        kl.append(float(np.sum(np.exp(pa) * (pa - pb))))
    kl = np.array(kl)
    return {"rows": int(n), "argmax_agree": float(1 - diff.mean()),
            "differ_but_first_in_second_top5": float((diff & in_b5).mean()),
            "differ_but_second_in_first_top5": float((diff & in_a5).mean()),
            "differ_outside_top5": float((diff & ~in_b5).mean()),
            "kl_rows": len(common), "kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)),
            "kl_p99": float(np.percentile(kl, 99)), "kl_max": float(kl.max())}


def own(A, name):
    o, n = A.seq[name]
    hit = A.am[o:o + n - 1] == A.tok[o + 1:o + n]
    return {"top1": float(hit.mean()), "mean_logprob": float(np.nanmean(A.nlp[o:o + n - 1]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--altar", required=True)
    ap.add_argument("--glm", required=True)
    ap.add_argument("--runs", default="all,all-f32")
    ap.add_argument("--seqs", default="short,prose,python,agentic")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    r1, r2 = a.runs.split(",")
    # an altar-1 run not finished yet (no final/argmax.npy) is left out, with the pairs that need it
    X = {r: R(a.altar, r) for r in (r1, r2) if os.path.exists(os.path.join(a.altar, r, "final", "argmax.npy"))}
    Y = {r: R(a.glm, r) for r in (r1, r2)}
    res = {}
    hdr = f"{'seq':8s} {'pair':34s} {'argmax=':>8s} {'diff,in top5':>12s} {'diff,out':>9s} {'KL mean':>8s} {'median':>8s} {'p99':>7s} {'max':>7s}"
    print(hdr)
    for s in a.seqs.split(","):
        rs = {}
        pairs = [(f"altar {r1} | glm {r1}", Y[r1], X.get(r1)), (f"altar {r2} | glm {r2}", Y[r2], X.get(r2)),
                 (f"glm {r1} | glm {r2} (floor)", Y[r1], Y[r2]), (f"altar {r1} | altar {r2} (floor)", X.get(r1), X.get(r2)),
                 (f"altar {r2} | glm {r1} (cross order)", Y[r1], X.get(r2)), (f"altar {r1} | glm {r2} (cross order)", Y[r2], X.get(r1))]
        for tag, A, B in pairs:   # KL(A ‖ B): the parent first
            if A is None or B is None:
                continue
            p = pair(A, B, s)
            rs[tag] = p
            print(f"{s:8s} {tag:34s} {p['argmax_agree']:8.4f} {p['differ_but_second_in_first_top5']:12.4f} "
                  f"{1 - p['argmax_agree'] - p['differ_but_second_in_first_top5']:9.4f} {p['kl_mean']:8.4f} "
                  f"{p['kl_median']:8.4f} {p['kl_p99']:7.3f} {p['kl_max']:7.3f}")
        rs["own"] = {f"altar {r}": own(X[r], s) for r in X} | {f"glm {r}": own(Y[r], s) for r in (r1, r2)}
        print(f"{s:8s} top-1 vs next / mean log p: " + "  ".join(f"{k} {v['top1']:.3f} / {v['mean_logprob']:.3f}"
                                                              for k, v in rs["own"].items()))
        res[s] = rs
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
