#!/usr/bin/env python3
"""Which E2M1 subsets to use for the 3-bit grid code: every single subset and the best pairs.

    python -I subset_study.py MASTER --layers 3 40 --experts 0,64,128,192 [--rows 512] [--threads 4]

For each (layer, expert, projection) takes `rows` rows (a fixed seeded choice), quantises them with
formats.levelset3 for every subset of 4 E2M1 magnitudes (35 without 0, 35 with 0 and 3 others) and then
for every pair among the 10 best singles (the e4m3 scale's sign bit selecting the subset of each group), and
prints the mean relative RMS weight error per projection type. Deterministic.
"""
import argparse, itertools, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import formats as F  # noqa: E402
from run import Master, nkey, PROJ  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("master")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--experts", default="0,64,128,192")
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    M = Master(a.master)
    E = [.5, 1, 1.5, 2, 3, 4, 6]
    singles = [tuple(c) for c in itertools.combinations(E, 4)] + [(0,) + c for c in itertools.combinations(E, 3)]
    ws = []
    for L in a.layers:
        for e in map(int, a.experts.split(",")):
            for p in PROJ:
                w = M.get(nkey(L, e, p))[0]
                g = torch.Generator().manual_seed(7)
                ws.append((p, w[torch.randperm(w.shape[0], generator=g)[:a.rows]].contiguous()))

    def score(subs):
        r = {p: [] for p in PROJ}
        for p, w in ws:
            r[p].append(((F.levelset3_dequant(F.levelset3(w, subs)) - w).norm() / w.norm()).item())
        return {p: sum(v) / len(v) for p, v in r.items()}
    res = sorted(((score([s]), s) for s in singles), key=lambda t: sum(t[0].values()))
    print("single subsets, mean rel RMS error gate / up / down (best 10):")
    for sc, s in res[:10]:
        print(f"  {str(s):28s} " + " / ".join(f"{sc[p]:.4f}" for p in PROJ))
    top = [s for _, s in res[:10]]
    pr = sorted(((score([s, t]), (s, t)) for s, t in itertools.combinations(top, 2)), key=lambda t: sum(t[0].values()))
    print("pairs of subsets, the scale's sign bit selecting (best 5):")
    for sc, s in pr[:5]:
        print(f"  {str(s):56s} " + " / ".join(f"{sc[p]:.4f}" for p in PROJ))


if __name__ == "__main__":
    sys.exit(main())
