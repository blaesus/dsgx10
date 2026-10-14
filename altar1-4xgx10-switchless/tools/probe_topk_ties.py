#!/usr/bin/env python3
"""A3e: the order torch.topk returns over 168 experts, as the router calls it (topk(sel, 8, dim=-1, sorted=False) on
[T, 168] f32), on a device: is the output in descending order, and which ids win among equal keys (and in what
order)? route.cu sorts by (key descending, id ascending); this probe says whether torch agrees on the device the
reference runs on. GPU use only through ring.sh.

  python -I tools/probe_topk_ties.py [--device cuda|cpu] [--rows 4096]
"""
import argparse

import torch


def canon(row):
    return sorted(range(len(row)), key=lambda e: (-row[e], e))[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rows", type=int, default=4096)
    a = ap.parse_args()
    g = torch.Generator().manual_seed(20261014)
    print(f"torch {torch.__version__} device {a.device}" + (f" ({torch.cuda.get_device_name()})" if a.device == "cuda" else ""))
    cases = {
        # distinct keys like the router's (sigmoid + a bias of ~28)
        "distinct": 28.0 + torch.sigmoid(torch.randn(a.rows, 168, generator=g)),
        # ties everywhere: integer keys 0..5
        "ties (keys 0..5)": torch.randint(0, 6, (a.rows, 168), generator=g).float(),
        # distinct keys with an exact tie at the 8th place: keys 28 + k / 256, the top ranks unique, 8th..10th equal
        "tie at 8th": None,
    }
    t = torch.empty(a.rows, 168)
    for r in range(a.rows):
        perm = torch.randperm(168, generator=g)
        v = torch.arange(168, 0, -1).float()
        v[8] = v[7]
        v[9] = v[7]  # ranks 8, 9, 10 share the 8th key
        t[r, perm] = 28.0 + v / 256
    cases["tie at 8th"] = t
    for name, x in cases.items():
        xd = x.to(a.device)
        for sorted_ in (False, True):
            v, i = torch.topk(xd, 8, dim=-1, sorted=sorted_)
            i = i.cpu().tolist()
            xs = x.tolist()
            same_set = same_order = desc = 0
            for r in range(a.rows):
                c = canon(xs[r])
                same_set += set(i[r]) == set(c)
                same_order += i[r] == c
                vals = [xs[r][e] for e in i[r]]
                desc += all(vals[j] >= vals[j + 1] for j in range(7))
            print(f"{name:18s} sorted={sorted_!s:5s} rows {a.rows}: set = lowest ids among ties {same_set}, "
                  f"order = (key desc, id asc) {same_order}, descending keys {desc}")


if __name__ == "__main__":
    main()
