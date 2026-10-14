#!/usr/bin/env python3
"""Tables from run.py's results: per layer, format and projection the mean bpw, weight error, Gaussian-input
output error, the expert-level output error with its spread, the hardest experts, CPU time a tensor, and the
NVFP4 bit-exact count. Optionally compares two runs' packed-byte hashes (the reproducibility check).

    python -I summarise.py RESULTS.jsonl [...] [--experts 0,51,...] [--compare A.jsonl B.jsonl]
"""
import argparse, json, statistics as st, sys
from collections import defaultdict

PROJ = ("gate", "up", "down")


def load(paths):
    rs = []
    for p in paths:
        rs += [json.loads(l) for l in open(p) if l.strip()]
    return rs


def q(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(f * len(v)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="*")
    ap.add_argument("--experts", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()
    if a.compare:
        A, B = (load([p]) for p in a.compare)
        key = lambda r, p: (r["layer"], r["expert"], r["format"], p)
        ha = {key(r, p): d["sha256"] for r in A for p, d in r.get("proj", {}).items()}
        hb = {key(r, p): d["sha256"] for r in B for p, d in r.get("proj", {}).items()}
        common = sorted(set(ha) & set(hb))
        same = sum(ha[k] == hb[k] for k in common)
        print(f"reproducibility: {same} of {len(common)} packed tensors byte-identical")
        for k in common:
            if ha[k] != hb[k]:
                print("  differs:", k)
        return
    rs = load(a.results)
    if a.experts:
        keep = {int(x) for x in a.experts.split(",")}
        rs = [r for r in rs if r["expert"] in keep]
    by = defaultdict(list)
    for r in rs:
        by[(r["layer"], r["format"])].append(r)
    nv = [d for r in rs if r["format"] == "nvfp4" for d in r["proj"].values()]
    if nv:
        ok = sum(d["nv_weight_eq"] and d["nv_scale_eq"] and d["nv_scale2_eq"] for d in nv)
        print(f"NVFP4 bit-exact (codes, block scales, scale_2): {ok} of {len(nv)} tensors\n")
    for L in sorted({k[0] for k in by}):
        print(f"layer {L}")
        print("| format | n | bpw | weight err gate / up / down | output err (Gaussian x) gate / up / down |"
              " expert output err mean (p90, max) | s a tensor (CPU, wall) |")
        print("|---|--:|--:|---|---|---|---|")
        rows = sorted((k for k in by if k[0] == L), key=lambda k: -st.mean(r["expert_oerr"] for r in by[k]))
        for k in rows:
            R = by[k]
            ex = [r["expert_oerr"] for r in R]
            if "proj" not in R[0]:
                print(f"| {k[1]} | {len(R)} | | | | {st.mean(ex):.4f} ({q(ex, .9):.4f}, {max(ex):.4f}) | |")
                continue
            m = lambda f, p: st.mean(r["proj"][p][f] for r in R)
            bpw = st.mean(r["proj"][p]["bpw"] for r in R for p in PROJ)
            cpu = st.mean(r["proj"][p]["cpu_s"] for r in R for p in PROJ)
            wall = st.mean(r["proj"][p]["wall_s"] for r in R for p in PROJ)
            print(f"| {k[1]} | {len(R)} | {bpw:.3f} | " + " / ".join(f"{m('werr', p):.4f}" for p in PROJ) + " | "
                  + " / ".join(f"{m('oerr', p):.4f}" for p in PROJ) +
                  f" | {st.mean(ex):.4f} ({q(ex, .9):.4f}, {max(ex):.4f}) | {cpu:.1f}, {wall:.1f} |")
        for f in ("nvfp4", "e2m1sub3x2_g16", "trellis_k3"):
            if (L, f) in by:
                R = sorted(by[(L, f)], key=lambda r: -r["expert_oerr"])
                print(f"  hardest experts by {f} expert output error: " +
                      ", ".join(f"{r['expert']} ({r['expert_oerr']:.4f})" for r in R[:6]))
                for p in PROJ:
                    w = [r["proj"][p]["werr"] for r in by[(L, f)]]
                    o = [r["proj"][p]["oerr"] for r in by[(L, f)]]
                    print(f"    {p}: weight err min {min(w):.4f} median {st.median(w):.4f} max {max(w):.4f}; "
                          f"output err min {min(o):.4f} median {st.median(o):.4f} max {max(o):.4f}")
        print()


if __name__ == "__main__":
    sys.exit(main())
