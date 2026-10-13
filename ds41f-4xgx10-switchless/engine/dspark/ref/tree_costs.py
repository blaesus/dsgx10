#!/usr/bin/env python3
# Draft trees (docs/engine.md, "Draft trees: a feasibility study"): the step-cost model T(R, U) = a + b (R - 1) +
# c (U - 6) (R rows, U distinct routed experts per layer), fitted to plain steps without the draft pass (an engine
# bench at 1k, --mode bench --ks 1,6,7,8: consecutive, unrelated and rotated rows, and batching.md's 1k table),
# and bench_chain's 9..16-row overhead over a fit of its own <= 8-row points (bench_chain --rows R --experts U).
# Writes the costs for tree_study.py --costs.
#   tree_costs.py OUT.json ENGINE_BENCH_LOG [BENCH_CHAIN_LOG ...]
import json, re, sys

# batching.md, 1k (2026-10-12): rows, (consecutive, unrelated, rotated) GPU ms, distinct experts per layer
BATCHING = [
    (1, (18.56, 18.57, 18.55), (6.0, 6.0, 6.0)),
    (2, (22.71, 22.29, 24.04), (9.8, 9.6, 11.9)),
    (3, (26.50, 25.34, 29.60), (13.8, 12.7, 17.7)),
    (4, (30.45, 27.66, 34.57), (16.2, 15.3, 23.5)),
    (5, (33.67, 30.49, 39.47), (18.9, 17.2, 29.1)),
    (6, (37.68, 32.43, 44.31), (21.1, 20.1, 34.8)),
]


def solve3(rows):
    """least squares for y = a + b x1 + c x2: rows of (x1, x2, y)"""
    n = len(rows)
    s = [[0.0] * 4 for _ in range(3)]
    for x1, x2, y in rows:
        v = (1.0, x1, x2)
        for i in range(3):
            for j in range(3):
                s[i][j] += v[i] * v[j]
            s[i][3] += v[i] * y
    for i in range(3):  # Gauss-Jordan
        p = max(range(i, 3), key=lambda r: abs(s[r][i]))
        s[i], s[p] = s[p], s[i]
        for r in range(3):
            if r != i:
                f = s[r][i] / s[i][i]
                for c in range(4):
                    s[r][c] -= f * s[i][c]
    a, b, c = (s[i][3] / s[i][i] for i in range(3))
    rms = (sum((a + b * x1 + c * x2 - y) ** 2 for x1, x2, y in rows) / n) ** 0.5
    return a, b, c, rms


def engine_points(log):
    pts = []
    for R, ms, u in BATCHING:
        for k in range(3):
            pts.append((R - 1, u[k] - 6, ms[k], "batching.md"))
    if log:
        txt = open(log, errors="replace").read()
        gpu, uni, rot, dist = {}, {}, {}, {}
        for m in re.finditer(r"k (\d+) step, gpu\s+\(ms\): median ([\d.]+)", txt):
            gpu[int(m.group(1))] = float(m.group(2))
        for m in re.finditer(r"k (\d+) unrelated rows \d+ apart, gpu \(ms\): median ([\d.]+)", txt):
            uni[int(m.group(1))] = float(m.group(2))
        for m in re.finditer(r"k (\d+) the same, routing rotated, gpu \(ms\): median ([\d.]+)", txt):
            rot[int(m.group(1))] = float(m.group(2))
        for m in re.finditer(r"k (\d+) routed experts per layer \(distinct, of \d+\): consecutive ([\d.]+), "
                             r"unrelated ([\d.]+), rotated ([\d.]+)", txt):
            dist[int(m.group(1))] = tuple(float(m.group(q)) for q in (2, 3, 4))
        for k in sorted(dist):
            for q, src in enumerate((gpu, uni, rot)):
                if k in src:
                    pts.append((k - 1, dist[k][q] - 6, src[k], "eb"))
        print("engine bench: " + "; ".join(
            f"k {k}: {gpu.get(k)} / {uni.get(k)} / {rot.get(k)} ms at {dist.get(k)} experts" for k in sorted(dist)))
    return pts


def bench_chain_points(logs):
    pts = []
    for f in logs:
        for line in open(f, errors="replace"):
            m = re.match(r"rows\s+(\d+),\s+(\d+) experts/layer, base\s+step median\s+([\d.]+)", line)
            if m:
                pts.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    # B4's (kernels/README.md): the same binary family, default MoE variants
    pts += [(9, 51, 59.96), (12, 66, 74.25), (16, 86, 93.88)]
    return pts


def main():
    if len(sys.argv) < 3:
        sys.exit("tree_costs.py OUT.json ENGINE_BENCH_LOG [BENCH_CHAIN_LOG ...]")
    pts = engine_points(sys.argv[2])
    a, b, c, rms = solve3([(x1, x2, y) for x1, x2, y, _ in pts])
    print(f"engine fit over {len(pts)} steps: T = {a:.2f} + {b:.2f} (R - 1) + {c:.3f} (U - 6) ms, rms {rms:.2f}")
    bc = bench_chain_points(sys.argv[3:])
    low = [(R - 1, U - 6, t) for R, U, t in bc if R <= 8]
    high = [(R, U, t) for R, U, t in bc if R > 8]
    o16 = 0.0
    if len(low) >= 3:
        a2, b2, c2, rms2 = solve3(low)
        print(f"bench_chain <= 8 rows ({len(low)}): T = {a2:.2f} + {b2:.2f} (R - 1) + {c2:.3f} (U - 6), rms {rms2:.2f}")
        res = [(R, U, t - (a2 + b2 * (R - 1) + c2 * (U - 6))) for R, U, t in high]
        for R, U, r in res:
            print(f"  bench_chain {R} rows, {U} experts: {r:+.2f} ms over the <= 8-row fit")
        if res:
            o16 = sum(r for _, _, r in res) / len(res)
    else:
        print(f"bench_chain: {len(low)} points at <= 8 rows, no fit")
    out = dict(b_row=round(b, 3), c_exp=round(c, 4), o16=round(o16, 2))
    json.dump(out, open(sys.argv[1], "w"))
    print("costs:", out)


if __name__ == "__main__":
    main()
