#!/usr/bin/env python3
"""Per-kernel profile of the decode step's graph from an nsys report (glm_run trace under run.sh's trace step).

  step_profile.py REPORT.nsys-rep [REPORT2 ...] [--rows-order 1,6] [--top 40]

Each report holds 8 graph steps for every row count of the trace (in --rows-order). For the main stream's kernels
of the last 4 steps of each row count it prints, per kernel kind (name + grid + where in the layer), the median
over steps of the summed *critical-path increments* (end of the kernel minus the end of the kernel before it on
the stream: what the kernel adds to the step with PDL overlapping the launches) and of the summed durations.
With several reports (the four ranks), one column per report.

  step_profile.py --prefill REPORT.nsys-rep [REPORT2 ...] [--top 40]

A prefill chunk (glm_run pftrace under run.sh's pftrace step: one eager chunk in the profiler range): the main
stream's kernels summed per kind (name, grid, and for the projections and collectives the kernel before it) and per
class (projections, indexer, attention, MoE prefill, collectives, the rest), critical-path increments and durations.
"""
import argparse
import collections
import os
import sqlite3
import statistics
import subprocess
import sys


def load(rep):
    db = rep.replace(".nsys-rep", ".sqlite")
    if not os.path.exists(db) or os.path.getmtime(db) < os.path.getmtime(rep):
        subprocess.run(["/usr/local/cuda-13.0/bin/nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                        "-o", db, rep], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    rows = c.execute("select start, end, streamId, shortName, gridX, gridY, blockX from CUPTI_ACTIVITY_KIND_KERNEL "
                     "order by start").fetchall()
    return [(s, e, st, names[n], gx, gy, bx) for s, e, st, n, gx, gy, bx in rows]


def steps_of(ks):
    """Split the main stream's kernels into steps at k_step_in; the side streams' kernels separately."""
    main = collections.Counter(k[2] for k in ks if k[3].startswith("k_step_in"))
    ms = main.most_common(1)[0][0]
    steps, cur, side = [], None, []
    for k in ks:
        if k[2] != ms:
            side.append(k)
            continue
        if k[3].startswith("k_step_in"):
            cur = []
            steps.append(cur)
        if cur is not None:
            cur.append(k)
    return steps, side


def label(seq):
    """A kind for each kernel of a step: its name, grid and the name of the kernel before it (the gemv's role)."""
    out = []
    for i, k in enumerate(seq):
        n = k[3]
        prev = seq[i - 1][3] if i else ""
        out.append(f"{n} g{k[4]}" + (f" <{prev}" if n in ("k_gemv", "k_allreduce_hidden", "k_allgather_rows") else ""))
    return out


def profile(ks, nrows, keep=4):
    steps, side = steps_of(ks)
    per = len(steps) // nrows
    res = []
    for b in range(nrows):
        grp = steps[b * per:(b + 1) * per][-keep:]
        agg = collections.defaultdict(lambda: [[], [], 0])
        tot = []
        for st in grp:
            labs = label(st)
            sums = collections.defaultdict(lambda: [0.0, 0.0, 0])
            for i, k in enumerate(st):
                inc = k[1] - (st[i - 1][1] if i else k[0])
                inc = max(inc, 0)
                sums[labs[i]][0] += inc
                sums[labs[i]][1] += k[1] - k[0]
                sums[labs[i]][2] += 1
            for lab, v in sums.items():
                agg[lab][0].append(v[0])
                agg[lab][1].append(v[1])
                agg[lab][2] = v[2]
            tot.append(st[-1][1] - st[0][0])
        res.append(({k: (statistics.median(v[0]) / 1e6, statistics.median(v[1]) / 1e6, v[2]) for k, v in agg.items()},
                    statistics.median(tot) / 1e6, len(grp[0])))
    return res


CLASSES = [  # (class, kernel-name prefixes), first match
    ("projections (FP8 gemm / GEMV)", ("k_gemm", "k_gemv")),
    ("indexer (weights, scores, selection)", ("k_index_weights", "k_q_stage", "k_score", "k_sel_")),
    ("attention (prep, sparse MLA)", ("k_prep", "k_attn")),
    ("MoE prefill (router GEMV excluded)", ("k_route", "k_pup", "k_pdown", "k_count", "k_scan", "k_scatter",
                                             "k_combine", "k_quant")),
    ("collectives (with waiting)", ("allgather", "reduce_scatter", "allreduce", "alltoall", "k_allgather",
                                    "k_allreduce", "k_ar_norm")),
    ("norms, residual, embedding, other", ("",)),
]


def klass(name):
    for c, ps in CLASSES:
        if any(name.startswith(p) for p in ps):
            return c
    return CLASSES[-1][0]


def prefill_profile(ks):
    """One chunk: the main stream's kernels (the stream with the most), per kind and per class."""
    main = collections.Counter(k[2] for k in ks).most_common(1)[0][0]
    seq = [k for k in ks if k[2] == main]
    side = [k for k in ks if k[2] != main]
    kinds = collections.defaultdict(lambda: [0.0, 0.0, 0])
    classes = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for i, k in enumerate(seq):
        inc = max(k[1] - (seq[i - 1][1] if i else k[0]), 0)
        n = k[3]
        prev = seq[i - 1][3] if i else ""
        role = n.startswith(("k_gemm", "k_gemv")) or klass(n).startswith("collectives")
        lab = f"{n} g{k[4]}x{k[5]}" + (f" <{prev}" if role else "")
        for d in (kinds[lab], classes[klass(n)]):
            d[0] += inc
            d[1] += k[1] - k[0]
            d[2] += 1
    total = seq[-1][1] - seq[0][0]
    side_ms = sum(k[1] - k[0] for k in side)
    return kinds, classes, total, len(seq), side_ms, streams_of(ks, main)


def union(iv):
    """Sorted disjoint union of (start, end) intervals."""
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def measure(u):
    return sum(e - s for s, e in u)


def intersect(a, b):
    i = j = t = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        t += max(0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t


def streams_of(ks, main):
    """Per stream (the main one first): kernels, busy time (the union of its kernels), the kernel time per class,
    and the busy time overlapped with the main stream's; the main stream's idle gaps (waiting for another stream,
    or launch gaps) between its first start and last end."""
    by = collections.defaultdict(list)
    for k in ks:
        by[k[2]].append(k)
    mu = union([(k[0], k[1]) for k in by[main]])
    out = []
    for sid in [main] + sorted(s for s in by if s != main):
        u = union([(k[0], k[1]) for k in by[sid]])
        cl = collections.defaultdict(float)
        for k in by[sid]:
            cl[klass(k[3])] += k[1] - k[0]
        out.append((sid, len(by[sid]), measure(u), intersect(u, mu) if sid != main else measure(u), dict(cl)))
    return out


def stream_report(profs):
    print("\nper stream (ms; busy = the union of its kernels; with main = the part of it while the main stream runs a "
          "kernel; main idle = chunk - main busy):")
    for i, p in enumerate(profs):
        total = p[2]
        for j, (sid, n, busy, ov, cl) in enumerate(p[5]):
            what = "main" if j == 0 else "side"
            extra = f" idle {(total - busy) / 1e6:8.1f}" if j == 0 else f" with main {ov / 1e6:8.1f}"
            print(f"  report {i} {what} stream {sid}: {n:6d} kernels, busy {busy / 1e6:8.1f}{extra}; " +
                  ", ".join(f"{c.split(' (')[0]} {v / 1e6:.1f}" for c, v in sorted(cl.items(), key=lambda x: -x[1])))


def prefill_main(a):
    profs = [prefill_profile(load(r)) for r in a.reports]
    print("chunk ms (first kernel start to last end): " + " ".join(f"{p[2] / 1e6:.1f}" for p in profs) +
          f"; {profs[0][3]} kernels on the main stream; side streams' kernel time " +
          " ".join(f"{p[4] / 1e6:.1f}" for p in profs) + " ms")
    hdr = " ".join(f"{'inc' + str(i):>9s} {'dur' + str(i):>9s} {'%':>5s}" for i in range(len(profs)))
    print(f"\n{'class':44s} {'n':>6s} " + hdr)
    for c, _ in CLASSES:
        v = [p[1].get(c, (0, 0, 0)) for p in profs]
        print(f"{c[:44]:44s} {v[0][2]:6d} " + " ".join(f"{x[0] / 1e6:9.1f} {x[1] / 1e6:9.1f} {100 * x[0] / p[2]:5.1f}"
                                                     for x, p in zip(v, profs)))
    print(f"\n{'kernel (grid, after)':70s} {'n':>6s} " + hdr)
    keys = sorted(profs[0][0], key=lambda k: -profs[0][0][k][0])
    for k in keys[:a.top]:
        v = [p[0].get(k, (0, 0, 0)) for p in profs]
        print(f"{k[:70]:70s} {v[0][2]:6d} " + " ".join(f"{x[0] / 1e6:9.1f} {x[1] / 1e6:9.1f} {100 * x[0] / p[2]:5.1f}"
                                                     for x, p in zip(v, profs)))
    stream_report(profs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--rows-order", default="1,6")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--prefill", action="store_true")
    a = ap.parse_args()
    if a.prefill:
        return prefill_main(a)
    rows = [int(x) for x in a.rows_order.split(",")]
    profs = [profile(load(r), len(rows)) for r in a.reports]
    for bi, R in enumerate(rows):
        print(f"== {R} rows: step ms (first kernel start to last end) " +
              " ".join(f"{p[bi][1]:.3f}" for p in profs) + f"; {profs[0][bi][2]} kernels a step")
        keys = sorted(profs[0][bi][0], key=lambda k: -profs[0][bi][0][k][0])
        print(f"{'kernel (grid, after)':58s} {'n':>4s} " + " ".join(f"{'inc' + str(i):>8s} {'dur' + str(i):>8s}"
                                                               for i in range(len(profs))))
        for k in keys[:a.top]:
            v = [p[bi][0].get(k, (0, 0, 0)) for p in profs]
            print(f"{k[:58]:58s} {v[0][2]:4d} " + " ".join(f"{x[0]:8.3f} {x[1]:8.3f}" for x in v))
        print()


if __name__ == "__main__":
    sys.exit(main())
