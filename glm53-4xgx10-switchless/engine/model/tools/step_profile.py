#!/usr/bin/env python3
"""Per-kernel profile of the decode step's graph from an nsys report (glm_run trace under run.sh's trace step).

  step_profile.py REPORT.nsys-rep [REPORT2 ...] [--rows-order 1,6] [--top 40]

Each report holds 8 graph steps for every row count of the trace (in --rows-order). For the main stream's kernels
of the last 4 steps of each row count it prints, per kernel kind (name + grid + where in the layer), the median
over steps of the summed *critical-path increments* (end of the kernel minus the end of the kernel before it on
the stream: what the kernel adds to the step with PDL overlapping the launches) and of the summed durations.
With several reports (the four ranks), one column per report.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--rows-order", default="1,6")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
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
