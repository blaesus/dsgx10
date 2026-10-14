#!/usr/bin/env python3
"""The gate's cost model from a server log (docs/serving.md, "Drafts"): rank 0's log of a ring_job.sh spec phase (or
of g8_job.sh runs) holds, per server start, its "drafts:" line and its runs' "serve: run of N: S steps in T ms, M ms
a step (median), ..." lines. For one-session runs of at least MIN steps (default 8) this takes the median step time
of each start, labelled by its drafts (off, k forced, the gate), and fits the verify step's cost c(k) = c1 + k * drow
by least squares over the forced k (every step there verifies k drafts with k passes): --spec-cost C1,DROW.

  spec_cost.py RANK0_LOG [RANK0_LOG...] [--min 8]
"""
import re
import statistics
import sys


def starts(path):
    cur = None
    for line in open(path, errors='replace'):
        m = re.match(r'^drafts: (.*)', line)
        if m:
            d = m.group(1)
            if d.startswith('off'):
                cur = 'plain'
            else:
                f = re.search(r'forced to (\d+)', d)
                cur = ('k%s' % f.group(1)) if f else 'gate (up to %s)' % re.search(r'up to (\d+)', d).group(1)
            yield cur, None
            continue
        m = re.match(r'^serve: run of (\d+): (\d+) steps in ([\d.]+) ms, ([\d.]+) ms a step \(median\), (\d+) tokens', line)
        if m and cur:
            yield cur, (int(m.group(1)), int(m.group(2)), float(m.group(4)), int(m.group(5)))


def main(a):
    mn = 8
    if '--min' in a:
        i = a.index('--min')
        mn = int(a[i + 1])
        del a[i:i + 2]
    by = {}
    for p in a:
        for cfg, r in starts(p):
            if r and r[0] == 1 and r[1] >= mn:
                by.setdefault(cfg, []).append(r)
    pts = []
    for cfg in sorted(by):
        rs = by[cfg]
        ms = [r[2] for r in rs]
        tps = sum(r[3] for r in rs) / sum(r[1] for r in rs)
        med = statistics.median(ms)
        print('%-14s %3d one-session runs: step median of medians %.2f ms (range %.2f-%.2f), %.2f tokens a step, '
              '%.1f tok/s' % (cfg, len(rs), med, min(ms), max(ms), tps, tps * 1e3 / med))
        if cfg.startswith('k'):
            pts.append((int(cfg[1:]), med))
    if len(pts) >= 2:
        n = len(pts)
        mx = sum(k for k, _ in pts) / n
        my = sum(c for _, c in pts) / n
        drow = sum((k - mx) * (c - my) for k, c in pts) / sum((k - mx) ** 2 for k, _ in pts)
        c1 = my - drow * mx
        print('fit over k %s: c(k) = %.2f + k * %.2f ms  ->  --spec-cost %.1f,%.1f' % (
            ','.join(str(k) for k, _ in pts), c1, drow, c1, drow))
        for k, c in pts:
            print('  k %d: measured %.2f, fit %.2f' % (k, c, c1 + k * drow))


if __name__ == '__main__':
    main(sys.argv[1:])
