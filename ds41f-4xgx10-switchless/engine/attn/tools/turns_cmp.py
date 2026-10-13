#!/usr/bin/env python3
"""turns_cmp.py [--gain G] [--worst W] BASE.txt NEW.txt: two tjoin.py tables, row by row: the server's prefill
ms and rank 0's encoder ms (layers 0-19 of the turn), new / base, their geometric means, and the rule a
default change must pass on the prefill: no turn more than W slower (0.02) and the geomean at least G faster
(0.02). Exits 0 if it passes, 1 if not."""
import argparse
import math
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--gain', type=float, default=0.02)
ap.add_argument('--worst', type=float, default=0.02)
ap.add_argument('base')
ap.add_argument('new')
args = ap.parse_args()


def table(path):
    rows = []
    for ln in open(path):
        if ln.startswith('turn') or not ln.strip():
            continue
        head, _, tail = ln.partition('|')
        f = head.split()
        label = ' '.join(f[:-4])  # everything before the last four numbers (prompt, new, ttft, prefill)
        prompt, new, ttft, pre = int(f[-4]), int(f[-3]), float(f[-2]), float(f[-1])
        t = tail.split()
        enc = float(t[3]) if len(t) > 3 else None
        rows.append((label, prompt, new, ttft, pre, enc))
    return rows


base, new = table(args.base), table(args.new)
if not base or len(base) != len(new) or any(a[0] != b[0] for a, b in zip(base, new)):
    sys.exit('turns_cmp.py: the tables do not match row for row')
print('%-18s %7s %6s %9s %9s %6s %9s %9s %6s' % ('turn', 'prompt', 'new', 'prefill', 'base', 'ratio', 'encoder',
                                                   'base', 'ratio'))
lp, le, worst = [], [], (0.0, '')
for a, b in zip(base, new):
    rp = b[4] / a[4]
    lp.append(math.log(rp))
    re_ = b[5] / a[5] if a[5] and b[5] else None
    if re_:
        le.append(math.log(re_))
    if rp > worst[0]:
        worst = (rp, b[0])
    print('%-18s %7d %6d %9.1f %9.1f %6.3f %9s %9s %6s' % (
        b[0], b[1], b[2], b[4], a[4], rp, '%.1f' % b[5] if b[5] else '-', '%.1f' % a[5] if a[5] else '-',
        '%.3f' % re_ if re_ else '-'))
gp = math.exp(sum(lp) / len(lp))
ge = math.exp(sum(le) / len(le)) if le else float('nan')
print('geomean: prefill %.3f, encoder %.3f; worst prefill %.3f (%s)' % (gp, ge, worst[0], worst[1]))
ok = gp <= 1 - args.gain and worst[0] <= 1 + args.worst
print('rule (every turn <= %.2f, geomean <= %.2f): %s' % (1 + args.worst, 1 - args.gain, 'pass' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
