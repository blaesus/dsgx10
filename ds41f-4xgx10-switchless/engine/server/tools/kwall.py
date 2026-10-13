"""Per-step wall and collective time, and per-collective duration percentiles (nsys sqlite).

usage: kwall.py file.sqlite
For each row count: every step's wall (ms) in order; then for each collective kernel the 10th
percentile, median and 90th percentile of its duration (us). The 10th percentile approximates the
collective's own cost when no rank is late; the rest of its time is waiting for the slowest rank.
"""
import sqlite3
import sys
from collections import defaultdict

db = sqlite3.connect(sys.argv[1])
names = dict(db.execute('select id, value from StringIds'))
rows = db.execute('select start, end, demangledName, gridX, gridY, gridZ from CUPTI_ACTIVITY_KIND_KERNEL '
                  'order by start').fetchall()


def short(n):
    n = names[n]
    if n.startswith('void '):
        n = n[5:]
    n = n.split('(')[0]
    for p in ('dsk::<unnamed>::', 'attn::<unnamed>::', 'comm::<unnamed>::', 'model::ops::<unnamed>::', 'comm::',
              'dsk::'):
        n = n.replace(p, '')
    return n


COLL = ('allreduce', 'allgather', 'argmax_kernel', 'k_wait_seq', 'k_sample')
SIDE = ('k_mix_allreduce',)  # collectives on the side stream, off the step's critical path
steps = []
for s, e, dn, gx, gy, gz in rows:
    n = short(dn)
    if n.startswith('k_step_in'):
        steps.append([])
    if steps:
        steps[-1].append((s, e, n, gx * gy * gz))
groups = defaultdict(list)
for st in steps:  # rows: the first layer's hc_pre has a block per row (the all-reduce had 5 before it spread)
    pre = [g for s, e, n, g in st if n.startswith('hc_pre_kernel')]
    ar = [g for s, e, n, g in st if n.startswith('k_allreduce_hc')]
    if pre:
        groups[pre[0]].append(st)
    elif ar:
        groups[ar[0] // 5].append(st)
for R in sorted(groups):
    sts = groups[R][3:]
    walls = [(st[-1][1] - st[0][0]) / 1e6 for st in sts]
    print('R = %d walls (ms): %s' % (R, ' '.join('%.1f' % w for w in walls)))
    dur = defaultdict(list)
    for st in sts:
        for s, e, n, g in st:
            if any(c in n for c in COLL):
                dur[n].append((e - s) / 1e3)
    tot10 = side10 = 0.0
    for n, d in sorted(dur.items()):
        d.sort()
        k = len(d)
        per_step = k / len(sts)
        if any(c in n for c in SIDE):
            side10 += d[k // 10] * per_step
        else:
            tot10 += d[k // 10] * per_step
        print('  %-40s %5.1f/step  p10 %7.1f  med %7.1f  p90 %7.1f us%s' %
              (n[:40], per_step, d[k // 10], d[k // 2], d[k * 9 // 10], ' (side stream)' if any(c in n for c in SIDE) else ''))
    print('  collectives at their p10 durations: %.3f ms per step on the main stream, %.3f on the side stream' %
          (tot10 / 1e3, side10 / 1e3))
