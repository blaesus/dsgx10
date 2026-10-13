"""Per-step kernel breakdown of an nsys capture of graph-replayed steps (sqlite export).

usage: kprof.py file.sqlite [top]
Splits the kernel timeline into steps at k_step_in, groups them by row count (the grid of the step's
first hc_pre, a block per row), drops each group's first 3 steps, and reports per step: wall (first kernel
start to last end), each class's exclusive time on the timeline (time it alone advanced the step's
end: overlapped work counts for the kernel that started first), idle gaps, busy (summed) time, and
the top kernels.
"""
import sqlite3
import sys
from collections import defaultdict

db = sqlite3.connect(sys.argv[1])
top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
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


def klass(n):
    if any(s in n for s in ('allreduce', 'allgather', 'argmax_kernel', 'k_wait_seq', 'reduce_scatter', 'k_sample')):
        return 'collective'
    if any(s in n for s in ('gemv', 'moe')):
        return 'weights'
    if any(s in n for s in ('attn_kernel', 'attn_combine', 'score', 'select', 'emit', 'carry', 'kv_write', 'q_prep',
                            'merge', 'q_norm', 'rope', 'index_fused', 'index_dense', 'decode_prep')):
        return 'attention'
    if n.startswith('hc_') or 'k_mix_ranks' in n:
        return 'mhc'
    return 'other'


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
    if not sts:
        continue
    ns = len(sts)
    wall = sorted((st[-1][1] - st[0][0]) / 1e3 for st in sts)
    excl = defaultdict(float)
    busy = defaultdict(float)
    per = defaultdict(lambda: [0.0, 0.0, 0, 0])
    gaps = 0.0
    for st in sts:
        st = sorted(st)
        end = st[0][0]
        for s, e, n, g in st:
            if s > end:
                gaps += (s - end) / 1e3
            x = max(0, e - max(s, end)) / 1e3
            end = max(end, e)
            c = klass(n)
            excl[c] += x
            busy[c] += (e - s) / 1e3
            per[n][0] += x
            per[n][1] += (e - s) / 1e3
            per[n][2] += 1
            per[n][3] = g
    print('R = %d: %d steps, %d kernels per step; wall median %.3f ms, min %.3f ms' %
          (R, ns, len(sts[0]), wall[ns // 2] / 1e3, wall[0] / 1e3))
    print('  %-11s %9s %9s' % ('class', 'excl ms', 'busy ms'))
    for c in ('weights', 'attention', 'mhc', 'collective', 'other'):
        print('  %-11s %9.3f %9.3f' % (c, excl[c] / ns / 1e3, busy[c] / ns / 1e3))
    print('  %-11s %9.3f' % ('idle gaps', gaps / ns / 1e3))
    print('  top kernels by exclusive time (ms/step excl, busy; per step; avg us; grid):')
    for n, (x, b, c, g) in sorted(per.items(), key=lambda kv: -kv[1][0])[:top]:
        print('    %7.3f %7.3f %4d %8.1f %6d  %s [%s]' % (x / ns / 1e3, b / ns / 1e3, c // ns, b / c, g, n[:60], klass(n)))
