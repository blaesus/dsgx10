"""Spec steps' draft pass from an nsys capture of the spec loop (sqlite export).

usage: specprof.py file.sqlite [top]
  run4.sh --mode decode ... --big --spec --profile under WRAP="nsys profile -t cuda --cuda-graph-trace=node
  --capture-range=cudaProfilerApi ..." (docs/engine.md, Launch and stop)
Splits the kernel timeline into spec steps at k_spec_in and groups them by row count (the first hc_pre's
grid, a block per row). Per step: the wall from k_spec_in's start to k_spec_next's end, the part up to
k_sample's end (the rows through the model and their sampling), and the draft pass after it (accept,
draft, k_spec_next): its wall, idle gaps, and each kernel's exclusive time on the timeline (overlapped
work counts for the kernel that started first). Each group's first 3 steps are dropped.
"""
import sqlite3
import sys
from collections import defaultdict

db = sqlite3.connect(sys.argv[1])
top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
names = dict(db.execute('select id, value from StringIds'))
rows = db.execute('select start, end, demangledName, gridX, gridY, gridZ from CUPTI_ACTIVITY_KIND_KERNEL '
                  'order by start').fetchall()


def short(n):
    n = names[n]
    if n.startswith('void '):
        n = n[5:]
    n = n.split('(')[0]
    for p in ('dsk::<unnamed>::', 'attn::<unnamed>::', 'comm::<unnamed>::', 'model::ops::<unnamed>::',
              'model::detail::<unnamed>::', 'dspark::<unnamed>::', 'model::<unnamed>::', 'comm::', 'dsk::',
              'dspark::'):
        n = n.replace(p, '')
    return n


steps = []
for s, e, dn, gx, gy, gz in rows:
    n = short(dn)
    if n.startswith('k_spec_in'):
        steps.append([])
    if steps:
        steps[-1].append((s, e, n, gx * gy * gz))
groups = defaultdict(list)
for st in steps:
    pre = [g for s, e, n, g in st if n.startswith('hc_pre_kernel')]
    done = [i for i, k in enumerate(st) if k[2].startswith('k_spec_next')]
    samp = [i for i, k in enumerate(st) if k[2].startswith('k_sample')]
    if pre and done and samp:
        groups[pre[0]].append(st[:done[-1] + 1])
for R in sorted(groups):
    sts = groups[R][3:]
    if not sts:
        continue
    ns = len(sts)
    wall, main, dwall, gaps, nk = [], [], [], [], []
    per = defaultdict(lambda: [0.0, 0.0, 0])
    for st in sts:
        st = sorted(st)
        t0 = st[0][0]
        t_s = max(e for s, e, n, g in st if n.startswith('k_sample'))
        t1 = st[-1][1]
        wall.append((t1 - t0) / 1e3)
        main.append((t_s - t0) / 1e3)
        dwall.append((t1 - t_s) / 1e3)
        end, gap, k = t_s, 0.0, 0
        for s, e, n, g in st:
            if s < t_s:
                continue
            k += 1
            if s > end:
                gap += (s - end) / 1e3
            x = max(0, e - max(s, end)) / 1e3
            end = max(end, e)
            per[n][0] += x
            per[n][1] += (e - s) / 1e3
            per[n][2] += 1
        gaps.append(gap)
        nk.append(k)

    def med(v):
        return sorted(v)[len(v) // 2]

    print('R = %d: %d steps; medians: wall %.3f ms, rows and sampling %.3f ms, draft pass %.3f ms '
          '(%d kernels, idle gaps %.3f ms)' % (R, ns, med(wall) / 1e3, med(main) / 1e3, med(dwall) / 1e3, med(nk),
                                             med(gaps) / 1e3))
    print('  draft pass kernels by exclusive time (ms/step excl, busy; per step; avg us):')
    for n, (x, b, c) in sorted(per.items(), key=lambda kv: -kv[1][0])[:top]:
        print('    %7.3f %7.3f %4d %8.1f  %s' % (x / ns / 1e3, b / ns / 1e3, c // ns, b / c, n[:70]))
