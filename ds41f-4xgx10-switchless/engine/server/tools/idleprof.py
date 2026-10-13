#!/usr/bin/env python3
"""GPU idle per step of a decode loop, from an nsys capture (sqlite export; graph nodes traced).

usage: idleprof.py file.sqlite [skip]
  run4.sh --mode decode ... --big [--spec] --profile under WRAP="nsys profile -t cuda --cuda-graph-trace=node
  --capture-range=cudaProfilerApi ..." (docs/engine.md, Launch and stop), then nsys export -t sqlite
  (docs/engine.md, "The end of a turn and the host's share of a step").
Splits the GPU timeline into steps at k_step_in / k_spec_in, takes each step's rows from its first
hc_pre's grid (a block per row), drops the first `skip` steps (default 3), and reports per step:
  span     first kernel's start to last kernel's end (every stream)
  gap      the next step's first start minus this step's last end: the GPU ran nothing
  inside   the span minus the union of the step's kernels and copies: idle inside the step
  wait     the Engram waits (k_wait_seq) beyond their own cost (each one's p10)
  lead     the next step's first kernel start minus the end of the host call that launched it
Host-made idle is the gap when the next step's launch came after this step ended, plus the waits.
"""
import sqlite3
import sys
from collections import defaultdict

db = sqlite3.connect(sys.argv[1])
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 3
names = dict(db.execute('select id, value from StringIds'))
tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
ker = db.execute('select start, end, demangledName, gridX * gridY * gridZ, correlationId, streamId '
                 'from CUPTI_ACTIVITY_KIND_KERNEL order by start').fetchall()
other = []
for t in ('CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
    if t in tables:
        other += db.execute(f'select start, end, correlationId, streamId from {t}').fetchall()
api = {}
if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables:
    for s, e, c, n in db.execute('select start, end, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        api[c] = (s, e, names.get(n, str(n)))


def short(n):
    n = names[n]
    if n.startswith('void '):
        n = n[5:]
    n = n.split('(')[0]
    return n.split('::')[-1]


steps = []
for s, e, dn, g, c, st in ker:
    n = short(dn)
    if n.startswith('k_step_in') or n.startswith('k_spec_in'):
        steps.append({'k': [], 'c': c})
    if steps:
        steps[-1]['k'].append((s, e, n, g, st))
# copies and memsets go to the step whose span holds their start
starts = [st['k'][0][0] for st in steps]
import bisect  # noqa: E402

for s, e, c, stm in other:
    i = bisect.bisect_right(starts, s) - 1
    if i >= 0:
        steps[i]['k'].append((s, e, 'copy', 0, stm))
waits = sorted(e - s for st in steps for s, e, n, g, _ in st['k'] if n.startswith('k_wait_seq'))
wait_floor = waits[len(waits) // 10] if waits else 0
rows_of = []
for st in steps:
    pre = [g for s, e, n, g, _ in st['k'] if n.startswith('hc_pre_kernel')]
    rows_of.append(pre[0] if pre else 0)
out = defaultdict(list)
for i in range(skip, len(steps) - 1):
    k = sorted(steps[i]['k'])
    t0 = k[0][0]
    t1 = max(e for s, e, *_ in k)
    union, cur_s, cur_e = 0, None, None
    for s, e, *_ in k:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                union += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    union += cur_e - cur_s
    nxt = min(s for s, *_ in steps[i + 1]['k'])
    gap = nxt - t1
    w = sum(max(0, (e - s) - wait_floor) for s, e, n, *_ in k if n.startswith('k_wait_seq'))
    a = api.get(steps[i + 1]['c'])
    lead = nxt - a[1] if a else None
    late = a is not None and a[1] > t1  # the next step was launched after this one ended
    out[rows_of[i]].append(dict(span=t1 - t0, gap=gap, inside=(t1 - t0) - union, wait=w, lead=lead, late=late,
                                host=(gap if late else 0) + w))


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float('nan')


def mean(v):
    return sum(v) / len(v) if v else float('nan')


print('%s: %d steps (first %d dropped); k_wait_seq floor (p10) %.1f us' % (sys.argv[1], len(steps), skip,
                                                                          wait_floor / 1e3))
print('rows  steps   span ms   gap us (mean, med, max)   inside us (mean)   wait us (mean)   launched late   '
      'lead ms (med)   host-made idle us (mean, % of span)')
allv = [x for v in out.values() for x in v]
for R in sorted(out) + ['all']:
    v = allv if R == 'all' else out[R]
    if not v:
        continue
    span = mean([x['span'] for x in v])
    host = mean([x['host'] for x in v])
    leads = [x['lead'] for x in v if x['lead'] is not None]
    print('%4s %6d %9.3f   %7.1f %7.1f %8.1f   %10.1f   %12.1f   %6d of %-4d   %9.3f   %10.1f  %5.2f %%' %
          (R, len(v), med([x['span'] for x in v]) / 1e6, mean([x['gap'] for x in v]) / 1e3,
           med([x['gap'] for x in v]) / 1e3, max(x['gap'] for x in v) / 1e3, mean([x['inside'] for x in v]) / 1e3,
           mean([x['wait'] for x in v]) / 1e3, sum(x['late'] for x in v), len(v), med(leads) / 1e6 if leads else
           float('nan'), host / 1e3, 100 * host / span))
