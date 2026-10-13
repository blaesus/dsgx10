#!/usr/bin/env python3
"""abcmp.py <dir> [jobs...]: an engine A/B's table from abserve.sh's <dir>/ab_<job>_<m|b>_<spec|k1>.json (m: the
baseline, b: the candidate) and the servers' logs. Per job and draft setting: every request's tokens, baseline
against candidate; per case the times and the change (candidate against baseline, in time: negative is faster),
then the mean over the jobs; the geometric mean over the bar's cases; per-rank memory.
  python3 engine/kernels/tests/abcmp.py /path/to/results ab1 ab2"""
import json
import math
import os
import re
import statistics
import sys

D = sys.argv[1]
JOBS = sys.argv[2:] or ['ab1', 'ab2']
CFGS = ('k1', 'spec')
TYPES = ('prose', 'code', 'agentic', 'json')
CTXS = (16384, 25984)


def load(job, b, cfg):
    fn = '%s/ab_%s_%s_%s.json' % (D, job, b, cfg)
    if not os.path.exists(fn):
        return None
    return {r['case']: r for r in json.load(open(fn))['records']}


def rate(rs):  # decode tok/s over requests (bench.py's: tokens after the first over the decode time)
    return sum(r['completion'] - 1 for r in rs) / sum(r['decode_ms'] / 1e3 for r in rs)


def cases(R):
    """case -> (time in ms, shown value, unit); the time decides the change."""
    out = {}
    for t in TYPES:
        rs = [R['bench/%s/%d' % (t, k)] for k in range(2)]
        out['bench %s' % t] = (1e3 / rate(rs), rate(rs), 'tok/s')
    ed = [r for c, r in R.items() if c.startswith('edit/')]
    tot = sum(r['prefill_ms'] + r['decode_ms'] for r in ed)
    out['edit, 54 turns'] = (tot, tot / 1e3, 's')
    out['edit, decode'] = (1e3 / rate(ed), rate(ed), 'tok/s')
    out['edit, prefill'] = (sum(r['prefill_ms'] for r in ed), sum(r['prefill_ms'] for r in ed), 'ms')
    for c in CTXS:
        for k in ('cold', '+200', '+2000'):
            rs = [R['turn/%d/%s/%d' % (c, k, i)] for i in range(3)]
            ttft = statistics.median(r['prefill_ms'] for r in rs)
            turn = statistics.median(r['prefill_ms'] + r['decode_ms'] for r in rs)
            out['%dk %s TTFT' % (round(c / 1000), k)] = (ttft, ttft, 'ms')
            if k != 'cold':
                out['%dk %s turn' % (round(c / 1000), k)] = (turn, turn, 'ms')
    return out


# the bar's cases: what a user waits for (the bench's decode, the edit bench's turns, the +200 / +2000 turns)
BAR = ['bench %s' % t for t in TYPES] + ['edit, 54 turns'] + ['%dk %s turn' % (round(c / 1000), k) for c in CTXS
                                                              for k in ('+200', '+2000')]


def mem(job, b, cfg):
    fn = '%s/ab_%s_%s_%s_server.log' % (D, job, b, cfg)
    if not os.path.exists(fn):
        return None, '', ''
    t = open(fn, errors='replace').read()
    m = re.search(r'free device memory after the prefill buffers, ranks 0-3: ([0-9. ]+) GiB', t)
    c = re.search(r"moe: the experts' scale codes[^\n]*", t)
    h = re.search(r'rank 0: the head coded[^\n]*', t)
    return ([float(x) for x in m.group(1).split()] if m else None), (c.group(0) if c else ''), (h.group(0) if h else '')


allch = {}
for job in JOBS:
    for cfg in CFGS:
        M, B = load(job, 'm', cfg), load(job, 'b', cfg)
        if not M or not B:
            print('%s %s: missing (%s)' % (job, cfg, 'baseline' if not M else 'candidate'))
            continue
        same = [c for c in M if c in B and M[c]['hash'] == B[c]['hash']]
        diff = [c for c in M if c in B and M[c]['hash'] != B[c]['hash']]
        tm = sum(1 for r in M.values() if r.get('trace_equal'))
        tb = sum(1 for r in B.values() if r.get('trace_equal'))
        print('%s %s: %d requests, tokens equal in %d, DIFFER in %d %s; edit turns equal to the trace: baseline %d, '
              'candidate %d' % (job, cfg, len(M), len(same), len(diff), diff[:4], tm, tb))
        cm, cb = cases(M), cases(B)
        for k in cm:
            allch.setdefault((cfg, k), {})[job] = (cm[k], cb[k])
for job in JOBS:
    for b in ('m', 'b'):
        for cfg in CFGS:
            f, c, h = mem(job, b, cfg)
            if f:
                print('memory %s %s %s: free after the prefill buffers %s GiB  %s  %s' % (job, b, cfg, f, c, h))

print()
print('| case | %s | change, mean |' % ' | '.join('%s baseline | %s candidate | %s change' % (j, j, j) for j in JOBS))
print('|---|%s---|' % ('---|---|---|' * len(JOBS)))
gm = {cfg: {j: [] for j in JOBS} for cfg in CFGS}
for (cfg, k), byjob in allch.items():
    row = []
    chs = []
    for j in JOBS:
        if j not in byjob:
            row.append(' - | - | - ')
            continue
        (tm, vm, u), (tb, vb, _) = byjob[j]
        ch = 100 * (tb / tm - 1)
        chs.append(ch)
        row.append('%.1f | %.1f | %+.2f %%' % (vm, vb, ch))
        if k in BAR:
            gm[cfg][j].append(tb / tm)
    unit = next(iter(byjob.values()))[0][2]
    print('| %s %s (%s) | %s | %+.2f %% |' % (cfg, k, unit, ' | '.join(row), sum(chs) / max(1, len(chs))))
print()
print("time change, candidate against baseline (negative: faster); the bar's cases: %s" % ', '.join(BAR))
for cfg in CFGS:
    for j in JOBS:
        v = gm[cfg][j]
        if v:
            print('  %s %s: geometric mean %+.2f %% over %d cases; worst %+.2f %%' % (
                cfg, j, 100 * (math.exp(sum(math.log(x) for x in v) / len(v)) - 1), len(v), 100 * (max(v) - 1)))
allv = [x for cfg in CFGS for j in JOBS for x in gm[cfg][j]]
if allv:
    print('  all: geometric mean %+.2f %% over %d; worst case %+.2f %%' % (
        100 * (math.exp(sum(math.log(x) for x in allv) / len(allv)) - 1), len(allv), 100 * (max(allv) - 1)))
