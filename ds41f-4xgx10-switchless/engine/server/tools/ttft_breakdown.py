#!/usr/bin/env python3
"""A small turn's time to first token by category and size (docs/engine.md, "A small turn's time to first
token"): ttft_roles.py's --json of two ranks' captures (roles_nsys0.jsonl, roles_nsys2.jsonl), the routes'
floors (ttft_routes.py --json: routes.jsonl) and the uninstrumented part's join (ttft_join.py --json:
join_base.jsonl), all in ANA_DIR. Prints ms per turn, the medians over a group's turns, on each rank, and the
floors: the touched experts' bytes for the MoE, the dense weights' bytes, and a boundary's activations (~102 KB
a row) and hc_fn's parts (2.9 MB), all at 235 GB/s.
  ttft_breakdown.py [ANA_DIR] [phase]"""
import collections
import json
import statistics as st
import sys

A = (sys.argv[1] if len(sys.argv) > 1 else '/data/dsgb10/bench/ttft/ana').rstrip('/') + '/'
PH = sys.argv[2] if len(sys.argv) > 2 else 'long'
BW = 235e9
CAT = {
    'moe': ['moe_up', 'moe_down', 'bucket'],
    'shared+gate': ['sh_up', 'sh_down', 'gate', 'router'],
    'attn_gemm': ['wqa_wkv', 'wq_b', 'wo_a', 'wo_b', 'compressor', 'ix_wqb', 'ix_wproj'],
    'sparse_attn': ['sparse_attn'],
    'index': ['ix_scan'],
    'mhc': ['bnd_attn', 'bnd_ffn'],
    'coll': ['ag_attn', 'ag_ffn', 'ag_topk', 'rep_in'],
    'engram': ['engram'],
    'head': ['head'],
}
ORDER = ['moe', 'shared+gate', 'attn_gemm', 'sparse_attn', 'index', 'mhc', 'coll', 'coll_tail', 'engram', 'small',
         'gaps', 'head']
rolecat = {r: c for c, rs in CAT.items() for r in rs}


def q_rows(M):
    return (((M + 3) // 4) + 7) & ~7


def cats(r):
    out = collections.Counter()
    for k, (own, gcs, gh, n) in r['main'].items():
        p, rl = k.split('|')
        c = rolecat.get(rl, 'small')
        out[c] += own
        out['coll_tail'] += gcs
        out['gaps'] += gh
        out['pass_' + p] += own + gcs + gh
    return out


def floors(size, routes):
    """ms per turn: the expert sweep (routes), the dense weights, the boundaries (activations + hc_fn parts)"""
    f = {}
    if size < 128:
        passes = [(size, 40)]
    else:  # encoder chunks of <= 8192 rows, then the replay
        passes = [(min(8192, size - c), 20) for c in range(0, size, 8192)] + [(128, 20)]
    f['moe'] = routes
    sh = (3 * 576 * 5120 + 384 * 5120 * 2) / BW * 1e3
    att = ((1280 + 512) * 5120 + 8192 * 1280 + 2048 * 4096 + 4 * 5120 * 2048) / BW * 1e3
    f['shared+gate'] = sum(L for M, L in passes) * sh
    f['attn_gemm'] = sum(L for M, L in passes) * att
    f['mhc'] = sum(2 * L * (q_rows(M) * (5120 * 2 * 2 + 20480 * 2 * 2) + 2.9e6) / BW * 1e3 for M, L in passes)
    return f


roles = {}
for rk in ('nsys0', 'nsys2'):
    g = collections.defaultdict(list)
    for line in open(A + 'roles_%s.jsonl' % rk):
        r = json.loads(line)
        g[tuple(r['group'])].append(r)
    roles[rk] = g
rt = collections.defaultdict(list)
for line in open(A + 'routes.jsonl'):
    r = json.loads(line)
    rt[(r['phase'], r['size'])].append(r)
jb = collections.defaultdict(list)
for line in open(A + 'join_base.jsonl'):
    r = json.loads(line)
    if 'gpu' in r:
        jb[(r['phase'], r['size'])].append(r)

sizes = sorted({sz for (ph, sz) in roles['nsys0'] if ph == PH})
print('phase %s: ms per turn, medians; floor = bytes at 235 GB/s (the MoE: the routes\' touched experts)' % PH)
hdr = '%-12s' % 'category' + ''.join('%17s' % ('+%d' % s) for s in sizes)
print(hdr)
print('%-12s' % '' + ''.join('%17s' % 'r0 / r2 / floor' for s in sizes))
rows = collections.OrderedDict()
for c in ORDER + ['pass_ex', 'pass_enc', 'pass_rep']:
    vals = []
    for s in sizes:
        v = []
        for rk in ('nsys0', 'nsys2'):
            rs = roles[rk].get((PH, s), [])
            v.append(st.median(cats(r)[c] for r in rs) if rs else float('nan'))
        rr = rt.get((PH, s), [])
        fr = st.median(r['floor_enc'] + r['floor_dec'] for r in rr) if rr else float('nan')
        fl = floors(s, fr).get(c)
        vals.append('%6.1f %5.1f %5s' % (v[0], v[1], '%.1f' % fl if fl is not None else ''))
    print('%-12s' % c + ''.join('%17s' % x for x in vals))
print()
for s in sizes:
    js = jb.get((PH, s), [])
    if js:
        med = lambda k: st.median(r[k] for r in js)
        print('+%-5d base: TTFT %.1f prefill %.1f ops %.1f layers %.1f replay %.1f head %.1f rest %.1f; '
              'client - prefill %.1f' % (s, med('ttft'), med('prefill'), med('ops'), med('layers'), med('replay'),
                                         med('head'), med('rest'), st.median(r['ttft'] - r['prefill'] for r in js)))
