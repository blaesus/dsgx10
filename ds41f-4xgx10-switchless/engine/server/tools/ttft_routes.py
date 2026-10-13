#!/usr/bin/env python3
"""The experts each turn's layers touched (ttft_part.sh ROUTES: rank 0's dump of every pass's routes,
DIR/a<pos>.m<M>.L<LL>.i32, int32 [M][6]) and the expert sweep's floor: the touched experts' bytes per rank
(w1, w3, w2 of 576 columns: fp4, and the scales' 1-bit code since 2812072: 4.46 MB an expert a layer, 4.70
with the raw ue8m0 scales) at 235 GB/s.
  ttft_routes.py DIR TURNS_LOG [--json OUT]
A turn of n new rows at position q (TURNS_LOG's rows: prompt, cached) is, with CED (n >= 128), layers 0-19
over [q, q + n) and the replay, layers 20-39 over the last 128 positions; else every layer over [q, q + n).
Prints per phase and size the medians of the touched experts a layer (layers 0-19, 20-39) and the floor in
ms (layers 0-19, 20-39, the turn)."""
import array
import json
import os
import statistics as st
import sys

# fp4 weights, then the code blocks of the scales (dsk.h ScaleCode: 84 words a 16-row tile of w1 and w3, 12 of w2)
EXP_B = 3 * 576 * 5120 // 2 + 2 * 36 * 84 * 4 + 320 * 12 * 4
BW = 235e9
a = sys.argv[1:]
if len(a) < 2:
    sys.exit(__doc__)
D, TL = a[0], a[1]
jout = a[a.index('--json') + 1] if '--json' in a else None


def distinct(path):
    x = array.array('i')
    with open(path, 'rb') as f:
        x.frombytes(f.read())
    return len(set(x))


rows = []
for line in open(TL):
    f = line.rstrip('\n').split('\t')
    if f[0] == 'row':
        rows.append(dict(rid=int(f[1]), phase=f[2], size=int(f[3]), prompt=int(f[4]), cached=int(f[5])))
out = []
for r in rows:
    q, n = r['cached'], r['prompt'] - r['cached']
    ced = n >= 128
    d = {}  # per layer: the touched experts, summed over the turn's chunks of up to 8192 rows
    miss = 0
    for L in range(40):
        passes = [(r['prompt'] - 128, 128)] if ced and L >= 20 else \
            [(q + c, min(8192, n - c)) for c in range(0, n, 8192)]
        for a0, m in passes:
            p = os.path.join(D, 'a%d.m%d.L%02d.i32' % (a0, m, L))
            if os.path.exists(p):
                d[L] = d.get(L, 0) + distinct(p)
            else:
                miss += 1
    if miss:
        r['missing'] = miss
    r['experts'] = d
    r['enc'] = sum(d.get(L, 0) for L in range(20))
    r['dec'] = sum(d.get(L, 0) for L in range(20, 40))
    r['floor_enc'] = r['enc'] * EXP_B / BW * 1e3
    r['floor_dec'] = r['dec'] * EXP_B / BW * 1e3
    out.append(r)
groups = {}
for r in out:
    groups.setdefault((r['phase'], r['size']), []).append(r)
print('%-9s %6s %3s %8s %8s | %9s %9s %9s' % ('phase', 'size', 'n', 'exp 0-19', 'exp20-39', 'floor0-19', 'floor20-39',
                                               'floor'))
for (ph, sz), rs in groups.items():
    med = lambda k: st.median(r[k] for r in rs)
    miss = sum(r.get('missing', 0) for r in rs)
    print('%-9s %6d %3d %8.1f %8.1f | %9.1f %9.1f %9.1f%s' % (ph, sz, len(rs), med('enc') / 20, med('dec') / 20,
                                                            med('floor_enc'), med('floor_dec'),
                                                            st.median(r['floor_enc'] + r['floor_dec'] for r in rs),
                                                            '  (%d layer files missing)' % miss if miss else ''))
if jout:
    with open(jout, 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
