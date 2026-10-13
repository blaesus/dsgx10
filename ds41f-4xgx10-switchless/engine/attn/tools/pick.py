#!/usr/bin/env python3
"""pick.py [--later] [--prefix P] DIR: the engine's DSGB10_MOE_TUNE from moe_sweep.sh's logs in DIR (bk_d3.log,
bk_d9.log: bench_kernels moereal's down kernel at rings 3 and 9; bk_ut.log: up's prefetch and the tile order;
bp_mhcr.log: bench_pattn's mHC boundary table; P before each name, e.g. w2_ for the 2026-10-12 window's), on
stdout; why, on stderr.
A knob leaves its default only if its bits were the same, it gains 2 % or more on the cases it governs,
and no case (or row count) gets more than 2 % slower. Defaults: down_p (dgrp 0), ring 3, no up prefetch,
descending tile order, the boundary at 32 rows per CTA without prefetch.
--later: also a later sweep's candidate, the boundary by the chunk with a prefetch distance per rows
per CTA (stderr always shows it). If it wins, the line gains hcpfr=P4,P16,P32 (by the chunk, those distances at 4,
16 and 32 rows per CTA), which an engine must read before it is given such a line; hc and hcpf stay this rule's."""
import itertools
import math
import sys

LATER = '--later' in sys.argv[1:]
ARGS = [a for a in sys.argv[1:] if a != '--later']
PRE = ''
if '--prefix' in ARGS:
    i = ARGS.index('--prefix')
    PRE = ARGS[i + 1]
    del ARGS[i:i + 2]
D = ARGS[0] if ARGS else '.'


def err(*a):
    print(*a, file=sys.stderr)


def moe_rows(name):
    """{(G, U, T): {case: mean MoE ms}}, and the variants whose bits differed"""
    acc, bad = {}, set()
    try:
        lines = open(D + '/' + PRE + name).read().splitlines()
    except OSError:
        err('no', name)
        return {}, bad
    for ln in lines:
        f = ln.replace('|', ' ').split()
        if len(f) < 17 or f[0] not in ('replay', 'encoder'):
            continue
        try:
            case, v, moe = (f[0], int(f[1]), 'fused' in ln), (int(f[2]), int(f[3]), int(f[4])), float(f[14])
            v = tuple(max(x, 0) for x in v)  # -1: not swept, the default (0 for G, U and T)
        except ValueError:
            continue
        acc.setdefault(v, {}).setdefault(case, []).append(moe)
        if 'DIFFER' in ln:
            bad.add(v)
    return {v: {c: sum(x) / len(x) for c, x in cs.items()} for v, cs in acc.items()}, bad


def best(table, bad, cand, base, cases):
    """the candidate with the least summed MoE over cases, if it is >= 2 % better and no case > 2 % worse"""
    if base not in table:
        return base, None
    b = table[base]
    pick, ps = base, sum(b[c] for c in cases)
    for v in cand:
        if v in bad or v not in table or any(c not in table[v] for c in cases):
            continue
        s = sum(table[v][c] for c in cases)
        if s <= 0.98 * sum(b[c] for c in cases) and s < ps and all(table[v][c] <= 1.02 * b[c] for c in cases):
            pick, ps = v, s
    return pick, ps / sum(b[c] for c in cases) if cases else None


# the down kernel: per ring, the best column blocks per CTA for the fused cases and for the others; ring 9
# only if its picks beat the default (ring 3, down_p) by the same rule
down, dflt = [], None
for ring, name in ((3, 'bk_d3.log'), (9, 'bk_d9.log')):
    t, bad = moe_rows(name)
    if not t:
        continue
    cases = set(c for cs in t.values() for c in cs)
    fz, nf = [c for c in cases if c[2]], [c for c in cases if not c[2]]
    vs = [v for v in t if v[1:] == (0, 0)]
    gf, rf = best(t, bad, vs, (0, 0, 0), fz)
    gn, rn = best(t, bad, vs, (0, 0, 0), nf)
    if gf not in t or gn not in t:
        err('down, ring %d: no down_p rows' % ring)
        continue
    per = {c: t[gf][c] for c in fz}
    per.update({c: t[gn][c] for c in nf})
    tot = sum(per.values())
    if ring == 3:
        dflt = {c: t[(0, 0, 0)][c] for c in cases}
    err('down, ring %d: fused G %d (%.3f of down_p), not fused G %d (%.3f), %.2f ms over the cases' %
        (ring, gf[0], rf or 1, gn[0], rn or 1, tot))
    down.append((tot, ring, gf[0], gn[0], per))
ring, dgrpf, dgrp = 3, 0, 0
for tot, r, gf0, gn0, per in sorted(down):
    if dflt is None or set(per) != set(dflt):
        continue
    d = sum(dflt.values())
    worst = max(per[c] / dflt[c] for c in per)
    ok = r == 3 or (tot <= 0.98 * d and worst <= 1.02)
    err('down ring %d G %d/%d: %.3f of the default in all, worst case %.3f: %s' %
        (r, gf0, gn0, tot / d, worst, 'taken' if ok else 'not taken'))
    if ok:
        ring, dgrpf, dgrp = r, gf0, gn0
        break

# up's prefetch and the tile order (with down_p)
t, bad = moe_rows('bk_ut.log')
upf = tord = 0
if t:
    cases = sorted(set(c for cs in t.values() for c in cs))
    u, ru = best(t, bad, [v for v in t if v[0] == 0 and v[2] == 0], (0, 0, 0), cases)
    o, ro = best(t, bad, [v for v in t if v[0] == 0 and v[1] == 0], (0, 0, 0), cases)
    upf, tord = u[1], o[2]
    err('up prefetch %d (%.3f of none), tile order %d (%.3f of descending)' % (upf, ru or 1, tord, ro or 1))

# the mHC boundary: 32 rows per CTA or by the chunk (<= 128: 4, <= 768: 16), with a prefetch distance
hc = hcpf = 0
hcpfr = None
cols = [(r, p) for r in (32, 16, 8, 4) for p in (0, 2, 4, 8)]
tab = {}
try:
    for ln in open(D + '/' + PRE + 'bp_mhcr.log').read().splitlines():
        f = ln.split()
        if len(f) == 17 and f[0].isdigit():
            tab[int(f[0])] = dict(zip(cols, map(float, f[1:])))
except (OSError, ValueError):
    err('no mhcr table')
if tab:
    heur = lambda n: 4 if n <= 128 else 16 if n <= 768 else 32  # noqa: E731
    bestv = (1.0, 0, 0)
    for mode, rfn in ((0, lambda n: 32), (-1, heur)):
        for p in (0, 2, 4, 8):
            rs = [tab[n][(rfn(n), p)] / tab[n][(32, 0)] for n in tab]
            g = math.exp(sum(map(math.log, rs)) / len(rs))
            err('boundary %s, prefetch %d: %s (geomean %.3f)' % ('by chunk' if mode else '32 rows', p,
                                                                 ' '.join('%.2f' % r for r in rs), g))
            if g <= 0.98 and max(rs) <= 1.02 and g < bestv[0]:
                bestv = (g, mode, p)
    _, hc, hcpf = bestv
    # a later sweep's candidate, not this one's: by the chunk, with a distance per rows per CTA (W2 part A: none
    # at 4 rows, the smallest chunks, and 2 at 16 and 32); the engine takes one distance today
    bl = None
    for pm in itertools.product((0, 2, 4, 8), repeat=3):
        pr = dict(zip((4, 16, 32), pm))
        rs = [tab[n][(heur(n), pr[heur(n)])] / tab[n][(32, 0)] for n in tab]
        g = math.exp(sum(map(math.log, rs)) / len(rs))
        if g <= 0.98 and max(rs) <= 1.02 and (bl is None or g < bl[0]):
            bl = (g, pm, rs)
    if bl:
        err('later: boundary by chunk, prefetch %s at 4/16/32 rows: %s (geomean %.3f)%s' %
            ('/'.join(map(str, bl[1])), ' '.join('%.2f' % r for r in bl[2]), bl[0], '' if LATER else ', not this sweep'))
        if LATER and bl[0] < bestv[0] and len(set(bl[1])) > 1:
            hcpfr = bl[1]
out = 'dgrp=%d dgrpf=%d dring=%d upf=%d tord=%d hc=%d hcpf=%d' % (dgrp, dgrpf, ring, upf, tord, hc, hcpf)
print(out + (' hcpfr=%s' % ','.join(map(str, hcpfr)) if hcpfr else ''))
