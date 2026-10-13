#!/usr/bin/env python3
"""Lossless weight compression, Part 1: order-0 entropy and GPU-decodable schemes per weight class,
from wcomp's histograms (sample.jsonl) and the rank file's TOC (wcomp --list). Stdlib only.

Bits are per code (FP8 byte, FP4 value, bf16 / f32 word) unless said otherwise."""
import heapq
import json
import math
import re
import sys
from collections import defaultdict

SRC = sys.argv[1] if len(sys.argv) > 1 else 'sample.jsonl'


def H(c):
    t = sum(c)
    return -sum(x / t * math.log2(x / t) for x in c if x) if t else 0.0


def huffman(c):
    """optimal prefix-code lengths for counts c (0 for absent symbols)"""
    syms = [i for i, w in enumerate(c) if w]
    L = [0] * len(c)
    if len(syms) == 1:
        L[syms[0]] = 1
        return L
    heap = [(c[i], k, [i]) for k, i in enumerate(syms)]
    heapq.heapify(heap)
    k = len(heap)
    while len(heap) > 1:
        w1, _, a = heapq.heappop(heap)
        w2, _, b = heapq.heappop(heap)
        for s in a:
            L[s] += 1
        for s in b:
            L[s] += 1
        heapq.heappush(heap, (w1 + w2, k, a + b))
        k += 1
    return L


def limited(c, lmax):
    """optimal code lengths of at most lmax bits (package-merge)"""
    L = huffman(c)
    if max(L) <= lmax:
        return L
    syms = sorted((w, i) for i, w in enumerate(c) if w)
    m = len(syms)
    assert (1 << lmax) >= m, (lmax, m)
    leaves = []
    for k, (w, _) in enumerate(syms):
        v = [0] * m
        v[k] = 1
        leaves.append((w, v))
    cur = leaves
    for _ in range(lmax - 1):
        packs = [(cur[j][0] + cur[j + 1][0], [x + y for x, y in zip(cur[j][1], cur[j + 1][1])])
                 for j in range(0, len(cur) - 1, 2)]
        cur = sorted(leaves + packs, key=lambda t: t[0])
    cnt = [0] * m
    for _, v in cur[:2 * m - 2]:
        for k in range(m):
            cnt[k] += v[k]
    L = [0] * len(c)
    for k, (_, i) in enumerate(syms):
        L[i] = cnt[k]
    return L


def cost(c, L):
    return sum(w * l for w, l in zip(c, L))


def bits(c, lmax=None):
    """mean code length of c's own Huffman code (lmax: length-limited)"""
    L = limited(c, lmax) if lmax else huffman(c)
    return cost(c, L) / sum(c)


def cond(rows):
    """H(X | context) from count rows, one per context"""
    tot = sum(sum(r) for r in rows)
    return sum(sum(r) / tot * H(r) for r in rows) if tot else 0.0


def sparse_rows(d, n=256):
    """{prev * n + x: count} -> list of count rows by prev"""
    rows = defaultdict(lambda: [0] * n)
    for k, v in d.items():
        k = int(k)
        rows[k // n][k % n] += v
    return list(rows.values())


def kind(name):
    """the weight class of a tensor name"""
    if name.startswith('mtp.'):
        if '.ffn.experts.' in name:
            return 'mtp_experts'
        if 'markov_head' in name or 'confidence' in name:
            return 'mtp_heads'
        return 'mtp_dense'
    if name == 'head.weight':
        return 'head'
    if name == 'embed.weight':
        return 'embed'
    if '.ffn.experts.' in name:
        return 'experts_' + name.split('.')[-2]  # w1 / w2 / w3
    if '.ffn.shared_experts.' in name:
        return 'shared'
    if '.engram.' in name:
        return 'engram'
    if '.indexer.wq_b' in name:
        return 'indexer_wq_b'
    if '.indexer.' in name:
        return 'indexer_bf16'
    if '.compressor.' in name:
        return 'compressor'
    if '.ffn.gate.' in name:
        return 'router'
    if '.attn.wo_a' in name:
        return 'wo_a'
    if '.attn.' in name:
        return 'attn_' + name.split('.')[-2]
    if '.hc_' in name:
        return 'mhc'
    return 'other'


# ---------------------------------------------------------------- per tensor
def fp8_stats(d):
    h = d['h']
    n = sum(h)
    ce = [0] * 16  # exponent field, DFloat11-style
    cm = [0] * 128  # magnitude (sign raw)
    for b, w in enumerate(h):
        ce[(b >> 3) & 15] += w
        cm[b & 127] += w
    hs = [r for r in d['hs'].values()]
    return dict(n=n, H0=H(h), Hs=cond(hs), Hprev=cond(sparse_rows(d['pair'])),
                Hexp=H(ce), huff=bits(h), h12=bits(h, 12), h10=bits(h, 10), h8=bits(h, 8),
                df11=bits(ce) + 4, df11_8=bits(ce, 8) + 4, mag12=bits(cm, 12) + 1,
                per_scale12=sum(cost(r, limited(r, 12)) for r in hs) / n,
                nscale=len(hs), hist=h, ce=ce)


def fp4_stats(d):
    hn, hb = d['hn'], d['hb']
    n = sum(hn)  # values
    ce = [0] * 4  # e2m1: s e e m
    for v, w in enumerate(hn):
        ce[(v >> 1) & 3] += w
    hns = list(d['hns'].values())
    sh = d['sh']
    ns = sum(sh)
    srows = sparse_rows(d['spair'])
    hsc = cond(srows)
    # neighbouring scales of a 4-byte group as one symbol (pairs from j-1 -> j, every j pooled)
    sp = sorted((int(k), v) for k, v in d['spair'].items())
    spc = [v for _, v in sp]
    return dict(n=n, H0=H(hn), Hs=cond(hns), Hpair=H(hb) / 2, huff=bits(hn), hpair12=bits(hb, 12) / 2,
                hpair8=bits(hb, 8) / 2, hpair=bits(hb) / 2, df11=bits(ce) + 2,
                per_scale=sum(cost(r, huffman(r)) for r in hns) / n,
                sH0=H(sh), sHc=hsc, s_est=(H(sh) + 3 * hsc) / 4, shuff8=bits(sh, 8),
                spair12=bits(spc, 12) / 2, ns=ns, hn=hn, hb=hb, sh=sh)


def bf16_stats(d):
    h16 = {int(k): v for k, v in d['h16'].items()}
    n = sum(h16.values())
    ce, chi = [0] * 256, [0] * 256
    for w, c in h16.items():
        ce[(w >> 7) & 255] += c
        chi[w >> 8] += c
    Hw = H(list(h16.values()))
    return dict(n=n, Hw=Hw, Hexp=H(ce), Hsm_e=Hw - H(ce), Hhi=H(chi),
                df11=bits(ce) + 8, df11_12=bits(ce, 12) + 8, df11_8=bits(ce, 8) + 8,
                hi12=bits(chi, 12) + 8, ce=ce, chi=chi)


def f32_stats(d):
    h = d['hexp']
    return dict(n=sum(h), Hexp=H(h), df11=bits(h) + 24, df11_12=bits(h, 12) + 24, ce=h)


rows = [json.loads(l) for l in open(SRC)]
stats = []
for d in rows:
    f = dict(fp8=fp8_stats, fp4=fp4_stats, bf16=bf16_stats, f32=f32_stats)[d['class']]
    s = f(d)
    s.update(name=d['name'], cls=d['class'], kind=kind(d['name']), bytes=d['bytes'], read=d['read'])
    stats.append(s)
mixed = defaultdict(set)
for s in stats:
    mixed[s['kind']].add(s['cls'])
for s in stats:
    if len(mixed[s['kind']]) > 1:
        s['kind'] += '_' + s['cls']
json.dump([{k: v for k, v in s.items() if not isinstance(v, (list, dict))} for s in stats],
          open('stats.json', 'w'), indent=0)


# ---------------------------------------------------------------- per kind
def wavg(ss, key, w='n'):
    t = sum(s[w] for s in ss)
    return sum(s[w] * s[key] for s in ss) / t


def pooled(ss, key):
    acc = None
    for s in ss:
        v = s[key]
        acc = list(v) if acc is None else [a + b for a, b in zip(acc, v)]
    return acc


by_kind = defaultdict(list)
for s in stats:
    by_kind[s['kind']].append(s)

summary = {}
print('FP8 e4m3 (bits per byte; per-tensor tables unless "pl12", one <=12-bit table pooled over the kind)')
print(f'{"kind":14} {"n":>3} {"GB":>6} {"H0":>5} {"pool":>5} {"H|sc":>5} {"H|prv":>5} {"Hexp":>5} '
      f'{"huff":>5} {"<=12":>5} {"<=10":>5} {"<=8":>5} {"df11":>5} {"mag12":>5} {"sc12":>5} {"pl12":>5}')
for k, ss in sorted(by_kind.items()):
    if ss[0]['cls'] != 'fp8':
        continue
    ph = pooled(ss, 'hist')
    Lp = limited(ph, 12)
    pl12 = sum(cost(s['hist'], Lp) for s in ss) / sum(s['n'] for s in ss)
    r = {key: wavg(ss, key) for key in ('H0', 'Hs', 'Hprev', 'Hexp', 'huff', 'h12', 'h10', 'h8', 'df11',
                                        'mag12', 'per_scale12')}
    r.update(pool=H(ph), pl12=pl12, n=len(ss), gb=sum(s['n'] for s in ss) / 1e9, cls='fp8')
    summary[k] = r
    print(f'{k:14} {len(ss):3d} {r["gb"]:6.3f} {r["H0"]:5.3f} {r["pool"]:5.3f} {r["Hs"]:5.3f} {r["Hprev"]:5.3f} '
          f'{r["Hexp"]:5.3f} {r["huff"]:5.3f} {r["h12"]:5.3f} {r["h10"]:5.3f} {r["h8"]:5.3f} {r["df11"]:5.3f} '
          f'{r["mag12"]:5.3f} {r["per_scale12"]:5.3f} {pl12:5.3f}')

print('\nMXFP4 experts (bits per value; scales: bits per ue8m0 byte)')
print(f'{"kind":14} {"n":>3} {"GB":>6} {"H0":>5} {"H|sc":>5} {"Hpair":>5} {"huff":>5} {"pr12":>5} {"pr8":>5} '
      f'{"df11":>5} {"sc_tb":>5} | {"sH0":>5} {"sH|p":>5} {"s_est":>5} {"sh8":>5} {"sp12":>5}')
for k, ss in sorted(by_kind.items()):
    if ss[0]['cls'] != 'fp4':
        continue
    r = {key: wavg(ss, key) for key in ('H0', 'Hs', 'Hpair', 'huff', 'hpair12', 'hpair8', 'df11', 'per_scale')}
    for key in ('sH0', 'sHc', 's_est', 'shuff8', 'spair12'):
        r[key] = wavg(ss, key, 'ns')
    r.update(n=len(ss), gb=sum(s['read'] for s in ss) / 1e9, cls='fp4')
    summary[k] = r
    print(f'{k:14} {len(ss):3d} {r["gb"]:6.3f} {r["H0"]:5.3f} {r["Hs"]:5.3f} {r["Hpair"]:5.3f} {r["huff"]:5.3f} '
          f'{r["hpair12"]:5.3f} {r["hpair8"]:5.3f} {r["df11"]:5.3f} {r["per_scale"]:5.3f} | {r["sH0"]:5.3f} '
          f'{r["sHc"]:5.3f} {r["s_est"]:5.3f} {r["shuff8"]:5.3f} {r["spair12"]:5.3f}')

print('\nbf16 (bits per weight; pl12: exponent table pooled over the kind)')
print(f'{"kind":14} {"n":>3} {"GB":>6} {"Hword":>6} {"Hexp":>5} {"H(sm|e)":>7} {"Hhi":>5} {"df11":>6} '
      f'{"df<=12":>6} {"df<=8":>6} {"hi12":>6} {"pl12":>6}')
for k, ss in sorted(by_kind.items()):
    if ss[0]['cls'] != 'bf16':
        continue
    pe = pooled(ss, 'ce')
    Lp = limited(pe, 12)
    pl12 = sum(cost(s['ce'], Lp) for s in ss) / sum(s['n'] for s in ss) + 8
    r = {key: wavg(ss, key) for key in ('Hw', 'Hexp', 'Hsm_e', 'Hhi', 'df11', 'df11_12', 'df11_8', 'hi12')}
    r.update(n=len(ss), gb=sum(s['read'] for s in ss) / 1e9, cls='bf16', pl12=pl12)
    summary[k] = r
    print(f'{k:14} {len(ss):3d} {r["gb"]:6.3f} {r["Hw"]:6.3f} {r["Hexp"]:5.3f} {r["Hsm_e"]:7.3f} {r["Hhi"]:5.3f} '
          f'{r["df11"]:6.3f} {r["df11_12"]:6.3f} {r["df11_8"]:6.3f} {r["hi12"]:6.3f} {pl12:6.3f}')

print('\nf32 (bits per weight)')
for k, ss in sorted(by_kind.items()):
    if ss[0]['cls'] != 'f32':
        continue
    r = {key: wavg(ss, key) for key in ('Hexp', 'df11', 'df11_12')}
    r.update(n=len(ss), gb=sum(s['n'] for s in ss) * 4 / 1e9, cls='f32')
    summary[k] = r
    print(f'{k:14} {len(ss):3d} {r["gb"]:6.3f} Hexp {r["Hexp"]:5.3f} df11 {r["df11"]:6.3f} df<=12 {r["df11_12"]:6.3f}')

json.dump(summary, open('summary.json', 'w'), indent=1)

print('\nper tensor (fp8: H0 / <=12-bit Huffman bits per byte; fp4: per value; bf16: Hword / df11<=12)')
for k in ('attn_wq_a', 'attn_wkv', 'attn_wq_b', 'wo_a', 'attn_wo_b', 'shared', 'engram', 'experts_w1',
          'experts_w2', 'experts_w3', 'router', 'head', 'embed', 'mtp_dense', 'mtp_experts'):
    ss = by_kind.get(k, [])
    cells = []
    for s in ss:
        m = re.match(r'layers\.(\d+)\.', s['name'])
        tag = m.group(1) if m else s['name'].split('.')[0] + s['name'].split('.')[1]
        if s['cls'] == 'fp8':
            cells.append(f'{tag}:{s["H0"]:.2f}/{s["h12"]:.2f}')
        elif s['cls'] == 'fp4':
            cells.append(f'{tag}:{s["H0"]:.3f}/{s["hpair12"]:.3f}')
        else:
            cells.append(f'{tag}:{s["Hw"]:.2f}/{s["df11_12"]:.2f}')
    print(f'{k:12} ' + ' '.join(cells))
