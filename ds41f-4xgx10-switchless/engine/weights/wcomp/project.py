#!/usr/bin/env python3
"""Lossless weight compression, Part 1: bytes per rank and ms per step by scheme.

Inputs: summary.json (analyze.py: bits per code by weight kind, from the sampled tensors) and
toc_list.txt (wcomp --list: every tensor of rank 0's file, dtype and bytes). Each tensor of the file
takes its kind's bits; kinds that were not sampled (norms, biases, FP8 block scales) stay raw."""
import json
import sys
from collections import defaultdict

BW = 235e9  # B/s, weight streaming (engine.md: k = 1's 14.3 ms of weights)
OVH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0025  # offsets + flushed bits, of the raw bytes
LAYERS, EXPERTS = 40, 384
STEP1, STEP6 = 17.9, 37.3  # ms: k = 1; a 6-row spec step with its draft pass (engine.md)
U6 = 21  # distinct experts per layer at 6 rows (trees study: 15.7-16.8 at 4 chain rows, the fit's 37.3)

summ = json.load(open('summary.json'))


def kind(name):
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
        return 'experts_' + name.split('.')[-2]
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


CLS = {1: 'f32', 2: 'bf16', 3: 'fp8', 5: 'fp4'}


def lookup(k, cls):
    for key in (k, k + '_' + cls):
        if key in summ and summ[key]['cls'] == cls:
            return summ[key]
    return None


# bits per code, by scheme: 'ent' the order-0 entropy (FP4 values given their block scale), what ANS
# approaches; 't12' Huffman with a <= 12-bit decode table (FP8 bytes; bf16 / f32 exponents with the
# rest raw, DFloat11's split; FP4 nibbles with one table per scale value, scale pairs as one symbol);
# 't8' a <= 8-bit table (FP8 / bf16 exponents only; FP4 values raw, scales one by one)
def code_bits(r, cls, scheme):
    if cls == 'fp8':
        return dict(ent=r['H0'], t12=r['h12'], t8=r['df11'])[scheme], 8
    if cls == 'bf16':
        return dict(ent=r['Hw'], t12=r['df11_12'], t8=r['df11_8'])[scheme], 16
    if cls == 'f32':
        return dict(ent=r['Hexp'] + 24, t12=r['df11_12'], t8=r['df11_12'])[scheme], 32
    if cls == 'fp4':
        return dict(ent=r['Hs'], t12=r['per_scale'], t8=4.0)[scheme], 4
    if cls == 'scale':
        return dict(ent=min(r['sH0'], r['sHc']), t12=r['spair12'], t8=r['shuff8'])[scheme], 8


def saved_frac(name, dt, scheme):
    """fraction of the tensor's bytes a scheme saves, net of OVH"""
    k = kind(name)
    if dt == 4:  # ue8m0: the experts' scales are coded; FP8 block scales stay raw
        if '.experts.' not in name or '.shared_experts.' in name:
            return 0.0
        r, cls = lookup(k, 'fp4'), 'scale'
    elif dt in CLS:
        r, cls = lookup(k, CLS[dt]), CLS[dt]
    else:
        return 0.0
    if r is None:
        return 0.0
    b, raw = code_bits(r, cls, scheme)
    return max(0.0, 1 - b / raw - OVH)


toc = []
for line in open('toc_list.txt'):
    name, dt, lay, nb = line.rstrip('\n').split('\t')
    toc.append((name, int(dt), int(nb)))


def group(name):
    if name.startswith('mtp.'):
        return 'mtp_experts' if '.ffn.experts.' in name else 'mtp'
    if name.startswith('layers.'):
        return 'routed' if '.ffn.experts.' in name else 'dense'
    return name  # head.weight, embed.weight, norm.weight


# per group: raw bytes and bytes saved per scheme; per class too
SCHEMES = ('ent', 't12', 't8')
raw = defaultdict(float)
sav = defaultdict(float)
for name, dt, nb in toc:
    g = group(name)
    c = 'fp4 scales' if (dt == 4 and '.experts.' in name and '.shared_experts.' not in name) else \
        {1: 'f32', 2: 'bf16', 3: 'fp8', 4: 'e8m0', 5: 'fp4'}[dt]
    for key in ((g, c), (g, '*')):
        raw[key] += nb
        for s in SCHEMES:
            sav[key + (s,)] += nb * saved_frac(name, dt, s)
    raw[('*', '*')] += nb
    for s in SCHEMES:
        sav[('*', '*', s)] += nb * saved_frac(name, dt, s)

print(f'overhead {OVH:.2%} of raw bytes; per rank 0, GB raw, then saved % by scheme (ent / t12 / t8)')
for key in sorted(raw):
    if key[0] in ('norm.weight',):
        continue
    print(f'{key[0]:13} {key[1]:11} {raw[key] / 1e9:8.3f}  ' +
          '  '.join(f'{sav[key + (s,)] / raw[key]:6.1%}' for s in SCHEMES))

# per step
e1 = (raw[('routed', '*')]) / (LAYERS * EXPERTS)  # B per expert per layer
es = {s: sav[('routed', '*', s)] / (LAYERS * EXPERTS) for s in SCHEMES}
dense = raw[('dense', '*')] + raw[('head.weight', '*')]
dsav = {s: sav[('dense', '*', s)] + sav[('head.weight', '*', s)] for s in SCHEMES}
print(f'\nper step: dense (layers + head) {dense / 1e9:.3f} GB, an expert in every layer {e1 * LAYERS / 1e9:.4f} GB')

# draft pass (DSpark README, FP8 head copy): head 165 MB, per block attention 32 + shared 9 MB, main_proj
# 20 MB (all FP8); 21.7 distinct experts over the three blocks
fp8 = summ['mtp_dense_fp8']
draft_fp8 = 165e6 + 3 * 41e6 + 20e6
draft_exp = 21.7 * e1
for s in SCHEMES:
    f8 = max(0.0, 1 - code_bits(fp8, 'fp8', s)[0] / 8 - OVH)
    d_sav = draft_fp8 * f8 + 21.7 * es[s]
    k1 = dsav[s] + 6 * LAYERS * es[s]
    k6 = dsav[s] + U6 * LAYERS * es[s]
    k6d = k6 + d_sav
    print(f'{s}: k=1 {k1 / 1e9:.3f} GB = {k1 / BW * 1e3:.2f} ms ({k1 / BW * 1e3 / STEP1:.1%} of {STEP1}); '
          f'6 rows {k6 / 1e9:.3f} GB = {k6 / BW * 1e3:.2f} ms, with the draft pass {k6d / BW * 1e3:.2f} ms '
          f'({k6d / BW * 1e3 / STEP6:.1%} of {STEP6}); U=27: {(k6 + 6 * LAYERS * es[s]) / BW * 1e3:.2f} ms')
    fe = es[s] / e1
    fd = dsav[s] / dense
    t200 = 187 * fe + 8 * fd
    t2000 = 218 * fe + 4 * fd
    print(f'    experts {fe:.1%} saved, dense {fd:.1%}; +200 turn {t200:.1f} ms of ~220 ({t200 / 220:.1%}), '
          f'+2000 {t2000:.1f} ms of ~416 ({t2000 / 416:.1%})')
tot = raw[('*', '*')]
print(f'\nwhole rank file {tot / 2**30:.2f} GiB; saved ' +
      ', '.join(f'{s} {sav[("*", "*", s)] / 2**30:.2f} GiB' for s in SCHEMES))

# per kind, the k = 1 step's reads (experts: 6 a layer; 6 rows: U6 a layer)
print('\nper kind: GB per k = 1 step | bits per code ent, t12, t8 | t12 saved % | ms saved at k = 1, 6 rows')
rows = defaultdict(lambda: [0.0, 0.0, None])
for name, dt, nb in toc:
    g = group(name)
    if g not in ('dense', 'head.weight', 'routed'):
        continue
    k = kind(name)
    rc = (lookup(k, CLS[dt]), CLS[dt]) if dt in CLS else None
    if g == 'routed':
        rc = (lookup(k, 'fp4'), 'scale') if dt == 4 else rc
        k = 'experts ' + ('scales' if dt == 4 else 'values')
        nb = nb / EXPERTS * 6
    elif dt == 4:
        k = "fp8 block scales"
    elif dt in CLS:
        k += " " + CLS[dt]
    r = rows[k]
    r[0] += nb
    r[1] += nb * saved_frac(name, dt, 't12')
    if r[2] is None and rc and rc[0]:
        r[2] = rc
tot1 = tot6 = 0.0
for k, (nb, sv, rc) in sorted(rows.items(), key=lambda x: -x[1][1]):
    m6 = U6 / 6 if k.startswith('experts') else 1
    bb = ' '.join(f'{code_bits(rc[0], rc[1], s)[0]:6.3f}' for s in SCHEMES) if rc else '   raw' * 3
    print(f'{k:18} {nb / 1e9:7.4f} | {bb} | {sv / nb:6.1%} | {sv / BW * 1e3:5.3f} {sv * m6 / BW * 1e3:5.3f}')
    tot1 += sv / BW * 1e3
    tot6 += sv * m6 / BW * 1e3
print(f'{"total":18} {sum(r[0] for r in rows.values()) / 1e9:7.4f} | {tot1:.2f} ms at k = 1, {tot6:.2f} at 6 rows '
      f'(no draft pass)')
