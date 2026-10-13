# fixcost.py: the fixed-length codes' bytes and ms (docs/engine.md, "Lossless weight compression", Part 3)
# from wfix's output, with each decoder's added instructions a warp-step (fixdec.cu's SASS) costed between
# hidden and v2's measured exposure (bench_wcomp: 1.40 and 2.58 ps an added instruction a warp-step, wo_b
# and wq_b). Bytes saved at 235 GB/s. Usage: fixcost.py WFIX.jsonl... (FP8 lines need wfix's "mm")
import json, sys

BW = 235e9
K1, R6 = 17.9, 37.3                   # ms: a k = 1 step; a 6-row spec step with its draft pass
EXPO = (0.0, 1.40e-12, 2.58e-12)      # s an added instruction a warp-step
EXPERT = 4700160                      # bytes of one expert a layer (w1, w2, w3 and their scales)
E_K1 = 6 * 40 * EXPERT                # 6 experts a layer
E_R6 = 21 * 40 * EXPERT + 22 * EXPERT # 21 a layer, and the draft pass's ~22
HEAD = 330956800
F_K1, F_R6 = 1.741e9, 1.741e9 + 0.308e9  # FP8 a step; the draft pass's FP8 (its head copy, attention, ...)
D_SCALE, D_HEAD5, D_HEAD4, D_FP8 = 10, 44, 31, 85  # fixdec.cu, over the raw loops (+2 for the head's flag)

rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
sc = [d for d in rows if d['class'] == 'fp4s']
hd = [d for d in rows if d['class'] == 'bf16' and d['name'] == 'head.weight']
f8 = [d for d in rows if d['class'] == 'fp8' and 'mm' in d]

def ms(gross, steps, d):
    return [(gross / BW - steps * d * e) * 1e3 for e in EXPO]

def show(name, k1, r6, note=''):
    f = lambda v, t: '%s ms (%s %%)' % (' / '.join('%.2f' % x for x in v), ' / '.join('%.1f' % (100 * x / t) for x in v))
    print('%-28s k=1 %s | 6 rows %s %s' % (name, f(k1, K1), f(r6, R6), note))

print('ms saved: decode hidden / at v2 wo_b exposure / at v2 wq_b exposure')
if sc:
    pair = sum(d['in_pair'] for d in sc) / sum(d['groups'] for d in sc)
    grp = 4 + 1 / 8 + (1 - pair) * 32  # code word, flag bit, the raw group when it leaves the pair
    fr = (32 - grp) / 544             # a warp-step: 512 bytes of values, 32 of scales
    show('scales 1 bit (%.2f %% in pair)' % (100 * pair), ms(E_K1 * fr, E_K1 / 544, D_SCALE),
         ms(E_R6 * fr, E_R6 / 544, D_SCALE), '%.1f %% of the scales' % (100 * (32 - grp) / 32))
if hd:
    r = hd[0]['range']
    le15 = sum(r[:16]) / sum(r)
    for name, bits, d in (('head 5 bits', 13.0, D_HEAD5),
                          ('head 4 bits + tile base', le15 * 12 + (1 - le15) * 13 + 8 / 256, D_HEAD4)):
        g = ms(HEAD * (16 - bits) / 16, HEAD / 512, d)
        show(name + ' (%.1f %%)' % (100 * (16 - bits) / 16), g, g)
if f8:
    n = o = 0
    for d in f8:
        h, mm = d['hexp'], d['mm']
        b = max(range(9), key=lambda b: sum(h[b:b + 8]))
        tot = sum(mm)
        n += tot
        o += tot - sum(mm[lo * 16 + hi] for lo in range(b, 16) for hi in range(lo, min(16, b + 8)))
    lanes = o / n
    step = 256 + 192 + 4 + 2 * 32 * lanes  # nibbles, 3 code planes, lane flags, flagged lanes' 4th plane
    sv = (512 - step) / 512 - 0.0025       # less run offsets
    show('fp8 3 bits (%.1f %%, %.0f %% lanes)' % (100 * sv, 100 * lanes), ms(F_K1 * sv, F_K1 / 512, D_FP8),
         ms(F_R6 * sv, F_R6 / 512, D_FP8))
