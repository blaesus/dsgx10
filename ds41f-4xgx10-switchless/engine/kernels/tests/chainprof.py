"""Per call site breakdown of bench_chain / engine nsys captures (sqlite export).

usage: chainprof.py file.sqlite [nu] [--first] [--meta LOG]
Like kprof.py (steps split at k_step_in, first 3 dropped), but keyed by (kernel, grid) so each GEMV
call site is its own line, with bytes and GB/s on exclusive and busy time. Rows: bench_chain's
k_step_in has a block per row, the engine's first hc_pre too; older captures: the all-reduce's grid
(5 blocks per row).
Exclusive time: the main stream's kernels share the step's time by start order (a kernel is credited
with the time it alone advanced the end of the kernels before it, so a PDL launch that waits for a
collective counts from the collective's end). A side stream's kernel (sa: the mixes, sb: the KV writes,
the compressor's emit, the indexer's wproj) is credited only with time no main-stream kernel covers:
its overlap goes to the main-stream kernel it runs under. The side kernels are those that ran off
k_step_in's stream in an eagerly launched step (graph replays report a stream per node), by name and
grid; without one, by name. --first: the rule before 2026-10-12, overlapped time to whichever kernel
started first, side or not (a side kernel that started a microsecond before k_moe_up took its time).
--meta LOG: bench_chain's output, whose first "chainprof" line gives the experts' mean count (nu, a float under
--routes), the bytes a routed expert reads in moe_up and moe_down (up_b, down_b: coded under --scales) and the
head's (head_b: coded under --head). Without it, nu experts read raw and the raw head.
"""
import sqlite3
import sys
from collections import defaultdict

ARGV = sys.argv[1:]
META = {}
if '--meta' in ARGV:
    i = ARGV.index('--meta')
    for line in open(ARGV[i + 1]):
        if line.startswith('chainprof '):
            f = line.split(':', 1)[1].split()
            META = {k: float(v) for k, v in zip(f[0::2], f[1::2])}
            break
    else:
        sys.exit('chainprof: no "chainprof" line in %s' % ARGV[i + 1])
    del ARGV[i:i + 2]
ARGS = [a for a in ARGV if a != '--first']
FIRST = '--first' in ARGV
db = sqlite3.connect(ARGS[0])
NU = float(ARGS[1]) if len(ARGS) > 1 else META.get('nu', 0)
UP_B = META.get('up_b', 2 * (576 * 2560 + 576 * 160))  # a routed expert's bytes in moe_up (w1, w3), raw
DOWN_B = META.get('down_b', 5120 * 288 + 5120 * 18)  # and in moe_down (w2)
HEAD_B = META.get('head_b', 32320 * 5120 * 2)
names = dict(db.execute('select id, value from StringIds'))
rows = db.execute('select start, end, demangledName, gridX, gridY, gridZ, blockX, streamId from '
                  'CUPTI_ACTIVITY_KIND_KERNEL order by start').fetchall()
SIDE = ('hc_project', 'hc_finish', 'k_mix_ranks', 'k_mix_allreduce', 'kv_write', 'emit_kernel', 'carry_kernel',
        'gemv_kernel')  # by name, when the capture has no eager step


def short(n):
    n = names[n]
    if n.startswith('void '):
        n = n[5:]
    n = n[:n.index('>(') + 1] if '>(' in n else n.split('(')[0]  # keep template arguments: (int)1, ...
    for p in ('dsk::<unnamed>::', 'attn::<unnamed>::', 'comm::<unnamed>::', 'model::ops::<unnamed>::', 'comm::',
              'dsk::', 'attn::'):
        n = n.replace(p, '')
    return n


def klass(n):
    if any(s in n for s in ('allreduce', 'allgather', 'argmax_kernel', 'k_wait_seq', 'reduce_scatter')):
        return 'collective'
    if any(s in n for s in ('gemv', 'moe')):
        return 'weights'
    if any(s in n for s in ('attn_kernel', 'attn_combine', 'score', 'select', 'emit', 'carry', 'kv_write', 'q_prep',
                            'merge', 'q_norm', 'rope', 'index_fused', 'index_dense', 'decode_prep')):
        return 'attention'
    if n.startswith('hc_'):
        return 'mhc'
    return 'other'


def idx_src(L):
    return L in (2, 8, 14, 20) or (L >= 24 and L % 4 == 0)


FP8 = {'wq_a+wkv': 448 * 5120, 'wq_b': 8192 * 1280, 'wq_b+ix': 12288 * 1280, 'wo_b': 5120 * 2048,
       'engram wkv': 6400 * 6144}


def site(n, g, R, nu, pos=None):
    """(label, bytes) of a weight call site. pos: (layer, NORM launches of the persistent FP8 GEMV
    since the last all-reduce, threads per CTA), which tell its call sites apart."""
    if n.startswith('k_gemv_fp8p'):
        L, nnorm, bx = pos
        if n.endswith('(bool)1>'):
            lab = 'wq_a+wkv' if nnorm == 0 else ('wq_b+ix' if idx_src(L) else 'wq_b')
        else:
            lab = 'wo_b' if bx == 512 else 'engram wkv'
        return lab + ' (p)', FP8[lab] * (1 + 1 / 1024)
    if n.startswith('k_gemv_fp8'):
        lab = {56: 'wq_a+wkv', 1024: 'wq_b', 1536: 'wq_b+ix', 640: 'wo_b', 800: 'engram wkv'}.get(g)
        byt = {56: 448 * 5120, 1024: 8192 * 1280, 1536: 12288 * 1280, 640: 5120 * 2048, 800: 6400 * 6144}.get(g, 0)
        return lab or 'fp8 g%d' % g, byt * (1 + 1 / 1024)
    if n.startswith('k_gemv16'):
        if '<(int)1,' in n:
            return 'wo_a', 2048 * 4096 * (1 + 1 / 1024)
        lab = {48: 'gate', 12: 'gate/4', 2020: 'head'}.get(g, 'bf16 g%d' % g)
        rows_ = {48: 384, 12: 96, 2020: 32320}.get(g, 0)
        return lab, HEAD_B if lab == 'head' else rows_ * 5120 * 2
    if n.startswith('k_moe_up'):
        return 'moe_up', 2 * 576 * 5120 * (1 + 1 / 1024) + nu * UP_B
    if n.startswith('k_moe_down'):
        return 'moe_down', 5120 * 576 * (1 + 1 / 1024) + nu * DOWN_B
    if n.startswith('gemv_kernel') or n.startswith('gemv_rows_kernel'):
        return n.split('<')[0] + ' g%d' % g, 0
    return None, 0


steps, streams = [], []
for s, e, dn, gx, gy, gz, bx, sid in rows:
    n = short(dn)
    if n.startswith('k_step_in'):
        steps.append([])
        streams.append([])
    if steps:
        steps[-1].append((s, e, n, gx * gy * gz, bx))
        streams[-1].append(sid)
# the side streams' kernels, (name, grid), from the eager steps: few streams, k_step_in's the main one
on = defaultdict(lambda: [0, 0])
for st, sids in zip(steps, streams):
    if len(set(sids)) > 4:
        continue
    for (s, e, n, g, bx), sid in zip(st, sids):
        on[(n, g)][sid != sids[0]] += 1
side_keys = {k for k, (m, o) in on.items() if o > m}


def is_side(n, g):
    if FIRST:
        return False
    return (n, g) in side_keys if on else n.startswith(SIDE)


def credit(st):
    """each kernel's exclusive time (ns) in a step's kernels, sorted by start, and the idle gaps"""
    x = [0.0] * len(st)
    end = st[0][0]
    for i, (s, e, n, g, bx) in enumerate(st):  # the main stream's, by start order
        if not is_side(n, g):
            x[i] = max(0, e - max(s, end))
            end = max(end, e)
    cov = []  # [start, end) intervals already credited, merged
    for i, (s, e, n, g, bx) in enumerate(st):
        if not is_side(n, g):
            cov.append([s, e])
    cov.sort()
    merged = []
    for a, b in cov:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for i, (s, e, n, g, bx) in enumerate(st):  # a side kernel: what no kernel before it covers
        if not is_side(n, g):
            continue
        free, t = 0, s
        for a, b in merged:
            if b <= t:
                continue
            if a >= e:
                break
            if a > t:
                free += a - t
            t = max(t, b)
        if t < e:
            free += e - t
        x[i] = free
        merged.append([s, e])
        merged.sort()
        m2 = []
        for a, b in merged:
            if m2 and a <= m2[-1][1]:
                m2[-1][1] = max(m2[-1][1], b)
            else:
                m2.append([a, b])
        merged = m2
    span = max(e for s, e, n, g, bx in st) - st[0][0]
    return x, span - sum(b - a for a, b in merged)


groups = defaultdict(list)
for st in steps:
    pre = [g for s, e, n, g, bx in st if n.startswith('hc_pre_kernel')]
    ar = [g for s, e, n, g, bx in st if n.startswith('k_allreduce_hc')]
    if st[0][3] > 1 or (ar and ar[0] == 40 and not pre):  # bench_chain: k_step_in's grid
        groups[st[0][3]].append(st)
    elif pre:
        groups[pre[0]].append(st)
    elif ar:
        groups[ar[0] // 5].append(st)
for R in sorted(groups):
    nu = NU or (6 if R == 1 else 27)
    sts = groups[R][3:]
    if not sts:
        continue
    ns = len(sts)
    wall = sorted((st[-1][1] - st[0][0]) / 1e3 for st in sts)
    excl = defaultdict(float)
    busy = defaultdict(float)
    per = defaultdict(lambda: [0.0, 0.0, 0, 0.0])
    gaps = 0.0
    for st in sts:
        st = sorted(st)
        xs, gap = credit(st)
        gaps += gap / 1e3
        nar = nnorm = 0
        for (s, e, n, g, bx), x in zip(st, xs):
            x /= 1e3
            c = klass(n)
            excl[c] += x
            busy[c] += (e - s) / 1e3
            lab, byt = site(n, g, R, nu, (nar // 2, nnorm, bx))
            if n.startswith('k_allreduce_hc'):
                nar += 1
                nnorm = 0
            elif n.startswith('k_gemv_fp8p') and n.endswith('(bool)1>'):
                nnorm += 1
            key = lab or n[:40]
            per[key][0] += x
            per[key][1] += (e - s) / 1e3
            per[key][2] += 1
            per[key][3] += byt
    print('R = %d (%g experts/layer): %d steps, %d kernels per step; wall median %.3f ms, min %.3f ms; %s' %
          (R, nu, ns, len(sts[0]), wall[ns // 2] / 1e3, wall[0] / 1e3,
           'overlaps to the kernel that started first' if FIRST else
           'side-stream overlaps to the main stream (%s)' % ('%d side kernels from an eager step' % len(side_keys)
                                                             if on else 'side kernels by name')))
    print('  %-11s %9s %9s' % ('class', 'excl ms', 'busy ms'))
    for c in ('weights', 'attention', 'mhc', 'collective', 'other'):
        print('  %-11s %9.3f %9.3f' % (c, excl[c] / ns / 1e3, busy[c] / ns / 1e3))
    print('  %-11s %9.3f' % ('idle gaps', gaps / ns / 1e3))
    tb = sum(v[3] for v in per.values()) / ns
    tx = sum(v[0] for k, v in per.items() if v[3] > 0) / ns
    print('  weight sites: %.1f MB/step, exclusive %.3f ms -> %.1f GB/s' % (tb / 1e6, tx / 1e3, tb / tx / 1e3))
    print('  %-14s %5s %9s %9s %8s %8s %8s %8s' % ('site', 'n', 'excl ms', 'busy ms', 'avg us', 'MB/call', 'GB/s ex',
                                                 'GB/s busy'))
    for k, (x, b, c, byt) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        if x / ns < 5 and byt == 0:
            continue
        mb = byt / c / 1e6 if c else 0
        gx_ = byt / x / 1e3 if x and byt else 0
        gb_ = byt / b / 1e3 if b and byt else 0
        print('  %-14s %5d %9.3f %9.3f %8.1f %8.2f %8.1f %8.1f' % (k[:14], c // ns, x / ns / 1e3, b / ns / 1e3, b / c,
                                                                 mb, gx_, gb_))
