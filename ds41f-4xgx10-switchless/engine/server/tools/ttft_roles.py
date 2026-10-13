#!/usr/bin/env python3
"""A turn's critical path by role (docs/engine.md, "A small turn's time to first token"): every main-stream
operation of an nsys capture (ttft_prof.py's ranges and turns) named by what it does in the layer (wq_a + wkv,
the all-gathers, wq_b, sparse attention, wo_a, wo_b, the boundaries, the router gate, the shared expert, the
routed MoE, ...), with its own time (after the previous main-stream operation ends: PDL overlap counted once)
and the gap before it, split into waiting while the comm stream runs (a collective's tail) and the rest
(launches, host). Own time plus gaps over a turn's operations is the main stream's span to the token.
  ttft_roles.py CAPTURE.sqlite --turns TURNS_LOG [--phase long] [--json OUT]
Prints per group (phase, size) and pass the medians over its turns of each role's own time, its gaps, the
operations' count, and the floor where one is known (bytes this rank streams at 235 GB/s); the comm stream's
operations (the all-to-all, the reduce-scatters) by duration."""
import collections
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttft_prof as tp  # noqa: E402

BW = 235e9
# bytes per layer this rank streams, by role (FP8 weights 1 B, bf16 2 B; block scales ignored)
FLOOR_B = {
    'wqa_wkv': (1280 + 512) * 5120,      # every rank's wq_a and wkv for own rows (PLayer)
    'wq_b': 8192 * 1280,                 # this rank's 16 heads
    'wo_a': 2048 * 4096,                 # this rank's two groups
    'wo_b': 4 * 5120 * 2048,             # every rank's slice for own rows
    'gate': 384 * 5120 * 2,              # bf16, all 384 experts
    'sh_up': 2 * 576 * 5120,             # the shared expert's w1, w3 (FP8)
    'sh_down': 5120 * 576,
}


def role(part, name, seen):
    """an operation's role from its part, kernel name and what the layer ran before it (seen: names so far)"""
    if part == 'attn':
        if name == 'k_gemm_fp8':
            n = seen['k_gemm_fp8']
            return 'wqa_wkv' if n == 0 else 'wq_b' if n == 1 else 'wo_b'
        if name in ('k_gemm16', 'k_round_bf16'):
            return 'wo_a' if seen['attn_p_kernel'] else 'compressor'
        return {'allgather_multi_kernel': 'ag_attn', 'memcpy': 'ring_copy', 'kv_write_kernel': 'kv_write',
                'emit_kernel': 'emit', 'q_norm_kernel': 'q_norm', 'k_act_quant': 'act_quant',
                'attn_p_kernel': 'sparse_attn', 'k_sum4_bf16': 'sum4', 'hc_boundary_kernel': 'bnd_attn',
                'alltoall_rows_kernel': 'a2a'}.get(name, 'attn_other')
    if part == 'index':
        return {'k_gemm_fp8': 'ix_wqb', 'k_gemm16': 'ix_wproj', 'allgather_kernel': 'ag_topk'}.get(name, 'ix_scan')
    if part == 'ffn':
        if name == 'k_gemm_fp8':
            return 'sh_up' if seen['k_gemm_fp8'] == 0 else 'sh_down'
        if name.startswith('k_moe_up'):
            return 'moe_up'
        if name.startswith('k_moe_down'):
            return 'moe_down'
        if name in ('k_moe_bucket', 'k_moe_scatter', 'k_moe_sortpos'):
            return 'bucket'
        return {'k_gemm16': 'gate', 'k_router_topk': 'router', 'allgather_multi_kernel': 'ag_ffn',
                'reduce_scatter_bulk_kernel': 'rs', 'k_moe_combine_rs': 'rs', 'k_f32_to_bf16': 'f32_bf16',
                'hc_boundary_kernel': 'bnd_ffn'}.get(name, 'ffn_other')
    if part == 'engram':
        return 'engram'
    return part if part not in ('-', None) else 'misc'


def turn_roles(ops, P, ks):
    """per (pass, role): own ms, gap while the comm stream runs, other gap, count; comm-stream ms by role"""
    chunk = [k for k in ks if any(x.startswith('chunk ') for x in P[k])]
    if not chunk:
        return None
    main = collections.Counter(ops[k][2] for k in chunk).most_common(1)[0][0]
    head = [k for k in ks if 'head' in P[k]]
    t0 = min(ops[k][0] for k in ks)
    t1 = max(ops[k][1] for k in head) if head else max(ops[k][1] for k in ks)
    comm = sorted((ops[k][0], min(ops[k][1], t1)) for k in ks if ops[k][2] != main and ops[k][0] < t1)
    out = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    cs = collections.defaultdict(lambda: [0.0, 0])
    seen = collections.defaultdict(collections.Counter)  # (pass, layer, part) -> names so far
    pe = t0
    for k in sorted(ks, key=lambda k: ops[k][0]):
        s, e, sid, _, name = ops[k]
        if s >= t1:
            continue
        _, ph, L, pt = tp.part_of(P[k])
        if ph in ('head', 'rep_in', 'ced_project', 'ced_keep', 'spec_accept'):
            rl = ph
        else:
            rl = role(pt, name, seen[(ph, L, pt)])
        seen[(ph, L, pt)][name] += 1
        if sid != main:
            cs[(ph, rl)][0] += (min(e, t1) - s) / 1e6
            cs[(ph, rl)][1] += 1
            continue
        e = min(e, t1)
        g = max(0, s - pe)
        ov = 0
        if g > 0:
            for cs_, ce_ in comm:
                if cs_ >= s:
                    break
                ov += max(0, min(ce_, s) - max(cs_, pe))
            ov = min(ov, g)
        o = out[(ph, rl)]
        o[0] += max(0, e - max(s, pe)) / 1e6
        o[1] += ov / 1e6
        o[2] += (g - ov) / 1e6
        o[3] += 1
        pe = max(pe, e)
    return {'span': (t1 - t0) / 1e6, 'main': {'%s|%s' % k: v for k, v in out.items()},
            'comm': {'%s|%s' % k: v for k, v in cs.items()}}


def main():
    a = sys.argv[1:]
    if not a or a[0].startswith('--'):
        sys.exit(__doc__)
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    ranges, api, ops = tp.load(a[0])
    P = tp.paths(ranges, api, ops)
    rows = {}
    if opt('--turns'):
        for line in open(opt('--turns')):
            f = line.rstrip('\n').split('\t')
            if f[0] == 'row':
                rows[int(f[1])] = (f[2], int(f[3]), int(f[4]), int(f[5]))
    want = opt('--phase', 'long')
    groups = collections.OrderedDict()
    for key, ks in tp.analyse(ops, P).items():
        r = turn_roles(ops, P, ks)
        if not r:
            continue
        m = re.match(r'req (\d+) ', key)
        if m and int(m.group(1)) in rows:
            g = rows[int(m.group(1))][:2]
        else:  # another rank: its prefill command's position and rows against rank 0's rows
            top = P[ks[0]][0] if P[ks[0]] else ''
            mm = re.match(r'cmd 4 pos0 (\d+) rows (\d+)', top)
            g = next(((ph, sz) for ph, sz, pr, ca in rows.values() if mm and ca == int(mm.group(1)) and
                      pr - ca == int(mm.group(2))), ('rows', 0))
        r['group'] = g
        groups.setdefault(g, []).append(r)
    if opt('--json'):
        with open(opt('--json'), 'w') as f:
            for g, rs in groups.items():
                for r in rs:
                    f.write(json.dumps(r) + '\n')
    for (ph, sz), rs in groups.items():
        if ph != want:
            continue
        print('== %s +%d: %d turns, span %.1f ms (median)' % (ph, sz, len(rs), st.median(r['span'] for r in rs)))
        keys = sorted({k for r in rs for k in r['main']}, key=lambda k: (k.split('|')[0], k.split('|')[1]))
        passes = collections.OrderedDict()
        for k in keys:
            passes.setdefault(k.split('|')[0], []).append(k)
        for p, ks_ in passes.items():
            med = lambda k, i: st.median(r['main'].get(k, [0, 0, 0, 0])[i] for r in rs)
            tot = [st.median(sum(r['main'].get(k, [0, 0, 0, 0])[i] for k in ks_) for r in rs) for i in range(3)]
            print('  %-11s own %7.2f gap(comm) %6.2f gap(other) %5.2f' % (p, tot[0], tot[1], tot[2]))
            for k in sorted(ks_, key=lambda k: -med(k, 0) - med(k, 1) - med(k, 2)):
                rl = k.split('|')[1]
                n = med(k, 3)
                fl = ''
                if rl in FLOOR_B and p in ('enc', 'rep', 'ex'):
                    layers = 20 if p != 'ex' else 40
                    fl = 'floor %6.2f' % (FLOOR_B[rl] / BW * 1e3 * layers)
                print('    %-12s %7.2f + %6.2f + %5.2f  n %4d  %s' % (rl, med(k, 0), med(k, 1), med(k, 2), n, fl))
        ck = sorted({k for r in rs for k in r['comm']})
        if ck:
            print('  comm stream: ' + '; '.join('%s %.2f ms (%d)' % (k, st.median(r['comm'].get(k, [0, 0])[0] for r in rs),
                                                                     st.median(r['comm'].get(k, [0, 0])[1] for r in rs))
                                                for k in ck))


if __name__ == '__main__':
    main()
