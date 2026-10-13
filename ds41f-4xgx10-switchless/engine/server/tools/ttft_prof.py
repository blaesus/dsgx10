#!/usr/bin/env python3
"""Where a turn's GPU time goes, from an nsys capture of a test server (a build with -DDSGB10_NVTX: ranges
per turn, chunk, layer and part; model/nvtx.h), exported to sqlite:
  ttft_prof.py CAPTURE.sqlite [--turns TURNS_LOG] [--routes DIR] [--layers N] [--names] [--json OUT]
Each GPU operation (kernel, copy, memset) is put in the NVTX ranges open on the launching thread when its
API call was made: the turn (rank 0's `req N new X ctx Y how`) or the command (another rank's `cmd OP ...`),
the chunk (`chunk a=A M=M [ced] [sample]`), the pass and layer (`enc`, `ex`, `rep` Lnn), the part (`attn`,
`ffn`, `engram`, `index`, `spec_hidden`) and the rest (`ced_project`, `ced_keep`, `replay`, `head`,
`spec_accept`). Per turn, from its first GPU operation to the end of the head (the token):
  - the main stream's busy time by part and kernel class, and its idle time (gaps between operations),
    split into gaps while the comm stream runs (waiting for a collective) and the rest (launches, host);
  - the comm stream's busy time (the column groups' reduce-scatters and the all-to-all, overlapped);
  - the collectives on the main stream (all-gathers: they spin until the peers' data is in).
Turns group by (phase, size) with --turns (ttft_turns.py's rows: rank 0's request ids; another rank's
chunks match by their position), else by new rows; medians over a group's turns. --routes: the experts
each layer touched (ttft_part.sh ROUTES's dump) for the MoE's floor at 235 GB/s. --layers N: the per-layer
table of the group's first turn of N new rows. --names: the kernel names and their classes."""
import bisect
import collections
import json
import os
import re
import sqlite3
import statistics as st
import sys

BW = 235e9
# bytes per touched expert per layer per rank: w1, w3, w2 in fp4, and their scales' 1-bit code (dsk.h ScaleCode)
EXP_B = 3 * 576 * 5120 // 2 + 2 * 36 * 84 * 4 + 320 * 12 * 4

CLASSES = [  # (class, regex on the kernel's short name); the first match wins
    ('moe_up', r'^k_moe_up'),
    ('moe_down', r'^k_moe_down'),
    ('moe_combine_rs', r'^k_moe_combine'),
    ('moe_bucket', r'^k_moe_(bucket|sortpos|scatter)'),
    ('router', r'^k_router'),
    ('gemm_fp8', r'^k_gemm_fp8|gemm_fp8'),
    ('gemm_bf16', r'^k_gemm16|gemm16|gemm_bf16'),
    ('gemv', r'gemv|head_fp8'),
    ('sparse_attn', r'^attn_p_kernel|^attn_kernel|attn_combine'),
    ('index', r'index|^score|^select_kernel|^merge_kernel|^q_prep'),
    ('mhc', r'^hc_'),
    ('coll', r'allgather|alltoall|reduce_scatter|allreduce'),
    ('engram', r'engram|k_wait_seq'),
    ('attn_misc', r'kv_write|q_norm|act_quant|emit|carry|rope|k_iota|k_embed'),
    ('elementwise', r'sum4|f32_to_bf16|round_bf16|mul_bf16|k_fill'),
    ('sample', r'k_sample|k_step_in|argmax|k_spec'),
    ('spec', r'main_hidden|draft|accept|conf_kernel|markov'),
]


def klass(name):
    for c, rx in CLASSES:
        if re.search(rx, name):
            return c
    return 'other'


def load(path):
    db = sqlite3.connect(path)
    S = dict(db.execute('select id, value from StringIds'))
    tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    nv = db.execute('select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null and '
                    'eventType in (59, 60)').fetchall()
    ranges = [(s, e, t if t is not None else S.get(ti, '?'), g) for s, e, t, ti, g in nv]
    api = {c: (s, g) for c, s, g in db.execute('select correlationId, start, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME')}
    ops = []
    for s, e, sid, c, nm in db.execute('select start, end, streamId, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL'):
        ops.append([s, e, sid, c, S.get(nm, '?')])
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tabs:
        for s, e, sid, c, b in db.execute('select start, end, streamId, correlationId, bytes from CUPTI_ACTIVITY_KIND_MEMCPY'):
            ops.append([s, e, sid, c, 'memcpy'])
    if 'CUPTI_ACTIVITY_KIND_MEMSET' in tabs:
        for s, e, sid, c in db.execute('select start, end, streamId, correlationId from CUPTI_ACTIVITY_KIND_MEMSET'):
            ops.append([s, e, sid, c, 'memset'])
    return ranges, api, ops


def paths(ranges, api, ops):
    """each op's NVTX path (outermost first), by its API call's thread and time"""
    ev = collections.defaultdict(list)
    for i, (s, e, t, g) in enumerate(ranges):
        ev[g].append((s, 0, i))
        ev[g].append((e, 2, i))
    for k, op in enumerate(ops):
        a = api.get(op[3])
        if a:
            ev[a[1]].append((a[0], 1, k))
    out = [()] * len(ops)
    for g, es in ev.items():
        es.sort()
        stack = []
        for t, kind, i in es:
            if kind == 0:
                stack.append(i)
            elif kind == 2:
                if i in stack:
                    stack.remove(i)
            else:
                out[i] = tuple(ranges[j][2] for j in stack)
    return out


def part_of(path):
    """(turn key, phase, layer, part) of an op's path"""
    turn = phase = part = None
    layer = -1
    for x in path:
        if x.startswith('req ') or x.startswith('cmd '):
            turn = turn or x
        elif x.startswith('chunk '):
            turn = (turn, x) if turn and not isinstance(turn, tuple) else turn
            phase = phase or 'chunk'
        elif x[:4] in ('enc ', 'rep ') or x.startswith('ex '):
            phase, layer = x.split()[0], int(x.split()[1][1:])
        elif x.startswith('replay'):
            phase = 'rep_in'
        elif x in ('ced_project', 'ced_keep', 'head', 'spec_accept'):
            phase = x
        else:
            part = x.split()[0]
    return turn, phase or 'ops', layer, part or '-'


def union(iv):
    iv = sorted(iv)
    tot, cur_s, cur_e = 0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def analyse(ops, P):
    """per turn: its ops, grouped"""
    turns = collections.OrderedDict()
    cur = prev = None
    for k in sorted(range(len(ops)), key=lambda k: ops[k][0]):
        path = P[k]
        if not path:
            continue
        top = path[0]
        if top.startswith('req '):
            key = top
        elif top.startswith('cmd '):
            # another rank: a turn is a prefill command (`cmd 4 pos0 P rows N`: its chunks, the replay and
            # the head) and the commands after it up to the next one
            if (top.startswith('cmd 4 ') and top != prev) or cur is None:
                cur = 'turn@%d' % ops[k][0]
            key = cur
            prev = top
        else:
            continue
        turns.setdefault(key, []).append(k)
    return turns


def turn_stats(ops, P, ks, ndiag=False):
    chunk = [k for k in ks if any(x.startswith('chunk ') for x in P[k])]
    if not chunk:
        return None
    sc = collections.Counter(ops[k][2] for k in chunk)
    main = sc.most_common(1)[0][0]
    head = [k for k in ks if 'head' in P[k]]
    t0 = min(ops[k][0] for k in ks)
    t1 = max(ops[k][1] for k in head) if head else max(ops[k][1] for k in ks)
    chunks = sorted({x for k in chunk for x in P[k] if x.startswith('chunk ')})
    m = re.search(r'req (\d+) new (\d+) ctx (\d+) (\w+)', ks and P[ks[0]][0] or '')
    r = dict(span=(t1 - t0) / 1e6, chunks=chunks, req=int(m.group(1)) if m else None,
             new=int(m.group(2)) if m else None, ctx=int(m.group(3)) if m else None, how=m.group(4) if m else None)
    mains = sorted((ops[k][0], ops[k][1], k) for k in ks if ops[k][2] == main and ops[k][0] < t1)
    comm = sorted((ops[k][0], ops[k][1]) for k in ks if ops[k][2] != main and ops[k][0] < t1)
    # each operation's own time on the main stream: with programmatic dependent launch a kernel starts while
    # the one before it runs (and waits for it), so the overlap is counted to the earlier one
    own = []
    pe = t0
    for s, e, k in mains:
        e = min(e, t1)
        own.append((max(s, pe), max(e, pe), k))
        pe = max(pe, e)
    busy = collections.Counter()
    cls = collections.Counter()
    for s, e, k in own:
        _, ph, L, pt = part_of(P[k])
        c = klass(ops[k][4])
        busy[(ph, pt, c)] += (e - s) / 1e6
        cls[c] += (e - s) / 1e6
    # gaps on the main stream, attributed to the op after them; split by whether the comm stream runs
    cstarts = [s for s, e in comm]
    gaps = collections.Counter()
    gap_comm = 0.0
    prev_e = t0
    for s, e, k in mains:
        g = s - prev_e
        if g > 0:
            _, ph, L, pt = part_of(P[k])
            gaps[(ph, pt)] += g / 1e6
            # comm overlap within [prev_e, s)
            i = bisect.bisect_left(cstarts, prev_e) - 1
            ov = 0
            for j in range(max(0, i), len(comm)):
                cs_, ce_ = comm[j]
                if cs_ >= s:
                    break
                ov += max(0, min(ce_, s) - max(cs_, prev_e))
            gap_comm += min(ov, g) / 1e6
        prev_e = max(prev_e, e)
    r['main_busy'] = union([(s, min(e, t1)) for s, e, k in mains]) / 1e6
    r['idle'] = r['span'] - r['main_busy']
    r['idle_comm'] = gap_comm
    r['comm_busy'] = union([(s, min(e, t1)) for s, e in comm]) / 1e6
    r['cls'] = dict(cls)
    r['busy'] = {'%s|%s|%s' % k: v for k, v in busy.items()}
    r['gaps'] = {'%s|%s' % k: v for k, v in gaps.items()}
    # per layer: busy and gaps by pass, layer, part
    lay = collections.defaultdict(lambda: collections.Counter())
    prev_e = t0
    for s, e, k in own:
        _, ph, L, pt = part_of(P[k])
        g = max(0, s - prev_e)
        lay[(ph, L)]['busy_' + (pt if pt in ('attn', 'ffn', 'engram') else 'x')] += (e - s) / 1e6
        lay[(ph, L)]['gap'] += g / 1e6
        lay[(ph, L)][klass(ops[k][4])] += (e - s) / 1e6
        prev_e = max(prev_e, e)
    r['layers'] = {'%s|%d' % k: dict(v) for k, v in lay.items()}
    r['n_ops'] = len(mains)
    return r


def main():
    a = sys.argv[1:]
    if not a or a[0].startswith('--'):
        sys.exit(__doc__)
    opt = lambda k: a[a.index(k) + 1] if k in a else None
    ranges, api, ops = load(a[0])
    P = paths(ranges, api, ops)
    if '--names' in a:
        tot = collections.Counter()
        cnt = collections.Counter()
        for op in ops:
            tot[op[4]] += (op[1] - op[0]) / 1e6
            cnt[op[4]] += 1
        for nm, t in tot.most_common():
            print('%10.2f ms %7d  %-14s %s' % (t, cnt[nm], klass(nm), nm))
        return
    turns = analyse(ops, P)
    stats = []
    for key, ks in turns.items():
        r = turn_stats(ops, P, ks)
        if r:
            r['key'] = key
            stats.append(r)
    rows = {}
    if opt('--turns'):
        for line in open(opt('--turns')):
            f = line.rstrip('\n').split('\t')
            if f[0] == 'row':
                rows[int(f[1])] = (f[2], int(f[3]), int(f[4]), int(f[5]))
    # another rank's turns: match by the chunks' positions against rank 0's rows (prompt, cached)
    for r in stats:
        if r['req'] is not None and r['req'] in rows:
            ph, sz, pr, ca = rows[r['req']]
            r['group'] = (ph, sz)
        else:
            m = [re.match(r'chunk a=(\d+) M=(\d+)', c) for c in r['chunks']]
            a0 = min(int(x.group(1)) for x in m if x)
            n = sum(int(x.group(2)) for x in m if x)
            r['a0'], r['rows'] = a0, n
            g = [(ph, sz) for rid, (ph, sz, pr, ca) in rows.items() if ca == a0 and pr - ca == n]
            r['group'] = g[0] if g else ('rows', n)
    jout = opt('--json')
    if jout:
        with open(jout, 'w') as f:
            for r in stats:
                f.write(json.dumps({k: v for k, v in r.items()}, default=str) + '\n')
    groups = collections.OrderedDict()
    for r in stats:
        groups.setdefault(tuple(r['group']), []).append(r)
    allc = sorted({c for r in stats for c in r['cls']}, key=lambda c: -sum(r['cls'].get(c, 0) for r in stats))
    print('per turn, medians, ms: span = the first GPU operation to the token; main = the main stream busy; idle (of it '
          'with the comm stream running); comm busy')
    print('%-9s %6s %3s %8s %8s %8s %8s %8s | ' % ('phase', 'size', 'n', 'span', 'main', 'idle', 'idle_cs', 'comm') +
          ' '.join('%9s' % c[:9] for c in allc))
    for (ph, sz), rs in groups.items():
        med = lambda f: st.median(f(r) for r in rs)
        print('%-9s %6s %3d %8.1f %8.1f %8.1f %8.1f %8.1f | ' % (ph, sz, len(rs), med(lambda r: r['span']),
                                                                med(lambda r: r['main_busy']), med(lambda r: r['idle']),
                                                                med(lambda r: r['idle_comm']),
                                                                med(lambda r: r['comm_busy'])) +
              ' '.join('%9.2f' % med(lambda r: r['cls'].get(c, 0)) for c in allc))
    # by pass: busy and gaps
    print('\nby pass and part, medians, ms (busy / gaps before its operations)')
    for (ph, sz), rs in groups.items():
        keys = collections.Counter()
        for r in rs:
            for k, v in r['busy'].items():
                p, pt, c = k.split('|')
                keys[(p, pt)] += 0
        out = []
        for p, pt in sorted(keys):
            b = st.median(sum(v for k, v in r['busy'].items() if k.startswith('%s|%s|' % (p, pt))) for r in rs)
            g = st.median(r['gaps'].get('%s|%s' % (p, pt), 0) for r in rs)
            out.append('%s/%s %.1f+%.1f' % (p, pt, b, g))
        print('%-9s %6s: %s' % (ph, sz, '; '.join(out)))
    if opt('--layers'):
        n = int(opt('--layers'))
        for (ph, sz), rs in groups.items():
            if sz != n:
                continue
            r = rs[len(rs) // 2]
            print('\nper layer, %s +%d (turn %s): busy attn / ffn / other, gap; by class' % (ph, sz, r['key']))
            for k in sorted(r['layers'], key=lambda k: (k.split('|')[0], int(k.split('|')[1]))):
                v = r['layers'][k]
                cl = ' '.join('%s %.2f' % (c, x) for c, x in sorted(v.items(), key=lambda t: -t[1])
                              if not c.startswith('busy') and c != 'gap' and x >= 0.01)
                print('  %-12s attn %6.2f ffn %6.2f x %6.2f gap %6.2f | %s' % (
                    k, v.get('busy_attn', 0), v.get('busy_ffn', 0), v.get('busy_x', 0) + v.get('busy_engram', 0),
                    v.get('gap', 0), cl))


if __name__ == '__main__':
    main()
