#!/usr/bin/env python3
"""The levers of a small turn's time to first token (docs/engine.md, "A small turn's time to first token"), each
projected end to end on the sampled edit bench (r2's report/joined.jsonl: every request's new tokens and the bench's
server time). A lever is the ms it saves on each prefill pass of a turn, by the pass's kind and rows: an exact
continuation (9-127 new: 40 layers over n rows), the CED encoder (chunks of <= 8192 over layers 0-19) and the replay
(128 rows over layers 20-39); turns of <= 8 new run as decode steps and save nothing. The models come from the
study's measurements (ttft_roles.py on rank 0's capture, 0eb3a7d's engine) and the targets stated with each.
  ttft_levers.py [JOINED_JSONL]"""
import json
import math
import os
import sys

R2 = os.path.expanduser("~/dsgb10/editruns/r2/report/joined.jsonl")


def q_rows(M):
    return (((M + 3) // 4) + 7) & ~7


def passes(n):
    """the prefill passes of a turn of n new rows: (kind, M rows, Q rows per rank, layers)"""
    if n <= 8:
        return [('rows', n, n, 40)]
    if n < 128:
        return [('exact', n, q_rows(n), 40)]
    out = [('enc', min(8192, n - c), q_rows(min(8192, n - c)), 20) for c in range(0, n, 8192)]
    return out + [('rep', 128, 32, 20)]


def load(path=None):
    return [json.loads(l) for l in open(path or R2)]


def project(lever, rows=None, label=''):
    """lever(kind, M, Q, layers, turn) -> ms saved by that pass; prints the gain on off / max / all"""
    rows = rows or load()
    res = {}
    for th in ('off', 'max', 'all'):
        rs = [r for r in rows if th == 'all' or r['thinking'] == th]
        tot = sum(r['total_s'] for r in rs) * 1e3
        sav = 0.0
        for r in rs:
            n = r['n_prompt'] - r['reused']
            sav += sum(lever(k, M, Q, L, r) for k, M, Q, L in passes(n))
        res[th] = (sav, 100 * sav / tot)
    print('%-40s saves %6.0f / %5.0f / %6.0f ms: %5.2f %% (off) %5.2f %% (max) %5.2f %% (all)' % (
        label, res['off'][0], res['max'][0], res['all'][0], res['off'][1], res['max'][1], res['all'][1]))
    return res


BW = 235e9


def interp(x, pts):
    pts = sorted(pts)
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def bnd_t(n):
    """a boundary's measured time, ms, by its own rows (0.224 up to 128 rows, 0.25 at 344, DRAM-bound beyond)"""
    return max(interp(n, [(0, 0.224), (128, 0.224), (184, 0.229), (344, 0.25)]), n * 110e3 / (0.7 * BW) * 1e3)


def own_rows(M):
    return min(q_rows(M), M)  # rank 0's rows (the full share)


# L1: hc_boundary_tuning by the chunk with prefetch 2 (built, off): the kernels agent's bench ratios (0.66 at 32
# rows, 0.92 at 500, 0.99 at 8192), log-interpolated
def l1_ratio(n):
    if n <= 32:
        return 0.66
    if n <= 500:
        return 0.66 + (math.log2(n) - 5) / (math.log2(500) - 5) * 0.26
    return min(0.99, 0.92 + (math.log2(n) - math.log2(500)) / (13 - math.log2(500)) * 0.07)


def L1(k, M, Q, L, r):
    if k == 'rows':
        return 0.0
    n = own_rows(M)
    return 2 * L * bnd_t(n) * (1 - l1_ratio(n))


# L2: an exact boundary that streams at 85 % of DRAM (target, not built): activations ~110 KB a row + hc_fn's
# parts 2.9 MB, plus 5 us
def L2(k, M, Q, L, r, lo=0.0):
    if k == 'rows':
        return 0.0
    n = own_rows(M)
    if n > 512:
        return 0.0
    t = max(lo, (n * 110e3 + 2.9e6) / (0.85 * BW) * 1e3 + 0.005)
    return 2 * L * max(0.0, bnd_t(n) - t)


def L2s(k, M, Q, L, r):  # no boundary under 60 us
    return L2(k, M, Q, L, r, 0.06)


# L3: the dense GEMMs at small M (router gate, shared expert, wq_a + wkv, wq_b, wo_a, wo_b) in tiles that give
# every SM work, streaming at 85 % of DRAM where the bytes bind: ms a layer by the pass's rows (estimates from
# each GEMM's measured time and its bytes; see engine.md)
# (per GEMM the larger of its bytes at 85 % of 235 GB/s and its MMA work at 150 TFLOPS FP8 / 75 bf16, against the
# measured sum of gate, shared expert, wq_a + wkv, wq_b, wo_a and wo_b: 0.608 ms a layer at 44 rows, 0.659 in the
# replay, 0.741 at 200, 0.773 at 350, 0.995 at 728, 1.526 at 1368)
L3_PTS = [(9, 0.194), (44, 0.194), (128, 0.245), (200, 0.32), (350, 0.32), (728, 0.37), (1368, 0.58), (2048, 0.4),
          (4096, 0.0)]
L3_75 = [(9, 0.139), (44, 0.139), (128, 0.19), (200, 0.27), (350, 0.27), (728, 0.33), (1368, 0.55), (2048, 0.37),
         (4096, 0.0)]


def L3(k, M, Q, L, r):
    if k == 'rows':
        return 0.0
    return L * interp(M, L3_PTS)


def L3s(k, M, Q, L, r):  # at 75 % of DRAM
    if k == 'rows':
        return 0.0
    return L * interp(M, L3_75)


# L3a: the router gate alone (3 CTAs at small M: 0.119 ms a layer against 0.017 of bytes)
def L3a(k, M, Q, L, r):
    if k == 'rows' or M > 4096:
        return 0.0
    return L * 0.099


# L4a: the replay's records to the ranks that run them (an all-to-all) instead of every record to every rank
def L4a(k, M, Q, L, r):
    return 1.0 if k == 'rep' else 0.0


# L4b: 8 column groups instead of 4 from 512 rows (the last reduce-scatter half as long; ~0.06 ms a layer more down)
TAIL = [(200, 0.127), (350, 0.165), (728, 0.367), (1368, 0.658), (8192, 2.3)]


def L4b(k, M, Q, L, r):
    if k not in ('enc', 'exact') or M < 512 or M > 2520:
        return 0.0
    return L * max(0.0, interp(M, TAIL) / 2 - 0.06)


# L5: launch gaps (1 ms a turn at most)
def L5(k, M, Q, L, r):
    return 1.0 if k in ('rep', 'exact') else 0.0


# L7 (the kernels agent's): the last layer for the last row only
def L7(k, M, Q, L, r):
    return 3.0 if k == 'rep' else 1.6 if k == 'exact' else 0.0


# L6: the shared expert beside the routed up on a side stream (MoePCfg::side, in the kernel, unused): its 0.09 ms a
# layer hidden but for its 38 us of bytes; instead of L3's shared-expert part
def L6(k, M, Q, L, r):
    if k == 'rows' or M > 1024:
        return 0.0
    return L * 0.05


# L9: at small M each rank streams its own slice once (wq_a + wkv, the gate, wo_b with the reduce-scatter that
# sum4 stands for, Engram's wkv) instead of every rank's for its own rows: on top of L3, ~5 ms a turn (estimate)
def L9(k, M, Q, L, r):
    if k == 'rows' or M > 1024:
        return 0.0
    return L * 0.11 + (1.0 if k in ('enc', 'exact') else 0.0)


rows = load(sys.argv[1] if len(sys.argv) > 1 else R2)
res = {}
for name, f in [('L1 boundary tuning (built, off)', L1), ('L2 boundary at 85 % of DRAM', L2),
                ('L2s boundary, no faster than 60 us', L2s),
                ('L3 small-M GEMM tiles (85 %)', L3), ('L3s small-M GEMM tiles (75 %)', L3s),
                ('L3a router gate tiles only', L3a), ('L6 shared expert on a side stream', L6),
                ('L9 own slices once at small M', L9), ('L4a replay records a2a', L4a),
                ('L4b 8 column groups >= 512 rows', L4b), ('L5 launch gaps', L5), ('L7 last layer last row', L7)]:
    res[name] = project(f, rows, name)
print()
project(lambda *a: L2(*a) + L3(*a) + L4a(*a) + L4b(*a), rows, 'bundle L2 + L3 + L4a + L4b')
project(lambda *a: L2(*a) + L3(*a) + L4a(*a) + L4b(*a) + L5(*a) + L7(*a), rows, 'all of them (L2 for L1)')
project(lambda *a: L1(*a) + L3(*a) + L4a(*a) + L4b(*a), rows, 'L1 + L3 + L4a + L4b')
project(lambda *a: L2s(*a) + L3s(*a) + L4a(*a) + L4b(*a), rows, 'L2s + L3s + L4a + L4b (the low ends)')
project(lambda *a: L2(*a) + L3(*a) + L9(*a) + L4a(*a) + L4b(*a), rows, 'L2 + L3 + L9 + L4a + L4b')
# per turn, ms: the savings of a +44 (exact), +200, +350, +728, +1368 turn
print()
for n in (27, 44, 200, 350, 728, 1368, 11632):
    ps = passes(n)
    print('+%-5d %s' % (n, '  '.join('%s %.1f' % (nm, sum(f(k, M, Q, L, None) for k, M, Q, L in ps))
                                     for nm, f in [('L1', L1), ('L2', L2), ('L2s', L2s), ('L3', L3), ('L3s', L3s),
                                                   ('L3a', L3a), ('L6', L6), ('L9', L9), ('L4a', L4a),
                                                   ('L4b', L4b), ('L5', L5), ('L7', L7)])))
