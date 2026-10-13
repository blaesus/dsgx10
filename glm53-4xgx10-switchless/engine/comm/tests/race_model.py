#!/usr/bin/env python3
"""A model check of the pipelined ring kernels' send-buffer reuse inside their grid (comm/collectives.cu,
kernels/prefill.cu), from the audit in docs/ring.md's send-buffer race section. Every block runs the
same op list; post(s) counts the block in, and wait(s) (wait_all or wait_posted) passes once every
block has counted in. A read R of a slot's old data by block B and a later write W of that slot by
block A are ordered iff R precedes B's post(s) and W follows A's wait(s) for some s (same block: a
barrier between them). Units: the 16-byte units of a send buffer; a unit's block is the block of the
thread that touches it, from the kernels' index arithmetic. Reports every unordered overlapping
(W, R) pair, and RAW: a read of s's data not after every block's writes of s. Runs in ~1 s:
  python3 engine/comm/tests/race_model.py"""
import sys

def rs_like(n, lens, fix):  # reduce_scatter_bulk_kernel / alltoall_rows_kernel / k_moe_combine_rs
    ops = [('w', 0, 0), ('post', 0)]
    if n > 1: ops.append(('w', 1, 1))
    for i in range(n):
        ops.append(('wait', i))
        if i + 1 < n: ops.append(('post', i + 1))
        ops.append(('r', i, i))
        if i + 2 < n:
            if fix and i > 0: ops.append(('wait', i + 1))
            ops.append(('w', i + 2, i + 2))
    return ops

def ag_multi(n, lens, fix):  # allgather_multi_kernel
    ops = [('w', 0, 0), ('post', 0)]
    if n > 1: ops.append(('w', 1, 1))
    for j in range(n):
        ops.append(('wait', j))
        if j + 1 < n: ops.append(('post', j + 1))
        ops.append(('r', j, j))
        if j + 2 < n:
            if fix and j > 0: ops.append(('wait', j + 1))
            ops.append(('w', j + 2, j + 2))
    return ops

def ag_bulk(n, lens, fix):  # allgather_bulk_kernel
    ops = [('w', 0, 0), ('post', 0)]
    for j in range(n):
        ops.append(('wait', j))
        if j + 1 < n: ops += [('w', j + 1, j + 1), ('post', j + 1)]
        ops.append(('r', j, j))
    return ops

def ar_bulk(n, lens, fix):  # allreduce_bulk_kernel: chunk i = RS seq 2i (data 'c'), AG seq 2i+1 (data 'q')
    ops = [('w', 0, ('c', 0)), ('post', 0), ('wait', 0), ('r', 0, ('c', 0)), ('w', 1, ('q', 0)), ('post', 1)]
    for i in range(n):
        s = 2 * i
        ops.append(('wait', s + 1))
        if i + 1 < n: ops += [('w', s + 2, ('c', i + 1)), ('post', s + 2)]
        ops.append(('r', s + 1, ('q', i)))
        if i + 1 < n: ops += [('wait', s + 2), ('r', s + 2, ('c', i + 1)), ('w', s + 3, ('q', i + 1)), ('post', s + 3)]
    return ops

def units(kind, op, data, lens, rank, stride, bdim):
    """[(unit, block)]: the units a block writes (op w) or reads in the own buffer (op r)"""
    blk = lambda t: (t % stride) // bdim
    if kind == 'rs':
        pl = lens[data]
        if op == 'w':
            return [(k * pl + j, blk(j)) for k in range(4) for j in range(pl)]
        return [(rank * pl + j, blk(j)) for j in range(pl)]
    if kind in ('agm', 'agb'):
        return [(u, blk(u)) for u in range(lens[data])]
    if kind == 'arb':
        t, i = data
        q = lens[i] // 4
        if t == 'c':
            if op == 'w':
                return [(u, blk(u)) for u in range(lens[i])]
            return [(rank * q + j, blk(j)) for j in range(q)]
        return [(u, blk(u)) for u in range(q)]

def check(kind, prog, n, lens, fix, stride, bdim):
    ops = prog(n, lens, fix)
    pos_post = {o[1]: k for k, o in enumerate(ops) if o[0] == 'post'}
    pos_wait = {}
    for k, o in enumerate(ops):
        if o[0] == 'wait': pos_wait.setdefault(o[1], []).append(k)
    def hb(kr, kw):  # read at kr (any block) before write at kw (another block)
        return any(kr < pos_post[s] and any(kw > kk for kk in pos_wait.get(s, [])) for s in pos_post)
    bad = []
    rw = [(k, o) for k, o in enumerate(ops) if o[0] in ('r', 'w')]
    for (kw, w) in rw:
        if w[0] != 'w': continue
        for (kr, r) in rw:
            if r[0] != 'r' or r[1] % 3 != w[1] % 3: continue
            for rank in range(4):
                W = units(kind, 'w', w[2], lens, rank, stride, bdim)
                Rr = units(kind, 'r', r[2], lens, rank, stride, bdim)
                if r[1] < w[1]:  # WAR on the slot's previous data: R must precede W
                    if hb(kr, kw): continue
                    m = dict(Rr)
                    hits = [(u, b, m[u]) for u, b in W if u in m and m[u] != b]
                    if hits: bad.append(('WAR', w, r, rank, len(hits), hits[0]))
                elif r[1] == w[1]:  # RAW on s's data: R after every block's writes (W before post(s), R after wait(s))
                    s = w[1]
                    ok = kw < pos_post[s] and any(kr > kk for kk in pos_wait.get(s, []))
                    if not ok: bad.append(('RAW', w, r, rank))
    return bad

if __name__ == '__main__':
    # rs: reduce_scatter_bulk_kernel, alltoall_rows_kernel and k_moe_combine_rs (one op order); agm:
    # allgather_multi_kernel; agb: allgather_bulk_kernel; arb: allreduce_bulk_kernel. fix: the waits of
    # comm::wait_posted. 1-8 pieces (the 3 slots wrapped more than twice), last pieces full, 1 unit, and
    # multiples and non-multiples of the grid's threads, in 4 grid shapes (threads in all, per block).
    # Exit 0 iff the unfixed rs order shows the race (the model's self-check) and nothing else does.
    P = 64  # units of a full piece (the engine's are multiples of the grid's threads too)
    ok = True
    for stride, bdim in ((16, 4), (8, 2), (32, 8), (64, 16)):
        out = []
        for kind, prog in (('rs', rs_like), ('agm', ag_multi), ('agb', ag_bulk), ('arb', ar_bulk)):
            for fix in (False, True):
                tot, bad = 0, []
                for n in range(1, 9):
                    for last in sorted({P, 1, 3, 5, 8, 16, 17, 24, 31, 32, 33, 40, 48, 63}):
                        if kind == 'arb' and last % 4: continue
                        tot += 1
                        if check(kind, prog, n, [P] * (n - 1) + [last], fix, stride, bdim): bad.append((n, last))
                expect = kind == 'rs' and not fix
                ok &= bool(bad) == expect and all(n >= 4 and last % stride for n, last in bad)
                out.append('%s%s %d/%d%s' % (kind, '+fix' if fix else '', len(bad), tot,
                                             ' (first %s)' % bad[:2] if bad else ''))
        print('%2d threads, %2d a block: %s' % (stride, bdim, '; '.join(out)))
    print('as expected: only the unfixed rs order, at 4+ pieces with a last piece not a multiple of the grid'
          if ok else 'NOT as expected')
    sys.exit(0 if ok else 1)
