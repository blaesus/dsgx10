#!/usr/bin/env python3
"""Window snapshots thinned with the distance from a session's end (serve.cpp keep_snap), through rank
0's socket (serve_test.py's client). A session of n positions, then forks of it (base session) at many
distances d back, each with new tokens after the fork point:
  - each fork against the same prompt prefilled cold in a fresh session: with --ced (the server's
    default) bit for bit (greedy tokens and the first token's logprob; the decoder replayed), without it
    the server must run --exact and serve_test's compare() applies (a short remainder runs through the
    step, so a near-tie may part);
  - the position it restarted from (cached) against the snapshots keep_snap keeps: at most d / 3 back
    from the fork point, and at most snap_every inside the dense last 8 * snap_every;
  - a short fork far back (fewer than 128 new positions) too, a divergence of the session itself, and
    a turn at that divergence's prompt end.

  python3 engine/server/tests/snap_test.py [--ced] [--n 40000]
  python3 engine/server/tests/snap_test.py --far 131072,250000 [--snap-all]   # timing only: forks far
      back from long sessions (prefill ms against the positions re-prefilled), no cold runs
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, compare, load_npy_i32, same, show  # noqa: E402


def keep_snap(s, E, every=1024):
    """serve.cpp's keep_snap."""
    d, dense = E - s, 8 * every
    if d < dense:
        return True
    g, lim = 2 * every, 2 * dense
    while d >= lim:
        g, lim = 2 * g, 2 * lim
    return s % g == 0


def bound(q, E, every=1024, thin=True):
    """The latest kept snapshot at or below q (the restart the planner should find; even positions)."""
    s = q // every * every
    while s >= 128 and thin and not keep_snap(s, E, every):
        s -= every
    return s if s >= 128 else 0


def lcp(x, y):
    i = 0
    while i < min(len(x), len(y)) and x[i] == y[i]:
        i += 1
    return i


def correctness(e, T, a):
    S = 0x5a10000000000000
    n = a.n
    res = []
    eq = same if a.ced else compare
    base = e.run(S + 1, T[:n], max_new=8, n_top=2)
    show('cold %d' % n, base)
    E = n + len(base['ids'])  # the history: the prompt, then what was decoded (at most this)
    off = n + 20000
    k = 2
    for d in (300, 2500, 7000, 9500, 15000, 21000, 33000, n - 3000):
        if d >= n - 200:
            continue
        q = n - d
        p = T[:q] + T[off:off + 300]
        off += 1000
        f = e.run(S + k, p, max_new=16, n_top=2, base=S + 1)
        show('fork %d back (at %d)' % (d, q), f)
        pq = lcp(T[:n], p)  # the common prefix (a new token may happen to match)
        want = [bound(pq, x, thin=not a.snap_all) for x in (E - 1, E)]
        ok_q = f['cached'] in want
        print('  restart at %d (expected %s): %d positions re-prefilled before the fork point; %s' % (
            f['cached'], want[-1], q - f['cached'], 'ok' if ok_q else 'NOT WHERE EXPECTED'))
        res.append(ok_q)
        c = e.run(S + k + 1, p, max_new=16, n_top=2, keep=False)
        show('  same prompt cold', c)
        res.append(eq('fork at %d vs cold' % q, f, c))
        e.release(S + k)
        k += 2
    # a short fork far back: fewer than 128 new positions; it restarts where it can replay the decoder
    q = n - 25000 if n > 30000 else n // 2
    p = T[:q] + T[off:off + 50]
    f = e.run(S + k, p, max_new=16, n_top=2, base=S + 1)
    show('short fork at %d' % q, f)
    c = e.run(S + k + 1, p, max_new=16, n_top=2, keep=False)
    show('  same prompt cold', c)
    res.append(eq('short fork vs cold', f, c))
    e.release(S + k)
    # the session itself diverging far back (it drops what follows), then a turn at that prompt's end
    # (its output dropped, as an agent drops its reasoning: every position came from the prefill)
    q = n - 18000 if n > 25000 else n // 3
    p = T[:q] + T[off + 1000:off + 1400]
    g = e.run(S + 1, p, max_new=16, n_top=2)
    show('diverge at %d' % q, g)
    c = e.run(S + k + 2, p, max_new=16, n_top=2, keep=False)
    show('  same prompt cold', c)
    res.append(eq('divergence vs cold', g, c))
    p2 = p + T[off + 2000:off + 2300]
    h = e.run(S + 1, p2, max_new=16, n_top=2)
    show('then at its prompt end (+300)', h)
    c = e.run(S + k + 3, p2, max_new=16, n_top=2, keep=False)
    show('  same prompt cold', c)
    res.append(h['cached'] >= len(p) - 1 and eq('prompt-end turn vs cold', h, c))
    e.release(S + 1)
    print('snap_test: %d of %d checks pass' % (sum(res), len(res)))
    return all(res)


def far(e, T, a):
    S = 0x5a20000000000000
    for i, n in enumerate(int(x) for x in a.far.split(',')):
        s = S + 64 * i
        t0 = time.time()
        b = e.run(s, T[:n], max_new=8)
        show('cold %d' % n, b)
        E = n + len(b['ids'])
        k = 1
        for d in (1000, 8000, 20000, 50000, 100000, 200000):
            if d >= n - 1000:
                continue
            q = n - d
            p = T[:q] + T[n + 1000 * k:n + 1000 * k + 300]
            f = e.run(s + k, p, max_new=8, base=s)
            print('far\t%d\t%d\t%d\t%d\t%.1f\t%.1f' % (n, d, f['cached'], q + 300 - f['cached'], f['prefill_us'] / 1e3,
                                                       1e3 * (f['t_first'] or 0)))
            show('  fork %d back' % d, f)
            e.release(s + k)
            k += 1
        e.release(s)
        print('n %d: %.1f s' % (n, time.time() - t0))
        sys.stdout.flush()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--n', type=int, default=40000)
    ap.add_argument('--ced', action='store_true', help='the server runs CED (its default); else --exact')
    ap.add_argument('--snap-all', action='store_true', help='the server keeps every snapshot (--snap-all)')
    ap.add_argument('--far', default='', help='sessions of these lengths, forks far back: timing only')
    a = ap.parse_args()
    T = load_npy_i32(a.tokens)
    e = Engine(a.socket)
    print('engine: %s, context %d' % (e.name, e.max_ctx))
    if a.far:
        return 0 if far(e, T, a) else 1
    return 0 if correctness(e, T, a) else 1


if __name__ == '__main__':
    sys.exit(main())
