#!/usr/bin/env python3
"""Agentic turns through a server: turns.py [--sock PATH] [--tokens NPY] [n ...]
For each n (default 16384 65536 131072 250000): a cold prompt of n tokens, then turns appended to the same
session, each the last prompt + its output + new tokens (+200, +2000, +8000), then another session's turn of
1000 in between, and +2000 (the session's state loaded from the store). The tokens are the token file's from
the start (default golden/synth270k, 270,000 tokens). Prints one row per request: rid, label, prompt, cached,
the client's time to first token and total ms, the server's prefill ms, tokens out (tab-separated, `row`
first; tjoin.py joins them with the server's turn lines). --sock: the server's socket (the test server's
default, /tmp/dsgb10-engine.sock)."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'tests'))
from serve_test import Engine, load_npy_i32  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--sock', default='/tmp/dsgb10-engine.sock')
ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
ap.add_argument('n', type=int, nargs='*')
args = ap.parse_args()
T = load_npy_i32(args.tokens)
e = Engine(args.sock)
print('engine: %s, context %d' % (e.name, e.max_ctx))
ns = args.n or [16384, 65536, 131072, 250000]
S = 0x10d0000000000000


def row(label, prompt, o):
    print('row\t%d\t%s\t%d\t%d\t%.1f\t%.1f\t%.1f\t%d' % (e.next_id - 1, label, len(prompt), o['cached'],
                                                       1e3 * (o['t_first'] or 0), 1e3 * o['t_total'],
                                                       o['prefill_us'] / 1e3, len(o['ids'])))
    sys.stdout.flush()


for i, n in enumerate(ns):
    if n + 12200 > len(T):
        sys.exit('turns.py: %d tokens and the turns need %d of the token file, which has %d' % (n, n + 12200, len(T)))
    s = S + 16 * i
    t0 = time.time()
    prompt = T[:n]
    off = n
    o = e.run(s, prompt, max_new=4)
    row('cold %d' % n, prompt, o)
    prompt = prompt + o['ids']
    for k in (200, 2000, 8000):
        prompt = prompt + T[off:off + k]
        off += k
        o = e.run(s, prompt, max_new=4)
        row('%d +%d' % (n, k), prompt, o)
        prompt = prompt + o['ids']
    o = e.run(s + 1, T[:1000], max_new=2, keep=False)
    row('other 1000', T[:1000], o)
    prompt = prompt + T[off:off + 2000]
    off += 2000
    o = e.run(s, prompt, max_new=4)
    row('%d load +2000' % n, prompt, o)
    e.release(s)
    print('n %d: %.1f s' % (n, time.time() - t0))
    sys.stdout.flush()
