#!/usr/bin/env python3
"""The session store's disk tier (docs/engine.md, Serving), through rank 0's socket (serve_test.py's
client). A session's turns, each after a filler request in another session; the same requests run on
two servers:
  - one whose store keeps everything (the default store, --disk-gib 0): the session waits in the store;
  - one whose store keeps nothing between requests (--session-gib 0.25: its budget is 0), with a disk
    tier: the filler's request spills the session, and its next turn reads it back.
--out writes each request's tokens, logprobs and restart point; --compare A B checks them equal, bit
for bit: a session back from disk continues exactly as if it had stayed. The turns: a long one (CED
replays the decoder), a short one (an exact continuation), a regenerate, a divergence far back (a window
snapshot), and a fork of the spilled session. The divergence and the fork restart inside the prompt
and add 300 positions, so they are also checked against the same prompt prefilled cold (CED: bit for
bit) in the same run. The +300 turn is compared with cold too, for information only: its history holds
decoded positions, whose roundings a cold prefill does not reproduce (docs/engine.md, Exact reuse).

  python3 engine/server/tests/disk_test.py --out store.json     # against a store-only server
  python3 engine/server/tests/disk_test.py --out disk.json      # against the spilling server
  python3 engine/server/tests/disk_test.py --compare store.json disk.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, load_npy_i32, same, show  # noqa: E402


def run(e, T, n):
    S = 0x5d15000000000000 + (n << 20)
    F = S + 0x80000
    k = [0]
    res, rec = [], {}

    def filler():  # another session's short request: the server parks (and may spill) the one before
        k[0] += 1
        f = e.run(F + k[0], T[200000 + 100 * k[0]:202000 + 100 * k[0]], max_new=2)
        e.release(F + k[0])
        return f

    def turn(name, sid, prompt, cold_check, **kw):
        filler()
        o = e.run(sid, prompt, max_new=24, n_top=2, **kw)
        show(name, o)
        rec[name] = {'ids': o['ids'], 'lp': o['lp'], 'cached': o['cached'], 'prompt': o['prompt']}
        if cold_check:
            c = e.run(S + 0xfff, prompt, max_new=24, n_top=2, keep=False)
            eq = same(name + ' vs cold' + (' (information)' if cold_check == 'info' else ''), o, c)
            if cold_check != 'info':
                res.append(eq)
        return o

    A = S + 1
    t1 = turn('cold %d' % n, A, T[:n], False)
    h = T[:n] + list(t1['ids'])
    p2 = h + T[n + 5000:n + 5300]
    t2 = turn('+300 (CED)', A, p2, 'info')
    p3 = p2 + list(t2['ids']) + T[n + 7000:n + 7040]
    turn('+40 (exact)', A, p3, False)
    turn('regenerate', A, p3, False)
    q = n - 8000
    turn('diverge %d back' % (n - q), A, T[:q] + T[n + 9000:n + 9300], True)
    q = n - 15000 if n > 20000 else n // 2
    turn('fork at %d' % q, A + 1, T[:q] + T[n + 11000:n + 11300], True, base=A)
    e.release(A)
    e.release(A + 1)
    return res, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--n', default='20000,64000', help='session lengths')
    ap.add_argument('--out', default='')
    ap.add_argument('--compare', nargs=2, default=None)
    a = ap.parse_args()
    if a.compare:
        x, y = (json.load(open(f)) for f in a.compare)
        ok = 0
        for name in x:
            eq = name in y and x[name] == y[name]
            ok += eq
            print('  %-34s %s' % (name, 'EQUAL' if eq else 'DIFFER: %s' % (
                {k: (x[name][k][:6], y.get(name, {}).get(k, [])[:6]) if isinstance(x[name][k], list) else (x[name][k], y.get(name, {}).get(k))
                 for k in x[name] if y.get(name, {}).get(k) != x[name][k]})))
        print('disk_test --compare: %d of %d requests equal' % (ok, len(x)))
        return 0 if ok == len(x) == len(y) else 1
    T = load_npy_i32(a.tokens)
    e = Engine(a.socket)
    print('engine: %s, context %d' % (e.name, e.max_ctx))
    res, rec = [], {}
    for n in (int(x) for x in a.n.split(',')):
        r, c = run(e, T, n)
        res += r
        rec.update({'%d: %s' % (n, k): v for k, v in c.items()})
    if a.out:
        json.dump(rec, open(a.out, 'w'))
    print('disk_test: %d of %d checks against cold pass; %d requests recorded' % (sum(res), len(res), len(rec)))
    return 0 if all(res) else 1


if __name__ == '__main__':
    sys.exit(main())
