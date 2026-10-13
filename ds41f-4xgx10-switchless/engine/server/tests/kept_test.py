#!/usr/bin/env python3
"""Kept sessions (docs/engine.md): sessions survive a planned engine stop. Through rank 0's socket
(serve_test.py's client), in two phases with fixed session ids:
  --phase before   A: 64k positions cold, B: 20k, C: 2k, then A again with +300 (A is live at the stop);
                   the histories go to --state
  --phase after    each session's next turn: A +300 after its history (CED replays the decoder), B +40
                   (an exact continuation), C regenerated, D a fork of B at 10k; each one's TTFT (the
                   client's: submit to first token); then A's turn prompt prefilled cold in a fresh
                   session, for its TTFT. The turns go to --out.
  --phase probe    C regenerated alone: --expect cold (a start that must have started clean: nothing
                   reused) or warm (C came back)
The reference runs both phases against one engine; the test runs "before", a planned stop and a start
(SIGTERM to rank 0: the engine keeps its sessions, as an engine stop of the service does when it lends
the ring), then "after". --compare REF TEST checks the after-turns equal, bit for bit: tokens,
logprobs, the prefix each reused.

  python3 engine/server/tests/kept_test.py --socket S --phase before --state st.json
  python3 engine/server/tests/kept_test.py --socket S --phase after --state st.json --out test.json
  python3 engine/server/tests/kept_test.py --compare ref.json test.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, load_npy_i32, show  # noqa: E402

S0 = 0x6b65707400000000  # 'kept'
A, B, C, D, COLD = S0 + 1, S0 + 2, S0 + 3, S0 + 4, S0 + 0xfff


def turn(e, name, sid, prompt, rec, **kw):
    t0 = time.time()
    o = e.run(sid, prompt, max_new=24, n_top=2, **kw)
    o['ttft'] = o['t_first'] if o['t_first'] is not None else time.time() - t0
    show(name, o)
    print('%-28s TTFT %.3f s (server: %.1f ms to the first token)' % ('', o['ttft'], o['prefill_us'] / 1e3))
    rec[name] = {'ids': list(o['ids']), 'lp': list(o['lp']), 'cached': o['cached'], 'prompt': o['prompt'],
                 'reason': o['reason'], 'ttft': o['ttft'], 'server_ms': o['prefill_us'] / 1e3}
    return o


def before(e, T, n):
    rec = {}
    a1 = turn(e, 'A cold %d' % n, A, T[:n], rec)
    b1 = turn(e, 'B cold 20000', B, T[100000:120000], rec)
    c1 = turn(e, 'C cold 2000', C, T[150000:152000], rec)
    pa = T[:n] + list(a1['ids']) + T[n + 6000:n + 6300]
    a2 = turn(e, 'A +300', A, pa, rec)
    hist = {'A': pa + list(a2['ids']), 'B': T[100000:120000] + list(b1['ids']), 'C': T[150000:152000],
            'C_ids': list(c1['ids'])}
    return rec, hist


def after(e, T, n, hist):
    rec = {}
    pa = hist['A'] + T[n + 16000:n + 16300]
    turn(e, 'A +300 (after)', A, pa, rec)
    turn(e, 'B +40 (after)', B, hist['B'] + T[125000:125040], rec)
    turn(e, 'C regenerate (after)', C, hist['C'], rec)
    turn(e, 'D fork of B at 10000', D, T[100000:110000] + T[200000:200300], rec, base=B)
    turn(e, 'A +300 cold (TTFT)', COLD, pa, rec, keep=False)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--n', type=int, default=64000, help="session A's first prompt")
    ap.add_argument('--phase', choices=['before', 'after', 'probe'])
    ap.add_argument('--state', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--compare', nargs=2, default=None)
    ap.add_argument('--expect', choices=['cold', 'warm'], default=None)
    ap.add_argument('--release', action='store_true', help='release the sessions at the end')
    a = ap.parse_args()
    if a.compare:
        x, y = (json.load(open(f)) for f in a.compare)
        ok, names = 0, [k for k in x if 'cold' not in k]
        for name in names:
            keys = ('ids', 'lp', 'cached', 'prompt', 'reason')
            eq = name in y and all(x[name][k] == y[name][k] for k in keys)
            ok += eq
            print('  %-28s %s  TTFT %.3f s -> %.3f s' % (name, 'EQUAL' if eq else 'DIFFER: %s' % {
                k: (str(x[name][k])[:60], str(y.get(name, {}).get(k))[:60]) for k in keys if y.get(name, {}).get(k) != x[name][k]},
                x[name]['ttft'], y.get(name, {}).get('ttft', -1)))
        for name in (k for k in y if 'cold' in k):
            print('  %-28s TTFT %.3f s (reference %.3f s)' % (name, y[name]['ttft'], x.get(name, {}).get('ttft', -1)))
        print('kept_test --compare: %d of %d turns equal' % (ok, len(names)))
        return 0 if ok == len(names) else 1
    T = load_npy_i32(a.tokens)
    e = Engine(a.socket)
    print('engine: %s, context %d' % (e.name, e.max_ctx))
    if a.phase == 'before':
        rec, hist = before(e, T, a.n)
        if a.state:
            json.dump(hist, open(a.state, 'w'))
    elif a.phase == 'after':
        rec = after(e, T, a.n, json.load(open(a.state)))
    else:
        rec = {}
        turn(e, 'C probe', C, T[150000:152000], rec)
    if a.out:
        json.dump(rec, open(a.out, 'w'))
    if a.release:
        for s in (A, B, C, D):
            e.release(s)
    if a.expect:
        warm = [k for k, v in rec.items() if v['cached']]
        good = not warm if a.expect == 'cold' else len(warm) == len(rec)
        print('kept_test --expect %s: %s' % (a.expect, 'as expected' if good else 'NOT: cached %s' % {k: v['cached'] for k, v in rec.items()}))
        return 0 if good else 1
    bad = [k for k, v in rec.items() if v['reason'] not in ('length', 'stop')]
    print('kept_test --phase %s: %d turns%s' % (a.phase, len(rec), ', errors: %s' % bad if bad else ''))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
