#!/usr/bin/env python3
"""Time-slicing (docs/batching.md, B1) through rank 0's socket, with requests that overlap: each must give
what it gives alone, bit for bit (greedy and seeded sampling, tokens and log-probabilities); a request that
comes during another's decode gets its first token after its own prefill, not after the other's decode;
the decoding requests take turns; a cancel of a paused request, a second request for a session in flight
(held until the first ends) and a RELEASE of a session in flight (applied at its end).

  python3 engine/server/tests/slice_test.py --socket S              # the checks (a server with --slice > 0)
  python3 engine/server/tests/slice_test.py --socket S --ttft-only  # only the overlap's first-token times
                                                                    # (for a --slice 0 server, to compare)
  python3 engine/server/tests/slice_test.py --socket S --phase stop # two requests in flight; prints "ready",
                                                                    # then waits for the server's stop
"""
import argparse
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import CANCEL, DONE, REASONS, TOKENS, Engine, load_npy_i32  # noqa: E402

S0 = 0x736c696300000000  # 'slic'


class Mux(Engine):
    """Several requests at once on one connection: frames go to their request's record."""

    def __init__(self, path):
        super().__init__(path)
        self.recs = {}

    def start(self, name, session, prompt, **kw):
        # no stop ids: each request runs to its max_new (greedy X stops after 16 tokens at the end-of-text id,
        # before Y could come 30 tokens into its decode)
        kw.setdefault('stop', ())
        rid, n_top = self.submit(session, prompt, **kw)
        self.recs[rid] = {'name': name, 'n_top': n_top, 'ids': [], 'lp': [], 't_tok': [], 'done': None,
                          't_submit': time.time()}
        return rid

    def cancel(self, rid):
        self.send(CANCEL, struct.pack('<Q', rid))

    def pump(self, until, timeout=600):
        t_end = time.time() + timeout
        while not until():
            if time.time() > t_end:
                raise TimeoutError('slice_test: no end in %d s' % timeout)
            t, p = self.frame()
            if t == TOKENS:
                r, n, fl = struct.unpack('<QII', p[:16])
                rec = self.recs[r]
                rec['ids'] += struct.unpack('<%dI' % n, p[16:16 + 4 * n])
                if fl & 1:
                    rec['lp'] += struct.unpack('<%df' % n, p[16 + 4 * n:16 + 8 * n])
                rec['t_tok'] += [time.time()] * n
            elif t == DONE:
                (r, reason, pt, cached, comp, q_us, p_us, d_us, el, _) = struct.unpack('<QIIIIQQQII', p[:56])
                self.recs[r]['done'] = dict(reason=REASONS[reason], prompt=pt, cached=cached, completion=comp,
                                            queue_us=q_us, prefill_us=p_us, decode_us=d_us,
                                            error=p[56:56 + el].decode(), t=time.time())

    def ntok(self, rid):
        return len(self.recs[rid]['ids'])

    def finished(self, *rids):
        return all(self.recs[r]['done'] is not None for r in rids)


def ttft(rec):
    return rec['t_tok'][0] - rec['t_submit'] if rec['t_tok'] else float('nan')


fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def same(name, rec, ref):
    d = next((i for i, (a, b) in enumerate(zip(rec['ids'], ref['ids'])) if a != b), None)
    eq = rec['ids'] == ref['ids'] and rec['lp'] == ref['lp']
    check(eq, '%s equals its run alone (%d tokens, %s)%s' % (
        name, len(rec['ids']), rec['done']['reason'],
        '' if eq else ': first token difference at %s, log-probabilities %s' % (
            d, 'equal' if rec['lp'] == ref['lp'] else 'differ')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--phase', choices=['checks', 'stop'], default='checks')
    ap.add_argument('--ttft-only', action='store_true')
    a = ap.parse_args()
    T = load_npy_i32(a.tokens)
    e = Mux(a.socket)
    print('engine: %s, context %d' % (e.name, e.max_ctx))
    # X decodes long, Y (16k, sampled) comes during X's decode, Z (2k) during both
    X = dict(prompt=T[:1000], max_new=600, n_top=0)
    Y = dict(prompt=T[20000:36000], max_new=160, n_top=0, temp=0.7, seed=5)
    Z = dict(prompt=T[40000:42000], max_new=300, n_top=0)

    if a.phase == 'stop':
        r1 = e.start('P', S0 + 51, T[:1000], max_new=4000)
        e.pump(lambda: e.ntok(r1) >= 20)
        r2 = e.start('Q', S0 + 52, T[40000:42000], max_new=4000)
        e.pump(lambda: e.ntok(r2) >= 1)
        print('ready')
        sys.stdout.flush()
        e.pump(lambda: e.finished(r1, r2))
        for r in (r1, r2):
            d = e.recs[r]['done']
            print('stop: %s %s after %d tokens: %s' % (e.recs[r]['name'], d['reason'], e.ntok(r), d['error']))
        good = all(e.recs[r]['done']['reason'] == 'error' and 'stopping' in e.recs[r]['done']['error'] for r in (r1, r2))
        print('slice_test --phase stop: %s' % ('both ended with "the server is stopping"' if good else 'NOT as expected'))
        return 0 if good else 1

    ref = {}
    if not a.ttft_only:
        print('1. each alone, in a fresh session')
        for name, sid, kw in (('X', 1, X), ('Y', 2, Y), ('Z', 3, Z)):
            r = e.start(name, S0 + sid, kw['prompt'], **{k: v for k, v in kw.items() if k != 'prompt'})
            e.pump(lambda: e.finished(r))
            ref[name] = e.recs[r]
            d = ref[name]['done']
            print('  %s alone: prompt %d, %d tokens, %s; first token after %.3f s, decode %.1f tok/s' % (
                name, d['prompt'], d['completion'], d['reason'], ttft(ref[name]),
                (d['completion'] - 1) / max(1e-9, d['decode_us'] / 1e6)))

    print('2. overlapping: Y comes 30 tokens into X\'s decode, Z at Y\'s first token')
    rx = e.start('X', S0 + 11, X['prompt'], **{k: v for k, v in X.items() if k != 'prompt'})
    e.pump(lambda: e.ntok(rx) >= 30)
    ry = e.start('Y', S0 + 12, Y['prompt'], **{k: v for k, v in Y.items() if k != 'prompt'})
    e.pump(lambda: e.ntok(ry) >= 1)
    rz = e.start('Z', S0 + 13, Z['prompt'], **{k: v for k, v in Z.items() if k != 'prompt'})
    e.pump(lambda: e.finished(rx, ry, rz))
    for r in (rx, ry, rz):
        rec, d = e.recs[r], e.recs[r]['done']
        print('  %s: first token after %.3f s (queue %.1f ms, prefill %.1f ms), %d tokens, %s, done %.2f s after X began' % (
            rec['name'], ttft(rec), d['queue_us'] / 1e3, d['prefill_us'] / 1e3, d['completion'], d['reason'],
            d['t'] - e.recs[rx]['t_submit']))
    if a.ttft_only:
        return 0
    for r in (rx, ry, rz):
        same(e.recs[r]['name'], e.recs[r], ref[e.recs[r]['name']])
    y0, y1 = e.recs[ry]['t_tok'][0], e.recs[ry]['done']['t']
    x_during = sum(y0 < t < y1 for t in e.recs[rx]['t_tok'])
    check(x_during > 0, 'X decoded while Y was in flight: %d of its tokens' % x_during)
    lag = ttft(e.recs[ry]) - ttft(ref['Y'])
    check(lag < 1.0, "Y's first token %.3f s after its submit, %.3f s later than alone" % (ttft(e.recs[ry]), lag))
    x_left = e.recs[rx]['done']['t'] - y0
    print('  X still had %.2f s to go when Y\'s first token came: without time-slicing Y would have waited for it' % x_left)

    print('3. a paused request cancelled: X2 pauses for Z2, then its cancel ends it at the next turn')
    r1 = e.start('X2', S0 + 21, X['prompt'], max_new=2000, n_top=0)
    e.pump(lambda: e.ntok(r1) >= 30)
    r2 = e.start('Z2', S0 + 22, Z['prompt'], **{k: v for k, v in Z.items() if k != 'prompt'})
    e.pump(lambda: e.ntok(r2) >= 1)
    e.cancel(r1)
    t_c = time.time()
    e.pump(lambda: e.finished(r1, r2))
    d1 = e.recs[r1]['done']
    check(d1['reason'] == 'cancelled', 'X2 %s after %d tokens, %.2f s after its cancel' % (d1['reason'], e.ntok(r1), d1['t'] - t_c))
    check(e.recs[r1]['ids'] == ref['X']['ids'][:e.ntok(r1)], "X2's tokens are X's, as far as they go")
    same('Z2', e.recs[r2], ref['Z'])

    print('4. a second request for a session in flight waits for the first to end')
    r1 = e.start('A1', S0 + 31, T[60000:62000], max_new=200, n_top=0)
    e.pump(lambda: e.ntok(r1) >= 10)
    r2 = e.start('A2', S0 + 31, T[60000:62000], max_new=40, n_top=0)
    e.pump(lambda: e.finished(r1, r2))
    d1, d2 = e.recs[r1]['done'], e.recs[r2]['done']
    check(e.recs[r2]['t_tok'] and e.recs[r2]['t_tok'][0] > d1['t'] - 0.05,
          "A2's first token came after A1's end (%.2f s after; A2 queued %.1f ms)" % (
              (e.recs[r2]['t_tok'][0] - d1['t']) if e.recs[r2]['t_tok'] else -1, d2['queue_us'] / 1e3))
    check(d2['cached'] > 0 and d2['reason'] in ('length', 'stop'), 'A2 reused the session: cached %d, %s' % (d2['cached'], d2['reason']))

    print('5. a RELEASE of a session in flight applies at its end')
    r1 = e.start('B1', S0 + 41, T[70000:71000], max_new=300, n_top=0)
    e.pump(lambda: e.ntok(r1) >= 10)
    e.release(S0 + 41)
    e.pump(lambda: e.finished(r1))
    d1 = e.recs[r1]['done']
    check(d1['reason'] == 'length' and d1['completion'] == 300, 'B1 ran to its end: %d tokens, %s' % (d1['completion'], d1['reason']))
    r2 = e.start('B2', S0 + 41, T[70000:71000] + list(e.recs[r1]['ids']) + T[72000:72100], max_new=8, n_top=0)
    e.pump(lambda: e.finished(r2))
    check(e.recs[r2]['done']['cached'] == 0, 'the session was gone after B1: B2 cached %d' % e.recs[r2]['done']['cached'])

    for s in (1, 2, 3, 11, 12, 13, 21, 22, 31, 41):
        e.release(S0 + s)
    print('slice_test: %d failed' % fails)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
