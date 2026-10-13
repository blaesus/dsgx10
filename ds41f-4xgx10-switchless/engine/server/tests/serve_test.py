#!/usr/bin/env python3
"""Rank 0's server over its Unix socket (docs/protocol.md), without the front-end: cold prefill, an
exact continuation, a divergence rebuilt from a window snapshot, a fork from a base session, a stored
session resumed, a regenerate, seeded sampling with logprobs, cancel, release. Every reuse case is
compared with the same prompt prefilled cold in a fresh session (greedy tokens and the first token's
logprob). Prompts are slices of a golden token sequence.

  python3 engine/server/tests/serve_test.py [--socket /tmp/dsgb10-engine.sock] [--tokens tokens.npy]
  python3 engine/server/tests/serve_test.py --speed      # cold prefill 1k..32k and decode tok/s
  python3 engine/server/tests/serve_test.py --ced        # CED's turn rule (the server's default)

The reuse checks without --ced need an exact server (dsgb10_rank --mode serve --exact). With --ced the
cases follow CED's turn rule (docs/engine.md, Serving): a turn of at least 128 new positions replays
the decoder and must equal the same prompt prefilled cold, bit for bit; a shorter one continues
exactly and must equal the same history replayed in a fresh session.
"""
import argparse
import array
import ast
import socket
import struct
import sys
import time

HELLO, SUBMIT, CANCEL, RELEASE = 0x01, 0x02, 0x03, 0x04
HELLO_E, TOKENS, DONE = 0x81, 0x82, 0x83
F_SEED, F_KEEP, F_LOGPROBS = 1, 2, 4
REASONS = ['stop', 'length', 'cancelled', 'context', 'error']


def load_npy_i32(fn):
    b = open(fn, 'rb').read()
    hl = struct.unpack('<H', b[8:10])[0]
    h = ast.literal_eval(b[10:10 + hl].decode())
    assert h['descr'] == '<i4', h
    a = array.array('i')
    a.frombytes(b[10 + hl:])
    return list(a)


class Engine:
    def __init__(self, path):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.connect(path)
        self.buf = b''
        t, p = self.frame()
        assert t == HELLO_E, t
        ver, self.max_ctx, vocab, eos, max_sess, nl = struct.unpack('<6I', p[:24])
        self.name = p[24:24 + nl].decode()
        self.send(HELLO, struct.pack('<II', 1, 0))
        self.next_id = 1

    def send(self, t, payload):
        self.s.sendall(struct.pack('<IHH', len(payload), t, 0) + payload)

    def recv_exact(self, n):
        while len(self.buf) < n:
            d = self.s.recv(1 << 20)
            if not d:
                raise EOFError('engine closed the socket')
            self.buf += d
        r, self.buf = self.buf[:n], self.buf[n:]
        return r

    def frame(self):
        size, t, _ = struct.unpack('<IHH', self.recv_exact(8))
        return t, self.recv_exact(size)

    def submit(self, session, prompt, max_new=32, base=0, temp=0.0, top_p=1.0, top_k=0, seed=None, keep=True,
               n_top=None, stop=(1,)):
        rid = self.next_id
        self.next_id += 1
        flags = (F_KEEP if keep else 0) | (F_SEED if seed is not None else 0) | (F_LOGPROBS if n_top is not None else 0)
        hdr = struct.pack('<QQQQffIIIIIIII', rid, session, base, seed or 0, temp, top_p, top_k, max_new, flags,
                          n_top or 0, 0, len(stop), len(prompt), 0)
        assert len(hdr) == 72
        self.send(SUBMIT, hdr + struct.pack('<%dI' % len(stop), *stop) + struct.pack('<%dI' % len(prompt), *prompt))
        return rid, n_top or 0

    def collect(self, req, cancel_after=None):
        rid, n_top = req
        out = {'ids': [], 'lp': [], 'top': [], 't_first': None}
        t0 = time.time()
        cancelled = False
        while True:
            t, p = self.frame()
            if t == TOKENS:
                r, n, fl = struct.unpack('<QII', p[:16])
                assert r == rid, (r, rid)
                out['ids'] += struct.unpack('<%dI' % n, p[16:16 + 4 * n])
                off = 16 + 4 * n
                if fl & 1:
                    out['lp'] += struct.unpack('<%df' % n, p[off:off + 4 * n])
                    off += 4 * n
                if fl & 2:
                    pairs = struct.unpack('<' + 'If' * (n * n_top), p[off:off + 8 * n * n_top])
                    for i in range(n):
                        out['top'].append(list(zip(pairs[2 * n_top * i::2][:n_top], pairs[2 * n_top * i + 1::2][:n_top])))
                if out['t_first'] is None:
                    out['t_first'] = time.time() - t0
                if cancel_after is not None and not cancelled and len(out['ids']) >= cancel_after:
                    self.send(CANCEL, struct.pack('<Q', rid))
                    cancelled = True
            elif t == DONE:
                (r, reason, pt, cached, comp, q_us, p_us, d_us, el, _) = struct.unpack('<QIIIIQQQII', p[:56])
                assert r == rid, (r, rid)
                out.update(reason=REASONS[reason], prompt=pt, cached=cached, completion=comp, queue_us=q_us,
                           prefill_us=p_us, decode_us=d_us, error=p[56:56 + el].decode(), t_total=time.time() - t0)
                return out

    def run(self, *a, cancel_after=None, **k):
        return self.collect(self.submit(*a, **k), cancel_after=cancel_after)

    def release(self, session):
        self.send(RELEASE, struct.pack('<Q', session))


def show(name, o):
    print('%-28s %-9s prompt %6d cached %6d  prefill %8.1f ms  %4d tokens %7.1f ms (%5.1f tok/s)  %s' % (
        name, o['reason'], o['prompt'], o['cached'], o['prefill_us'] / 1e3, o['completion'], o['decode_us'] / 1e3,
        max(0, o['completion'] - 1) / max(1e-9, o['decode_us'] / 1e6), o['error']))
    sys.stdout.flush()


def compare(name, a, b):
    """Greedy outputs of a reuse case (a) and the same prompt cold (b). Positions the reuse case took
    from its history may have been computed by the decode step, whose roundings differ slightly from the
    prefill's: a difference passes as a near-tie when, at the first differing token, both runs' top two
    are within 0.05 in logprob and the first token's logprob agrees to 1e-3."""
    same = 0
    while same < min(len(a['ids']), len(b['ids'])) and a['ids'][same] == b['ids'][same]:
        same += 1
    lp = abs(a['lp'][0] - b['lp'][0]) if a['lp'] and b['lp'] else float('nan')
    ok = same == len(a['ids']) == len(b['ids'])
    note = ''
    if not ok and same < len(a['top']) and same < len(b['top']):
        ta, tb = a['top'][same], b['top'][same]
        ma, mb = ta[0][1] - ta[1][1], tb[0][1] - tb[1][1]
        note = '; at token %d: %s vs %s (margins %.4f, %.4f)' % (same, ta[:2], tb[:2], ma, mb)
        if min(ma, mb) < 0.05 and lp < 1e-3:
            ok, note = True, note + ', a near-tie'
    print('  %-26s %s: %d/%d tokens equal, first logprob %.6f vs %.6f (|diff| %.2e)%s' % (
        name, 'EQUAL' if same == len(a['ids']) == len(b['ids']) else 'DIFFER', same, max(len(a['ids']), len(b['ids'])),
        a['lp'][0] if a['lp'] else 0, b['lp'][0] if b['lp'] else 0, lp, note))
    return ok


def close(name, a, b, tol=2e-3):
    """A restart that reruns prefilled positions through the step path (a regenerate reruns the prompt's
    last positions, a remainder of at most --small-rows) matches the prefill path to numerics, not bits:
    the same first token, its logprob within tol. The tokens after it may part where the top two are
    close, so their agreement is information only."""
    same = 0
    while same < min(len(a['ids']), len(b['ids'])) and a['ids'][same] == b['ids'][same]:
        same += 1
    lp = abs(a['lp'][0] - b['lp'][0]) if a['lp'] and b['lp'] else float('nan')
    ok = bool(a['ids']) and a['ids'][:1] == b['ids'][:1] and lp <= tol
    print('  %-26s %s: first token %s, first logprob %.6f vs %.6f (|diff| %.2e, at most %.0e); %d/%d tokens equal' % (
        name, 'CLOSE' if ok else 'DIFFER', 'equal' if a['ids'][:1] == b['ids'][:1] else 'differs',
        a['lp'][0] if a['lp'] else 0, b['lp'][0] if b['lp'] else 0, lp, tol, same, max(len(a['ids']), len(b['ids']))))
    return ok


def scenarios(e, T):
    S = 0x5e55000000000000
    res = []
    # 1. cold, then an exact continuation (the generated ids and new text appended)
    p1 = T[0:3000]
    a = e.run(S + 1, p1, max_new=32, n_top=2)
    show('cold 3000', a)
    hist = p1 + a['ids']
    p2 = hist + T[3000:3500]
    b = e.run(S + 1, p2, max_new=32, n_top=2)
    show('continue (+500)', b)
    c = e.run(S + 2, p2, max_new=32, n_top=2, keep=False)
    show('  same prompt cold', c)
    compare('continuation vs cold', b, c)  # positions 3000..3031 came from the decode step: information only
    cont = b
    # 2. divergence at 2500: the snapshot at 2048, then [2048, n) again
    p3 = T[0:2500] + T[5000:5600]
    b = e.run(S + 1, p3, max_new=32, n_top=2)
    show('diverge at 2500', b)
    c = e.run(S + 3, p3, max_new=32, n_top=2, keep=False)
    show('  same prompt cold', c)
    res.append(compare('divergence (snapshot)', b, c))
    # 3. a fork: a new session from session 1 (live) up to their common prefix
    p4 = T[0:2200] + T[7000:7300]
    b = e.run(S + 4, p4, base=S + 1, max_new=32, n_top=2)
    show('fork from live base', b)
    c = e.run(S + 5, p4, max_new=32, n_top=2, keep=False)
    show('  same prompt cold', c)
    res.append(compare('fork (live base)', b, c))
    # 4. session 1 is stored now: resume it (its end snapshot), then fork from a stored base
    p5 = p3 + T[9000:9100]
    b = e.run(S + 1, p5, max_new=32, n_top=2)
    show('stored, diverge at its end', b)
    c = e.run(S + 6, p5, max_new=32, n_top=2, keep=False)
    show('  same prompt cold', c)
    res.append(compare('stored session', b, c))
    p6 = T[0:4200] + T[12000:12100]
    b = e.run(S + 7, p6, base=S + 4, max_new=32, n_top=2)
    show('fork from stored base', b)
    c = e.run(S + 8, p6, max_new=32, n_top=2, keep=False)
    show('  same prompt cold', c)
    res.append(compare('fork (stored base)', b, c))
    # 5. regenerate: the same prompt again on the same session (the prompt-end snapshot); its last
    # positions run through the step this time, so it matches the cold prefill to numerics
    b2 = e.run(S + 7, p6, max_new=32, n_top=2)
    show('regenerate', b2)
    res.append(close('regenerate', b2, b))
    # 6. seeded sampling twice, with logprobs
    s1 = e.run(S + 9, T[0:1500], max_new=64, temp=0.8, top_p=0.95, seed=1234, n_top=3, keep=False)
    show('sampled (seed 1234)', s1)
    s2 = e.run(S + 10, T[0:1500], max_new=64, temp=0.8, top_p=0.95, seed=1234, n_top=3, keep=False)
    show('  again', s2)
    ok = s1['ids'] == s2['ids']
    print('  %-26s %s; top-3 of the first token: %s' % ('seeded sampling', 'EQUAL' if ok else 'DIFFER', s1['top'][:1]))
    res.append(ok)
    # 7. cancel during decode, then the session continues
    x = e.run(S + 11, T[0:1200], max_new=4000, stop=(), cancel_after=20)
    show('cancel after 20', x)
    res.append(x['reason'] == 'cancelled')
    y = e.run(S + 11, T[0:1200] + x['ids'][:10] + T[20000:20050], max_new=16)
    show('  then continue', y)
    res.append(y['reason'] in ('length', 'stop') and y['cached'] >= 1200)
    # 8. the continuation's history replayed in a fresh session: the first request, its 32 tokens
    # decoded, then p2 (positions 3000..3031 from the decode step on both sides)
    r1 = e.run(S + 12, p1, max_new=32, n_top=2)
    r2 = e.run(S + 12, p2, max_new=32, n_top=2)
    show('continuation replayed', r2)
    res.append(r1['ids'] == a['ids'] and compare('continuation vs replay', cont, r2))
    for s in range(1, 13):
        e.release(S + s)
    print('serve_test: %d of %d checks pass' % (sum(res), len(res)))
    return all(res)


def same(name, a, b):
    """Bit-identical greedy outputs: every token and the first token's logprob."""
    ok = a['ids'] == b['ids'] and a['lp'][:1] == b['lp'][:1]
    print('  %-26s %s (cached %d vs %d)' % (name, 'EQUAL' if ok else 'DIFFER: %s vs %s, lp %s vs %s' % (
        a['ids'][:8], b['ids'][:8], a['lp'][:1], b['lp'][:1]), a['cached'], b['cached']))
    return ok


def ced_scenarios(e, T):
    S = 0x5ced000000000000
    res = []

    def cold(sid, prompt):
        o = e.run(sid, prompt, max_new=32, n_top=2, keep=False)
        show('  same prompt cold', o)
        return o

    # 1. a CED turn, then a short one (60 new positions): an exact continuation from the live end
    p1 = T[0:3000]
    a = e.run(S + 1, p1, max_new=32, n_top=2)
    show('cold 3000 (CED)', a)
    p2 = p1 + a['ids'] + T[3000:3060]
    b = e.run(S + 1, p2, max_new=32, n_top=2)
    show('short turn (+60)', b)
    res.append(len(p1) <= b['cached'] and len(p2) - b['cached'] < 128)
    r1 = e.run(S + 2, p1, max_new=32, n_top=2)
    r2 = e.run(S + 2, p2, max_new=32, n_top=2)
    show('  replayed in a fresh session', r2)
    res.append(r1['ids'] == a['ids'] and same('short turn vs replay', b, r2))
    # 2. a CED turn, then a long one from the live ring (nothing decoded in between, so every position
    # came from the prefill): appended at the end, or diverging inside the ring; the encoder from there,
    # the decoder replayed: the same as cold
    for name, p3, q3 in (('long turn appended', T[0:3300], 3000), ('long turn from the ring', T[0:2950] + T[6000:6400], 2950)):
        e.run(S + 3, T[0:3000], max_new=1)
        b = e.run(S + 3, p3, max_new=32, n_top=2)
        show(name, b)
        res.append(b['cached'] == q3)
        res.append(same(name + ' vs cold', b, cold(S + 15, p3)))
        e.release(S + 3)
    # 3. snapshots inside a CED prefill hold the encoder only: a long turn may restart there, a short
    # one may not (2048 would leave 112 positions: 1024 instead)
    e.run(S + 4, T[0:3000], max_new=8)
    p4 = T[0:2500] + T[5000:5600]
    b = e.run(S + 4, p4, max_new=32, n_top=2)
    show('encoder snapshot, long', b)
    res.append(b['cached'] == 2048)
    res.append(same('encoder snapshot vs cold', b, cold(S + 5, p4)))
    e.run(S + 6, T[0:3000], max_new=8)
    p5 = T[0:2100] + T[5000:5060]
    b = e.run(S + 6, p5, max_new=32, n_top=2)
    show('encoder snapshot, short', b)
    res.append(b['cached'] == 1024)
    res.append(same('short restart vs cold', b, cold(S + 7, p5)))
    # 4. forks from a CED base: a long one equals cold; a short one at the base's prompt end continues
    # exactly, as the base itself would
    e.run(S + 8, T[0:3000], max_new=32)
    p6 = T[0:2980] + T[7000:7300]
    b = e.run(S + 9, p6, base=S + 8, max_new=32, n_top=2)
    show('long fork', b)
    res.append(b['cached'] == 2980)
    res.append(same('long fork vs cold', b, cold(S + 10, p6)))
    p7 = T[0:3000] + T[7000:7050]
    b = e.run(S + 11, p7, base=S + 8, max_new=32, n_top=2)
    show('short fork', b)
    res.append(b['cached'] == 3000)
    e.run(S + 12, T[0:3000], max_new=32)
    r = e.run(S + 12, p7, max_new=32, n_top=2)
    show('  the base continued instead', r)
    res.append(same('short fork vs continuation', b, r))
    # 5. a CED prefill cancelled between chunks leaves no valid decoder rings: a short turn after it
    # restarts where it can replay (n - q >= 128), and equals cold
    req = e.submit(S + 13, T[0:10000], max_new=8)
    time.sleep(0.5)
    e.send(CANCEL, struct.pack('<Q', req[0]))
    x = e.collect(req)
    show('cancelled CED prefill', x)
    res.append(x['reason'] == 'cancelled')
    p8 = T[0:4096] + T[20000:20050]
    b = e.run(S + 13, p8, max_new=32, n_top=2)
    show('short turn after it', b)
    res.append(b['cached'] <= len(p8) - 128)
    res.append(same('after the cancel vs cold', b, cold(S + 14, p8)))
    for s in range(1, 16):
        e.release(S + s)
    print('serve_test --ced: %d of %d checks pass' % (sum(res), len(res)))
    return all(res)


def speed(e, T):
    S = 0x5bee000000000000
    for n in (1024, 4096, 8192, 16384, 32768):
        if n >= len(T):
            break
        o = e.run(S + n, T[:n], max_new=128, stop=(), keep=False)
        print('cold %6d: prefill %8.1f ms (%6.0f tok/s), decode %5.1f tok/s over %d tokens' % (
            n, o['prefill_us'] / 1e3, n / (o['prefill_us'] / 1e6), (o['completion'] - 1) / (o['decode_us'] / 1e6),
            o['completion']))
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/long/tokens.npy')
    ap.add_argument('--speed', action='store_true')
    ap.add_argument('--ced', action='store_true', help="CED's turn rule (the server's default; else --exact)")
    a = ap.parse_args()
    T = load_npy_i32(a.tokens)
    e = Engine(a.socket)
    print('engine: %s, context %d' % (e.name, e.max_ctx))
    if a.speed:
        speed(e, T)
        return 0
    return 0 if (ced_scenarios(e, T) if a.ced else scenarios(e, T)) else 1


if __name__ == '__main__':
    sys.exit(main())
