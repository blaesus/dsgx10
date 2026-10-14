"""A client of rank 0's socket (docs/protocol.md, version 2), for the server's tests: submit, collect, cancel,
release; several requests in flight on one connection (the engine takes one front-end at a time)."""
import socket
import struct
import time

HELLO, SUBMIT, CANCEL, RELEASE = 0x01, 0x02, 0x03, 0x04
HELLO_E, TOKENS, DONE = 0x81, 0x82, 0x83
F_SEED, F_KEEP, F_LOGPROBS = 1, 2, 4
REASONS = ['stop', 'length', 'cancelled', 'context', 'error']
STOP_IDS = (154820, 154827, 154829)
VOCAB = 154856


def synth_ids(seed, n, lo=1000, hi=150000):
    """n token ids from a seed (ordinary vocabulary ids, no added tokens)."""
    out, x = [], seed * 0x9E3779B97F4A7C15 & (2 ** 64 - 1)
    for _ in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) & (2 ** 64 - 1)
        out.append(lo + (x >> 33) % (hi - lo))
    return out


class Engine:
    def __init__(self, path, timeout=600):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(timeout)
        self.s.connect(path)
        self.buf = b''
        t, p = self.frame()
        assert t == HELLO_E, t
        self.version, self.max_ctx, self.vocab, self.eos, self.max_sess, nl = struct.unpack('<6I', p[:24])
        self.name = p[24:24 + nl].decode()
        self.send(HELLO, struct.pack('<II', 2, 0))
        self.next_id = 1
        self.out = {}  # req id -> its result so far

    def close(self):
        self.s.close()

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
               n_top=None, stop=STOP_IDS):
        rid = self.next_id
        self.next_id += 1
        flags = (F_KEEP if keep else 0) | (F_SEED if seed is not None else 0) | (F_LOGPROBS if n_top is not None else 0)
        hdr = struct.pack('<QQQQffIIIIIIII', rid, session, base, seed or 0, temp, top_p, top_k, max_new, flags,
                          n_top or 0, 0, len(stop), len(prompt), 0)
        self.send(SUBMIT, hdr + struct.pack('<%dI' % len(stop), *stop) + struct.pack('<%dI' % len(prompt), *prompt))
        self.out[rid] = {'ids': [], 'lp': [], 'top': [], 'n_top': n_top or 0, 't0': time.time(), 't_first': None}
        return rid

    def pump(self, cancel_after=None):
        """one frame: (req id, DONE?)"""
        t, p = self.frame()
        if t == TOKENS:
            r, n, fl = struct.unpack('<QII', p[:16])
            o = self.out[r]
            o['ids'] += struct.unpack('<%dI' % n, p[16:16 + 4 * n])
            off = 16 + 4 * n
            if fl & 1:
                o['lp'] += struct.unpack('<%df' % n, p[off:off + 4 * n])
                off += 4 * n
            if fl & 2:
                k = o['n_top']
                pairs = struct.unpack('<' + 'If' * (n * k), p[off:off + 8 * n * k])
                for i in range(n):
                    o['top'].append(list(zip(pairs[2 * k * i::2][:k], pairs[2 * k * i + 1::2][:k])))
            if o['t_first'] is None:
                o['t_first'] = time.time() - o['t0']
            if cancel_after and r in cancel_after and len(o['ids']) >= cancel_after[r] and not o.get('cancel_sent'):
                self.send(CANCEL, struct.pack('<Q', r))
                o['cancel_sent'] = True
            return r, False
        if t == DONE:
            (r, reason, pt, cached, comp, q_us, p_us, d_us, el, _) = struct.unpack('<QIIIIQQQII', p[:56])
            self.out[r].update(reason=REASONS[reason], prompt=pt, cached=cached, completion=comp, queue_us=q_us,
                               prefill_us=p_us, decode_us=d_us, error=p[56:56 + el].decode(),
                               t_total=time.time() - self.out[r]['t0'])
            return r, True
        return None, False

    def wait(self, rids, cancel_after=None):
        left = set(rids)
        while left:
            r, d = self.pump(cancel_after)
            if d:
                left.discard(r)
        return [self.out[r] for r in rids]

    def run(self, *a, cancel_after=None, **k):
        r = self.submit(*a, **k)
        return self.wait([r], {r: cancel_after} if cancel_after else None)[0]

    def release(self, session):
        self.send(RELEASE, struct.pack('<Q', session))


def show(name, o):
    print('%-34s %-9s prompt %6d cached %6d  prefill %8.1f ms  %4d tokens %8.1f ms (%5.1f tok/s)%s' % (
        name, o['reason'], o['prompt'], o['cached'], o['prefill_us'] / 1e3, o['completion'], o['decode_us'] / 1e3,
        (o['completion'] - 1) / max(1e-9, o['decode_us'] / 1e6), ('  ERROR ' + o['error']) if o['error'] else ''),
        flush=True)
