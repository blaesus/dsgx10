#!/usr/bin/env python3
"""Multi-stream decode (docs/batching.md, B3): S requests at once, bench.py's four contents at 512 tokens
each (cycling when S > 4), through an OpenAI-compatible front (dsfront); or S token prompts (cut from a
tokens file) through the engine's socket (serve_test's protocol). Prints each stream's first token and
decode rate, the total over the wall from the first submit to the last end and over the window every
stream was decoding in, and, with --log, the engine's batch lines that came during the run
("serve: batch of N: ... drafts: ..."), summed.

  python3 bench/multistream.py --url http://127.0.0.1:8888/v1 --key-file F --model M --streams 1,2,3,4
  python3 bench/multistream.py --socket /tmp/dsgb10-engine.sock --streams 2,4 --log ~/dsgb10/logs/dsgb10_rank.0.log
"""
import argparse
import json
import os
import re
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import DECODE_PROMPTS, post_stream  # noqa: E402

BATCH_RE = re.compile(r'serve: batch of (\d+): (\d+) steps in ([\d.]+) ms, ([\d.]+) ms a step \(median\)'
                      r'(?:.*?; drafts: ([\d.]+) tokens and ([\d.]+) rows per step, (\d+) of (\d+) drafts accepted)?')


def http_streams(a, S):
    """S requests at once through the front: per stream its name, tokens, first token and times."""
    names = list(DECODE_PROMPTS)
    res = [None] * S

    def one(i):
        name = names[i % len(names)]
        body = dict(model=a.model, messages=[{'role': 'user', 'content': DECODE_PROMPTS[name]}],
                    max_tokens=a.max_tokens, temperature=0, chat_template_kwargs={'thinking': False})
        try:
            r = post_stream(a.url, a.key, body)
        except Exception as ex:  # noqa: BLE001
            t = time.perf_counter()
            r = dict(error=str(ex), ttft=float('nan'), t0=t, t1=t, t_first=None, usage=None)
        r['name'] = name
        res[i] = r

    th = [threading.Thread(target=one, args=(i,)) for i in range(S)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    out = []
    for r in res:
        ct = (r.get('usage') or {}).get('completion_tokens') or 0
        out.append(dict(name=r['name'], tokens=ct, ttft=r['ttft'], t0=r['t0'], t_first=r['t_first'] or r['t1'],
                        t1=r['t1'], error=r.get('error')))
    return out


def socket_streams(a, S):
    """S token prompts at once through the engine's socket (prompt tokens cut from the tokens file)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'engine', 'server', 'tests'))
    from serve_test import load_npy_i32  # noqa: E402
    from slice_test import Mux  # noqa: E402
    T = load_npy_i32(a.tokens)
    e = Mux(a.socket)
    sid0 = 0x6d73747200000000  # 'mstr'
    rids = []
    for i in range(S):
        off = 1000 + 25000 * i
        if off + a.prompt > len(T):
            off = (1000 + 1777 * i) % max(1, len(T) - a.prompt)
        rids.append(e.start('s%d' % i, sid0 + i, T[off:off + a.prompt], max_new=a.max_tokens, n_top=0))
    e.pump(lambda: e.finished(*rids), timeout=3600)
    out = []
    for i, r in enumerate(rids):
        rec, d = e.recs[r], e.recs[r]['done']
        first = rec['t_tok'][0] if rec['t_tok'] else d['t']
        out.append(dict(name=rec['name'], tokens=d['completion'], ttft=first - rec['t_submit'], t0=rec['t_submit'],
                        t_first=first, t1=d['t'], error=None if d['reason'] in ('length', 'stop') else d['reason']))
        e.release(sid0 + i)
    return out


def report(S, out):
    tot = sum(o['tokens'] for o in out)
    t0, t1 = min(o['t0'] for o in out), max(o['t1'] for o in out)
    tf = min(o['t_first'] for o in out)
    dec = sum(max(0, o['tokens'] - 1) for o in out)
    wall, win = max(1e-9, t1 - t0), max(1e-9, t1 - tf)
    print('streams %d: %d tokens in %.2f s from the first submit to the last end, %.1f tok/s in all; '
          '%d decoded in the %.2f s from the first token, %.1f tok/s' % (S, tot, wall, tot / wall, dec, win, dec / win))
    rates = []
    for o in out:
        dt = max(1e-9, o['t1'] - o['t_first'])
        rate = max(0, o['tokens'] - 1) / dt
        rates.append(rate)
        print('  %-8s %4d tokens, first token after %.3f s, then %.1f tok/s over %.2f s%s' % (
            o['name'], o['tokens'], o['ttft'], rate, dt, ' (%s)' % o['error'] if o['error'] else ''))
    return dict(streams=S, tokens=tot, wall_s=wall, tps_all=tot / wall, tps_decode=dec / win,
                tps_stream=rates, ttft=[o['ttft'] for o in out])


class LogTail:
    """The engine's batch lines written to its log since the last look."""

    def __init__(self, path):
        self.path = os.path.expanduser(path) if path else None
        self.off = os.path.getsize(self.path) if self.path and os.path.exists(self.path) else 0

    def batches(self):
        if not self.path or not os.path.exists(self.path):
            return []
        with open(self.path, 'rb') as f:
            f.seek(self.off)
            data = f.read()
            self.off += len(data)
        out = []
        for line in data.decode('utf-8', 'replace').splitlines():
            m = BATCH_RE.search(line)
            if m:
                out.append(m.groups())
        return out

    def report(self, verbose):
        bs = self.batches()
        if not bs:
            print('  engine log: no batch lines came during the run')
            return {}
        steps = sum(int(b[1]) for b in bs)
        med = statistics.median(float(b[3]) for b in bs)
        d = [b for b in bs if b[4] is not None]
        line = '  engine log: %d batch run(s) of %s request(s), %d steps, %.2f ms a step (median of the runs\' medians)' % (
            len(bs), '/'.join(sorted({b[0] for b in bs})), steps, med)
        res = dict(batches=len(bs), steps=steps, step_ms_median=med)
        if d:
            ds = sum(int(b[1]) for b in d)
            tps = sum(float(b[4]) * int(b[1]) for b in d) / max(1, ds)
            rps = sum(float(b[5]) * int(b[1]) for b in d) / max(1, ds)
            acc, ver = sum(int(b[6]) for b in d), sum(int(b[7]) for b in d)
            line += '; drafts: %.2f tokens and %.2f rows per step, %d of %d drafts accepted (%.1f%%)' % (
                tps, rps, acc, ver, 100.0 * acc / max(1, ver))
            res.update(tokens_per_step=tps, rows_per_step=rps, accepted=acc, verified=ver)
        print(line)
        if verbose:
            for b in bs:
                print('    batch of %s: %s steps, %s ms a step%s' % (
                    b[0], b[1], b[3], '' if b[4] is None else '; %s tokens, %s rows per step, %s of %s accepted' % b[4:]))
        return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='http://127.0.0.1:8888/v1', help='the front (OpenAI-compatible)')
    ap.add_argument('--key-file', default='')
    ap.add_argument('--model', default='')
    ap.add_argument('--socket', default='', help="the engine's socket instead of the front")
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy', help='--socket: the prompts')
    ap.add_argument('--prompt', type=int, default=1024, help='--socket: prompt tokens per stream')
    ap.add_argument('--streams', default='1,2,3,4')
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--log', default='', help="the engine's rank 0 log: its batch lines during each run")
    ap.add_argument('--verbose', action='store_true', help='every batch line of the log')
    ap.add_argument('--out', default='', help='the figures as JSON')
    a = ap.parse_args()
    a.key = open(a.key_file).read().strip() if a.key_file else ''
    if not a.socket and not a.model:
        ap.error('--model (the front) or --socket (the engine)')
    tail = LogTail(a.log)
    results = []
    for S in [int(x) for x in a.streams.split(',') if x]:
        for rep in range(a.reps):
            tail.batches()  # what came before the run is not its
            out = socket_streams(a, S) if a.socket else http_streams(a, S)
            r = report(S, out)
            r['rep'] = rep
            r['log'] = tail.report(a.verbose)
            results.append(r)
            sys.stdout.flush()
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
