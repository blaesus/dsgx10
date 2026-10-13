#!/usr/bin/env python3
"""An engine A/B's workload through rank 0's socket (docs/protocol.md; no front-end), for one server session of
one binary at one draft setting (the server's own: --no-spec or not). abserve.sh runs it; abcmp.py compares.
  warm-up  a 2k prompt and the prose prompt (not counted)
  bench    bench.py's four decode prompts as dsfront renders them, thinking off (--prompts: <type>.npy), greedy,
           512 tokens, twice each, a fresh session each (bench.py decode)
  edit     the edit bench's traces (--edit: index.jsonl and a golden dir a turn; r1: 54 turns of 16 tasks,
           thinking off and max) replayed in order: a session a task, its first turn forked from the previous
           task's session (the shared system prompt, as served), each turn the trace's prompt with its own
           number of tokens (a trace ends in EOS)
  turns    at 16384 and 25984 positions of --long, 3 times each in a fresh session: the cold prompt, then a turn
           of +200 tokens of --long, then one of +2000, each after the previous answer, each answering 64
           tokens (CED's turn rule: both turns are >= 128 new positions)
Every request: its tokens' hash, the engine's prefill and decode times (DONE), and for edit turns whether the
tokens equal the trace's.
  python3 engine/kernels/tests/abload.py --socket S --out OUT.json [--label L] [--parts bench,edit,turns]"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'tests'))
from serve_test import Engine, load_npy_i32  # noqa: E402

TYPES = ('prose', 'code', 'agentic', 'json')
EOS = 1


def hid(ids):
    return hashlib.sha1(struct.pack('<%dI' % len(ids), *ids)).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--parts', default='bench,edit,turns')
    ap.add_argument('--prompts', default='/data/dsgb10/kernels/abload/prompts')
    ap.add_argument('--edit', default='/data/dsgb10/bench/edit/r1')
    ap.add_argument('--long', default='/data/dsgb10/golden/long/tokens.npy')
    a = ap.parse_args()
    parts = a.parts.split(',')
    e = Engine(a.socket)
    print('engine: %s, context %d; %s' % (e.name, e.max_ctx, a.label))
    T = load_npy_i32(a.long)
    recs = []

    def run(case, session, prompt, max_new, keep, base=0, expect=None):
        t0 = time.time()
        o = e.run(session, prompt, max_new=max_new, keep=keep, base=base, stop=(EOS,))
        r = dict(case=case, reason=o['reason'], prompt=o['prompt'], cached=o['cached'], completion=o['completion'],
                 queue_ms=o['queue_us'] / 1e3, prefill_ms=o['prefill_us'] / 1e3, decode_ms=o['decode_us'] / 1e3,
                 wall_ms=(time.time() - t0) * 1e3, hash=hid(o['ids']), error=o['error'])
        if expect is not None:
            r['trace_equal'] = list(o['ids']) == list(expect)
        recs.append(r)
        print('%-34s %-6s prompt %6d cached %6d prefill %8.1f ms  %4d tokens %8.1f ms (%6.1f tok/s)  %s%s %s' % (
            case, r['reason'], r['prompt'], r['cached'], r['prefill_ms'], r['completion'], r['decode_ms'],
            max(0, r['completion'] - 1) / max(1e-9, r['decode_ms'] / 1e3), r['hash'],
            '' if expect is None else (' trace' if r['trace_equal'] else ' TRACE-DIFFERS'), r['error']))
        sys.stdout.flush()
        if o['reason'] == 'error':
            raise SystemExit('request %s failed: %s' % (case, o['error']))
        return o

    prompts = {t: load_npy_i32('%s/%s.npy' % (a.prompts, t)) for t in TYPES}
    S = 0x4ab0000000000000
    run('warmup/2k', S + 1, T[:2048], 32, False)
    run('warmup/prose', S + 2, prompts['prose'], 64, False)

    if 'bench' in parts:
        for rep in range(2):
            for i, t in enumerate(TYPES):
                run('bench/%s/%d' % (t, rep), S + 0x100 + 16 * rep + i, prompts[t], 512, False)

    if 'edit' in parts:
        rows = [json.loads(line) for line in open(a.edit + '/index.jsonl')]
        sid = {}
        prev = 0
        for r in rows:
            key = (r['run'], r['task'])
            first = key not in sid
            if first:
                sid[key] = S + 0x1000 + len(sid)
            tok = load_npy_i32('%s/%s/tokens.npy' % (a.edit, r['dir']))
            n, k = r['n_prompt'], r['n_out']
            run('edit/%s/%s/t%d' % (r['run'], r['task'], r['turn']), sid[key], tok[:n], k, True,
                base=prev if first else 0, expect=tok[n:n + k])
            if first:
                if prev:
                    e.release(prev)
                prev = sid[key]
        e.release(prev)

    if 'turns' in parts:
        for c in (16384, 25984):
            for rep in range(3):
                s = S + 0x2000 + 16 * rep + (c >> 10)
                p = T[:c]
                o = run('turn/%d/cold/%d' % (c, rep), s, p, 64, True)
                p = p + list(o['ids']) + T[c:c + 200]
                o = run('turn/%d/+200/%d' % (c, rep), s, p, 64, True)
                p = p + list(o['ids']) + T[c + 200:c + 2200]
                run('turn/%d/+2000/%d' % (c, rep), s, p, 64, True)
                e.release(s)

    json.dump(dict(label=a.label, engine=e.name, records=recs), open(a.out, 'w'), indent=1)
    print('%d requests; %d edit turns equal to the trace of %d' % (
        len(recs), sum(1 for r in recs if r.get('trace_equal')), sum(1 for r in recs if 'trace_equal' in r)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
