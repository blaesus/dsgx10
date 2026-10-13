#!/usr/bin/env python3
"""tjoin.py TURNS_OUT SERVER_LOG: turns.py's rows with rank 0's turn lines for the same request ids: the
state ops, chunks, the encoder (layers 0-19 of the turn's prefill), replay, head, draft windows, and the first
decode step. A request id's first turn line is taken: serve_test.py, run after turns.py against the same
server, counts its ids from 1 again."""
import re
import sys

turn = {}
for ln in open(sys.argv[2], errors='replace'):
    m = re.search(r'serve: req (\d+) turn: state ops ([\d.]+) ms \(GPU ([\d.]+)\), (\d+) chunks: layers 0-19 ([\d.]+) ms, '
                  r'replay ([\d.]+) ms, head ([\d.]+) ms, draft windows ([\d.]+) ms, rows ([\d.]+) ms; first token to '
                  r'decode ([\d.]+) ms, first decode step (-?[\d.]+) ms', ln)
    if m:
        turn.setdefault(int(m.group(1)), [float(x) for x in m.groups()[1:]])
print('%-18s %7s %6s %8s %8s | %6s %6s %3s %8s %7s %6s %6s %6s %6s' % (
    'turn', 'prompt', 'new', 'ttft', 'prefill', 'ops', 'opsGPU', 'ch', 'encoder', 'replay', 'head', 'drafts', 'tail',
    'dec1'))
for ln in open(sys.argv[1]):
    if not ln.startswith('row\t'):
        continue
    f = ln.rstrip('\n').split('\t')
    rid, label, prompt, cached, ttft, pre = int(f[1]), f[2], int(f[3]), int(f[4]), float(f[5]), float(f[7])
    t = turn.get(rid)
    s = '%-18s %7d %6d %8.1f %8.1f' % (label, prompt, prompt - cached, ttft, pre)
    if t:
        s += ' | %6.1f %6.1f %3d %8.1f %7.1f %6.1f %6.1f %6.1f %6.1f' % (t[0], t[1], int(t[2]), t[3], t[4], t[5], t[6], t[8],
                                                                     t[9])
    print(s)
