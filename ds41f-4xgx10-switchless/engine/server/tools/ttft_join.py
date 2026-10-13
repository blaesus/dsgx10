#!/usr/bin/env python3
"""ttft_turns.py's rows joined with rank 0's per-request lines (serve.cpp: `session ...: prompt, reused (how),
prefill` and `turn: state ops ..., layers 0-19, replay, head, draft windows, rows; first token to decode,
first decode step`):
  ttft_join.py TURNS_LOG SERVER_LOG [--rows] [--json OUT]
Prints, per phase and size, the medians (and the count) of: the client's time to first token, the server's
prefill, the state operations (host, GPU), the chunks' GPU time by part (layers: 0-19 for CED, every layer
for an exact chunk), the rest (prefill less state ops, layers, replay and head: host and waits), the draft
windows and the first decode step after the token. --rows: every request; --json: the joined rows."""
import json
import re
import statistics as st
import sys

args = [a for a in sys.argv[1:] if not a.startswith('--')]
show_rows = '--rows' in sys.argv
jout = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
if jout and jout in args:
    args.remove(jout)
if len(args) != 2:
    sys.exit(__doc__)

rows = {}
for line in open(args[0]):
    f = line.rstrip('\n').split('\t')
    if f[0] != 'row':
        continue
    rid = int(f[1])
    rows[rid] = dict(rid=rid, phase=f[2], size=int(f[3]), prompt=int(f[4]), cached=int(f[5]), ttft=float(f[6]),
                     total=float(f[7]), srv_prefill=float(f[8]), out=int(f[9]))
re_sess = re.compile(r'serve: req (\d+) session \S+: prompt (\d+), reused (\d+) \((\w+)\), prefill ([\d.]+) ms')
re_turn = re.compile(r'serve: req (\d+) turn: state ops ([\d.]+) ms \(GPU ([\d.]+)\), (\d+) chunks: layers 0-19 ([\d.]+) ms, '
                     r'replay ([\d.]+) ms, head ([\d.]+) ms, draft windows ([\d.]+) ms, rows ([\d.]+) ms; first token to '
                     r'decode ([\d.]+) ms, first decode step ([-\d.]+) ms')
for line in open(args[1]):
    m = re_sess.search(line)
    if m and int(m.group(1)) in rows:
        r = rows[int(m.group(1))]
        r.update(how=m.group(4), reused=int(m.group(3)), prefill=float(m.group(5)))
    m = re_turn.search(line)
    if m and int(m.group(1)) in rows:
        r = rows[int(m.group(1))]
        v = [float(x) for x in m.groups()[1:]]
        r.update(ops=v[0], ops_gpu=v[1], chunks=int(v[2]), layers=v[3], replay=v[4], head=v[5], windows=v[6],
                 rows_ms=v[7], tail=v[8], dec1=v[9])
missing = [rid for rid, r in rows.items() if 'layers' not in r or 'prefill' not in r]
if missing:
    print('no server lines for requests %s' % missing[:10])
for r in rows.values():
    if 'layers' in r and 'prefill' in r:
        r['new'] = r['prompt'] - r['reused']
        r['gpu'] = r['layers'] + r['replay'] + r['head'] + r['rows_ms']
        r['rest'] = r['prefill'] - r['ops'] - r['gpu']
cols = [('ttft', 'TTFT'), ('prefill', 'prefill'), ('ops', 'ops'), ('ops_gpu', 'ops GPU'), ('layers', 'layers'),
        ('replay', 'replay'), ('head', 'head'), ('rows_ms', 'rows'), ('rest', 'rest'), ('windows', 'windows'),
        ('dec1', 'decode 1')]
groups = {}
for r in sorted(rows.values(), key=lambda r: r['rid']):
    if 'gpu' in r:
        groups.setdefault((r['phase'], r['size']), []).append(r)
print('%-9s %6s %3s %7s %6s %7s ' % ('phase', 'size', 'n', 'ctx', 'how', 'new') +
      ' '.join('%8s' % c[1] for c in cols))
for (ph, sz), rs in groups.items():
    hows = sorted({r['how'] for r in rs})
    print('%-9s %6d %3d %7d %6s %7d ' % (ph, sz, len(rs), st.median(r['prompt'] for r in rs), '/'.join(hows)[:6],
                                       st.median(r['new'] for r in rs)) +
          ' '.join('%8.1f' % st.median(r[c] for r in rs) for c, _ in cols))
if show_rows:
    print()
    for r in sorted(rows.values(), key=lambda r: r['rid']):
        if 'gpu' in r:
            print('%4d %-9s %6d %7d %-8s %6d ' % (r['rid'], r['phase'], r['size'], r['prompt'], r['how'], r['new']) +
                  ' '.join('%8.1f' % r[c] for c, _ in cols))
if jout:
    with open(jout, 'w') as f:
        for r in sorted(rows.values(), key=lambda r: r['rid']):
            f.write(json.dumps(r) + '\n')
