#!/usr/bin/env python3
"""pi's turnaround between turns, from dsfront's dump (--dump-dir) and the timers' logs: for each pair of
consecutive requests of one pi run (pidrive.py's runs.jsonl windows), the gap from the first one's output
record (written at the engine's DONE, ~0.2 ms before [DONE] goes out) to the next one's request record
(written once dsfront has parsed, tokenized and submitted it): pi's turnaround plus dsfront's intake. DONE
sent before the step queued behind the last token would hide min(tail, gap) of each turn.
  piturn.py OUT DUMP RUN4_LOG FRONT_LOG     (OUT/runs.jsonl; OUT/events/*.jsonl from --mode json)"""
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

out, dump, run4, front = sys.argv[1:5]
runs = [json.loads(l) for l in open(os.path.join(out, 'runs.jsonl'))]
tail = {}
for line in open(run4, errors='replace'):
    m = re.search(r'serve: req (\d+) turn: .*last token to DONE ([-\d.]+) ms \(the step after it ([-\d.]+) ms', line)
    if m:
        tail[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
fe, t_end, t_plan = {}, {}, {}  # with stamp.py's clock on each line: [DONE] sent, the next request submitted
for line in open(front, errors='replace'):
    ns = re.match(r'(\d{18,}) ', line)
    m = re.search(r'req (\d+): end: the stop to the engine\'s DONE ([\d.]+) ms, to \[DONE\] ([\d.]+) ms', line)
    if m:
        fe[int(m.group(1))] = float(m.group(3))
        if ns:
            t_end[int(m.group(1))] = int(ns.group(1))
    m = re.search(r'req (\d+) plan: ', line)
    if m and ns:
        t_plan[int(m.group(1))] = int(ns.group(1))
reqs = {}
for p in glob.glob(os.path.join(dump, '*.json')):
    if p.endswith('.out.json'):
        continue
    n = int(os.path.basename(p).split('.')[0])
    o = p[:-5] + '.out.json'
    if not os.path.exists(o):
        continue
    r, x = json.load(open(p)), json.load(open(o))
    msgs = (r.get('request') or {}).get('messages') or []
    last = next((m for m in reversed(msgs) if m.get('role') == 'assistant'), None)
    calls = [c['function']['name'] for c in (last or {}).get('tool_calls') or []]
    reqs[n] = dict(n=n, t_in=os.stat(p).st_mtime_ns, t_out=os.stat(o).st_mtime_ns, finish=x.get('finish'),
                   total_s=x.get('total_s'), prompt=(x.get('usage') or {}).get('prompt_tokens'),
                   completion=(x.get('usage') or {}).get('completion_tokens'), calls_before=calls, n_msgs=len(msgs))


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float('nan')


pairs = defaultdict(list)
for ru in runs:
    mine = sorted((r for r in reqs.values() if ru['t0'] <= r['t_in'] <= ru['t1']), key=lambda r: r['n'])
    for a, b in zip(mine, mine[1:]):
        if a['n'] in t_end and b['n'] in t_plan:  # stamped: [DONE] sent -> the next request submitted
            gap = (t_plan[b['n']] - t_end[a['n']]) / 1e6
        else:  # the dump files' times (the kernel's coarse clock): DONE -> the next request's record
            gap = (b['t_in'] - a['t_out']) / 1e6
        t = tail.get(a['n'], (float('nan'), float('nan')))[0]
        names = b['calls_before']
        kind = 'none' if not names else names[0] if len(set(names)) == 1 else 'mixed'
        cyc = (b['t_in'] - a['t_in']) / 1e6
        pairs[ru['mode']].append(dict(a=a['n'], b=b['n'], gap=gap, tail=t, kind=kind, calls=len(names), cycle=cyc,
                                      task=ru['task'], prompt=b['prompt']))
    ru['requests'] = len(mine)
for mode, v in pairs.items():
    rs = [r for r in runs if r['mode'] == mode]
    print(f'## {mode} ({"stamped dsfront log" if t_end else "dump files' times"}): {len(rs)} pi runs ({sum(r["passed"] for r in rs)} passed their checks), '
          f'{sum(r["requests"] for r in rs)} requests, {len(v)} turnarounds')
    print('| after | turns | gap: median (p10-p90), min-max | tail (median) | gap >= tail | hidden: min(tail, gap), mean | '
          'turn cycle, mean | hidden / cycle |')
    print('|---|---|---|---|---|---|---|---|')
    for k in ['read', 'edit', 'write', 'mixed', 'all']:
        w = v if k == 'all' else [x for x in v if x['kind'] == k]
        if not w:
            continue
        g = [x['gap'] for x in w]
        wt = [x for x in w if x['tail'] == x['tail']]  # with the engine's tail (none from a mock)
        hid = [min(x['tail'], x['gap']) for x in wt]
        cyc = [x['cycle'] for x in wt]
        nan = float('nan')
        print('| %s | %d | %.1f ms (%.1f-%.1f), %.1f-%.1f | %.1f ms | %d of %d | %.1f ms | %.0f ms | %.2f %% |' % (
            k, len(w), st.median(g), q(g, .1), q(g, .9), min(g), max(g),
            st.median(x['tail'] for x in wt) if wt else nan, sum(1 for x in wt if x['gap'] >= x['tail']), len(wt),
            st.mean(hid) if hid else nan, st.mean(cyc) if cyc else nan, 100 * sum(hid) / sum(cyc) if cyc else nan))
    big = sorted(v, key=lambda x: -x['gap'])[:3]
    print('largest gaps: ' + '; '.join('%s req %d->%d %s %.1f ms (prompt %s)' % (x['task'], x['a'], x['b'], x['kind'], x['gap'], x['prompt'])
                                       for x in big))
    print()
# json mode: where the gap goes, from the event lines' arrival times
ev_rows = []
for p in sorted(glob.glob(os.path.join(out, 'events', 'json-*.jsonl'))):
    ev = [json.loads(l) for l in open(p)]
    tag = os.path.basename(p)[5:-6]
    ru = next((r for r in runs if r['mode'] == 'json' and f'rep{r["rep"]}-{r["task"]}' == tag), None)
    if not ru:
        continue
    mine = sorted((r for r in reqs.values() if ru['t0'] <= r['t_in'] <= ru['t1']), key=lambda r: r['n'])
    ends = [e['t'] for e in ev if e['type'] == 'message_end' and e['role'] == 'assistant']
    tool_end = [e['t'] for e in ev if e['type'] == 'tool_execution_end']
    tool_start = [e['t'] for e in ev if e['type'] == 'tool_execution_start']
    for a, b in zip(mine, mine[1:]):
        t_a = t_end.get(a['n'], a['t_out'])
        t_b = t_plan.get(b['n'], b['t_in'])
        me = next((t for t in ends if t >= t_a - 5e6), None)  # pi's message_end after this DONE
        if me is None or me > t_b:
            continue
        ts = [t for t in tool_start if me <= t <= t_b]
        te = [t for t in tool_end if me <= t <= t_b]
        ev_rows.append(dict(done_to_end=(me - t_a) / 1e6,
                            end_to_tool=((ts[0] - me) / 1e6) if ts else None,
                            tools=((te[-1] - ts[0]) / 1e6) if ts and te else None,
                            tool_to_next=((t_b - te[-1]) / 1e6) if te else None,
                            gap=(t_b - t_a) / 1e6))
if ev_rows:
    def md(k):
        v = [r[k] for r in ev_rows if r[k] is not None]
        return '%.1f (%.1f-%.1f)' % (st.median(v), q(v, .1), q(v, .9)) if v else '-'
    print('## json mode: the gap by pi\'s events (medians, p10-p90, ms), %d turnarounds' % len(ev_rows))
    print('| DONE at dsfront -> pi\'s message_end | -> first tool starts | tools run | last tool ends -> next request recorded | gap |')
    print('|---|---|---|---|---|')
    print('| %s | %s | %s | %s | %s |' % (md('done_to_end'), md('end_to_tool'), md('tools'), md('tool_to_next'), md('gap')))
