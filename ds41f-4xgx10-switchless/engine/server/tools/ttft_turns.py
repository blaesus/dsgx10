#!/usr/bin/env python3
"""Small agentic turns through a test server, for where a turn's time to first token goes (docs/engine.md,
"A small turn's time to first token"):
  ttft_turns.py [--sock PATH] [--edit RUN] [--synth NPY] [--sizes 44,200,350,728,1368] [--short-rounds 2]
                [--long 16384] [--rounds 6] [--big 11632]
The turns' new tokens are the edit bench's own: RUN's dumped prompts (RUN/dump/NNNNNN.json, the ids
dsfront sent) and their reuse (RUN/report/joined.jsonl), each request's new tokens laid end to end as a
stream that every turn takes its next n tokens from. Four phases, as the edit bench meets them:
  cold      the first task's first prompt (~1.4k), cold;
  snapshot  the other tasks' first prompts, each a new session forked from the one before (its window
            snapshot at 1024: the shared system prompt), as dsfront names the base;
  short     the last of them goes on: rounds of each size, live, each prompt the last one, its output
            (4 tokens, greedy) and n new tokens (contexts ~1.4k to ~10k);
  long      a new session: a cold prompt of --long tokens of --synth (not a turn of interest), then
            --rounds rounds of each size, live, then one turn of --big (contexts ~16k to ~33k).
Sizes rotate within a round. Prints one row per request: `row`, rid, phase, size, prompt, cached, the
client's time to first token and total ms, the server's prefill ms, tokens out (tab-separated); ttft_join.py
joins them with rank 0's `serve: req N turn:` lines. --sock: the server's socket (the test server's
default, /tmp/dsgb10-engine.sock)."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests'))
from serve_test import Engine, load_npy_i32  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--sock', default='/tmp/dsgb10-engine.sock')
ap.add_argument('--edit', default=os.path.expanduser('~/dsgb10/editruns/r2'))
ap.add_argument('--synth', default='/data/dsgb10/golden/synth270k/tokens.npy')
ap.add_argument('--sizes', default='44,200,350,728,1368')
ap.add_argument('--short-rounds', type=int, default=2)
ap.add_argument('--long', type=int, default=16384)
ap.add_argument('--rounds', type=int, default=6)
ap.add_argument('--big', type=int, default=11632)
ap.add_argument('--max-new', type=int, default=4)
args = ap.parse_args()
sizes = [int(x) for x in args.sizes.split(',') if x]

# the edit bench's requests: ids, session, and how much of each the server reused
joined = {}
for line in open(os.path.join(args.edit, 'report', 'joined.jsonl')):
    r = json.loads(line)
    if r['req_id'] in joined:
        sys.exit('ttft_turns.py: req_id %d twice in joined.jsonl' % r['req_id'])
    joined[r['req_id']] = r
reqs = []
for rid in sorted(joined):
    d = json.load(open(os.path.join(args.edit, 'dump', '%06d.json' % rid)))
    j = joined[rid]
    if len(d['ids']) != j['n_prompt']:
        sys.exit('ttft_turns.py: req %d: %d ids, n_prompt %d' % (rid, len(d['ids']), j['n_prompt']))
    reqs.append((rid, d['ids'], j))
stream = []
for rid, ids, j in reqs:
    stream += ids[j['reused']:]
firsts = [ids for rid, ids, j in reqs if j['turn'] == 0 and j['run'].endswith('off') and j['rep'] == 0]
print('edit bench %s: %d requests, a stream of %d new tokens; %d first prompts (%d-%d tokens)' %
      (args.edit, len(reqs), len(stream), len(firsts), min(map(len, firsts)), max(map(len, firsts))))
off = 0


def take(n):
    global off
    out = []
    while len(out) < n:
        k = min(n - len(out), len(stream) - off)
        out += stream[off:off + k]
        off = (off + k) % len(stream)
    return out


e = Engine(args.sock)
print('engine: %s, context %d' % (e.name, e.max_ctx))
S = 0x77f0000000000000


def row(phase, size, prompt, o):
    print('row\t%d\t%s\t%d\t%d\t%d\t%.1f\t%.1f\t%.1f\t%d' % (e.next_id - 1, phase, size, len(prompt), o['cached'],
                                                         1e3 * (o['t_first'] or 0), 1e3 * o['t_total'],
                                                         o['prefill_us'] / 1e3, len(o['ids'])))
    sys.stdout.flush()


t0 = time.time()
# cold and snapshot: each first prompt a new session, forked from the one before
s = S
prompt = firsts[0]
o = e.run(s, prompt, max_new=args.max_new)
row('cold', len(prompt), prompt, o)
for k, f in enumerate(firsts[1:]):
    prompt = f
    o = e.run(s + 1, prompt, max_new=args.max_new, base=s)
    row('snapshot', len(prompt), prompt, o)
    s += 1
# short: the last session goes on
prompt = prompt + o['ids']
for r in range(args.short_rounds):
    for k in range(len(sizes)):
        n = sizes[(k + r) % len(sizes)]
        prompt = prompt + take(n)
        o = e.run(s, prompt, max_new=args.max_new)
        row('short', n, prompt, o)
        prompt = prompt + o['ids']
for x in range(S, s + 1):
    e.release(x)
print('cold, snapshot, short: %.1f s' % (time.time() - t0))
# long: a cold prompt of synthetic text, then rounds of each size and one big turn
T = load_npy_i32(args.synth)
s = S + 0x100
prompt = T[:args.long]
o = e.run(s, prompt, max_new=args.max_new)
row('longcold', args.long, prompt, o)
prompt = prompt + o['ids']
for r in range(args.rounds):
    for k in range(len(sizes)):
        n = sizes[(k + r) % len(sizes)]
        prompt = prompt + take(n)
        o = e.run(s, prompt, max_new=args.max_new)
        row('long', n, prompt, o)
        prompt = prompt + o['ids']
if args.big:
    prompt = prompt + take(args.big)
    o = e.run(s, prompt, max_new=args.max_new)
    row('long', args.big, prompt, o)
e.release(s)
print('all: %.1f s' % (time.time() - t0))
