#!/usr/bin/env python3
"""Drafts change nothing (docs/serving.md, "Drafts"): the same requests against a server started with --no-spec
(--record FILE: their tokens and logprobs written) and against one with drafts (--compare FILE: the same tokens and
logprobs, bit for bit). The requests: one session greedy and one seeded with logprobs; a stop id inside the text
and a turn after it (the session's history as batch_loop leaves it); 2, 3 and 4 sessions at once (greedy and seeded
mixed); a request arriving mid-decode (the others paused and resumed); a turn on a session after it was evicted to
the disk tier and read back; turns without an eviction past and below the indexer's 2048 positions; a fork at a
shorter prefix; a long request whose pages grow. With synthetic routed experts the MTP's drafts are never right, so
the drafts server also runs with --spec-force 3 --spec-oracle and --repeat 2: the first pass verifies 3 MTP drafts a
step (all rejected), the second takes the first pass's tokens as drafts, about one in four changed (accepted runs of
0 to 3, stops inside them).

  spec_test.py --socket S --record FILE                 # against glm53_rank --no-spec
  spec_test.py --socket S --compare FILE [--repeat 2]   # against glm53_rank (drafts on)
"""
import argparse
import json
import struct
import sys

from engine import Engine, show, synth_ids

FAILS = []


def check(cond, what):
    print(('ok   ' if cond else 'FAIL ') + what, flush=True)
    if not cond:
        FAILS.append(what)


def bits(lps):
    return [struct.unpack('<I', struct.pack('<f', x))[0] for x in lps]


def scenarios(e, max_new):
    """name -> (ids, logprob bits)"""
    r = {}

    def keep(name, o):
        show(name, o)
        r[name] = {'ids': list(o['ids']), 'lp': bits(o['lp']), 'reason': o['reason']}

    P = [synth_ids(300 + k, 180 + 170 * k) for k in range(4)]
    samp = [dict(temp=0.0), dict(temp=0.8, top_p=0.95, seed=41, n_top=3), dict(temp=0.0, n_top=2),
            dict(temp=1.0, top_p=0.9, seed=43)]
    # one session
    keep('alone greedy', e.run(0x6100, P[0], max_new=max_new, keep=False, **samp[0]))
    keep('alone seeded', e.run(0x6101, P[1], max_new=max_new, keep=False, **samp[1]))
    # a stop id inside the text (its first occurrence ends it), then a turn on that session: its history as kept
    for name, k, at in (('greedy', 0, 9), ('seeded', 1, 14)):
        ids = r['alone ' + name]['ids']
        sid = 0x6110 + k
        o = e.run(sid, P[k], max_new=max_new, stop=(ids[at],), **samp[k])
        keep('stop inside, %s' % name, o)
        keep('stop inside, %s, a turn' % name, e.run(sid, P[k] + list(o['ids']) + synth_ids(330 + k, 30), max_new=max_new,
                                                    stop=(ids[at],), **samp[k]))
        e.release(sid)
    # 2, 3, 4 at once
    sid = 0x6200
    for n in (2, 3, 4):
        rids = []
        for k in range(n):
            sid += 1
            rids.append(e.submit(sid, P[k], max_new=max_new, keep=False, **samp[k]))
        for k, o in enumerate(e.wait(rids)):
            keep('%d at once: %d' % (n, k), o)
    # one arriving mid-decode
    rids = [e.submit(0x6300 + k, P[k], max_new=max_new, keep=False, **samp[k]) for k in range(2)]
    while len(e.out[rids[0]]['ids']) < 7:
        e.pump()
    rids.append(e.submit(0x6302, P[2], max_new=max_new, keep=False, **samp[2]))
    for k, o in enumerate(e.wait(rids)):
        keep('mid-decode arrival: %d' % k, o)
    # a session, evicted by fillers and read back from the disk tier, then a turn; a fork of it
    A = 0x6400
    PA = synth_ids(310, 2600)
    o1 = e.run(A, PA, max_new=max_new, **samp[1])
    keep('A cold', o1)
    e.run(0x6401, synth_ids(311, 5500), max_new=4)
    e.run(0x6402, synth_ids(312, 5500), max_new=4)
    PT = PA + list(o1['ids']) + synth_ids(313, 40)
    keep('A turn after eviction', e.run(A, PT, max_new=max_new, **samp[1]))
    # turns without an eviction: on a long session (past the indexer's 2048 positions) and a short one
    for name, sid, n in (('B', 0x6405, 2600), ('C', 0x6406, 900)):
        PB = synth_ids(320 + n, n)
        ob = e.run(sid, PB, max_new=max_new, **samp[0])
        keep('%s cold' % name, ob)
        keep('%s turn' % name, e.run(sid, PB + list(ob['ids']) + synth_ids(321, 40), max_new=max_new, **samp[0]))
        e.release(sid)
    keep('fork of A at 2000', e.run(0x6403, PA[:2000] + synth_ids(314, 50), base=A, max_new=max_new, keep=False,
                                    **samp[0]))
    for s in (A, 0x6401, 0x6402):
        e.release(s)
    # a long one: its pages grow (--grow) while it decodes
    keep('long', e.run(0x6500, synth_ids(315, 1000), max_new=4 * max_new, keep=False, **samp[3]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--record')
    g.add_argument('--compare')
    ap.add_argument('--max-new', type=int, default=40)
    ap.add_argument('--repeat', type=int, default=1,
                    help='the requests this many times (a server with --spec-oracle drafts from the earlier pass)')
    ap.add_argument('--save', help='this run\'s tokens too (diagnostics)')
    a = ap.parse_args()
    e = Engine(a.socket)
    runs = [scenarios(e, a.max_new) for _ in range(a.repeat)]
    e.close()
    if a.save:
        json.dump(runs, open(a.save, 'w'))
    if a.record:
        r = runs[0]
        json.dump(r, open(a.record, 'w'))
        check(all(x['reason'] in ('stop', 'length') for x in r.values()), 'recorded %d requests without drafts' % len(r))
        return
    ref = json.load(open(a.compare))
    for j, r in enumerate(runs):
        for name, x in r.items():
            shown = name if j == 0 else '%s (pass %d)' % (name, j + 1)
            y = ref.get(name)
            same = y is not None and x['ids'] == y['ids'] and x['lp'] == y['lp'] and x['reason'] == y['reason']
            if not same and y is not None:
                i = 0
                while i < min(len(x['ids']), len(y['ids'])) and x['ids'][i] == y['ids'][i]:
                    i += 1
                print('  %s: first difference at token %d: %s against %s' % (shown, i, x['ids'][i:i + 6],
                                                                              y['ids'][i:i + 6]))
            check(same, '%s: the tokens and logprobs of the server without drafts (%d tokens, %s)' % (
                shown, len(x['ids']), x['reason']))


if __name__ == '__main__':
    main()
    print('spec_test: %s' % ('passed' if not FAILS else 'FAILED: ' + '; '.join(FAILS)), flush=True)
    sys.exit(1 if FAILS else 0)
