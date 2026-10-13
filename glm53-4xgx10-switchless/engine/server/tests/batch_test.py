#!/usr/bin/env python3
"""Sessions decoding together (docs/serving.md, B1/B2): N requests in flight at once on one connection give, each,
the tokens it gives alone (greedy and seeded sampling), whether they share steps (up to --slots a step) or take
turns (more requests than slots, turns of --slice tokens); a request arriving mid-decode pauses the others for its
prefill and they go on unchanged.

  batch_test.py --socket S [--n 2,3,4,6]
"""
import argparse
import sys

from engine import Engine, show, synth_ids

FAILS = []


def check(cond, what):
    print(('ok   ' if cond else 'FAIL ') + what, flush=True)
    if not cond:
        FAILS.append(what)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    ap.add_argument('--n', default='2,3,4,6')
    ap.add_argument('--max-new', type=int, default=48)
    a = ap.parse_args()
    e = Engine(a.socket)
    ns = [int(x) for x in a.n.split(',')]
    m = max(ns)
    prompts = [synth_ids(100 + k, 200 + 150 * k) for k in range(m)]
    samp = [dict(temp=0.0) if k % 2 == 0 else dict(temp=0.9, top_p=0.95, seed=1000 + k) for k in range(m)]
    alone = []
    for k in range(m):
        o = e.run(0x7000 + k, prompts[k], max_new=a.max_new, keep=False, **samp[k])
        show('alone %d' % k, o)
        alone.append(o['ids'])
    sid = 0x8000
    for n in ns:
        rids = []
        for k in range(n):
            sid += 1
            rids.append(e.submit(sid, prompts[k], max_new=a.max_new, keep=False, **samp[k]))
        outs = e.wait(rids)
        same = all(outs[k]['ids'] == alone[k] for k in range(n))
        for k in range(n):
            if outs[k]['ids'] != alone[k]:
                print('  session %d: %s\n        alone: %s' % (k, outs[k]['ids'][:12], alone[k][:12]))
        check(same and all(o['reason'] in ('stop', 'length') for o in outs),
              '%d sessions at once: each gives its tokens alone (%s)' % (n, ' '.join(o['reason'] for o in outs)))
    # a request that comes while others decode
    rids = [e.submit(0x9000 + k, prompts[k], max_new=a.max_new, keep=False, **samp[k]) for k in range(2)]
    while len(e.out[rids[0]]['ids']) < 10:
        e.pump()
    rids.append(e.submit(0x9002, prompts[2], max_new=a.max_new, keep=False, **samp[2]))
    outs = e.wait(rids)
    check(all(outs[k]['ids'] == alone[k] for k in range(3)), 'a request arriving mid-decode: every session unchanged')
    e.close()


if __name__ == '__main__':
    main()
    print('batch_test: %s' % ('passed' if not FAILS else 'FAILED: ' + '; '.join(FAILS)), flush=True)
    sys.exit(1 if FAILS else 0)
