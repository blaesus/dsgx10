#!/usr/bin/env python3
"""Rank 0's server over its socket (docs/serving.md): HELLO's values, a cold prefill, the same request again
(resumed at the full prefix, the same tokens), a fork at a shorter prefix against the same prompt prefilled cold
(the same tokens: a restart at any position is exact), a session extended by a turn (only the suffix prefilled),
evict-and-restore through the disk tier (the same tokens), seeded sampling with logprobs, cancel, a bad request.
Prompts are synthetic ids; with synthetic routed experts the text means nothing, the checks are exact.

  serve_test.py --socket S --state FILE             # the main phase; FILE keeps the reference for --kept
  serve_test.py --socket S --state FILE --kept      # after a planned stop and a start: the session is still there

The engine must run with a pool small enough for the eviction (--kv-gib 1.2 with pages of 1024) and a disk tier.
"""
import argparse
import json
import sys

from engine import Engine, STOP_IDS, VOCAB, show, synth_ids

FAILS = []


def check(cond, what):
    print(('ok   ' if cond else 'FAIL ') + what, flush=True)
    if not cond:
        FAILS.append(what)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    ap.add_argument('--state', required=True)
    ap.add_argument('--kept', action='store_true')
    ap.add_argument('--n', type=int, default=2600)
    a = ap.parse_args()
    e = Engine(a.socket)
    print('engine: %s, version %d, context %d, vocab %d, eos %d, sessions %d' % (e.name, e.version, e.max_ctx, e.vocab,
                                                                                 e.eos, e.max_sess), flush=True)
    check(e.version == 2 and e.vocab == VOCAB and e.eos in STOP_IDS and e.name == 'glm-5.3', 'HELLO: version 2, vocab 154856, eos a stop id, glm-5.3')
    A = 0xA000
    if a.kept:
        st = json.load(open(a.state))
        o = e.run(A, st['prompt'], max_new=st['max_new'])
        show('kept: the turn again', o)
        check(o['reason'] in ('stop', 'length') and o['cached'] == len(st['prompt']) - 1,
              'kept session: resumed at the full prefix (%d of %d)' % (o['cached'], len(st['prompt'])))
        check(list(o['ids']) == st['ids'], 'kept session: the same tokens')
        e.release(A)
        return
    PA = synth_ids(1, a.n)
    # 1-2: cold, then the same again
    o1 = e.run(A, PA, max_new=32)
    show('cold', o1)
    check(o1['reason'] in ('stop', 'length') and o1['cached'] == 0 and o1['completion'] >= 1, 'cold prefill: no reuse')
    check(all(0 <= t < VOCAB for t in o1['ids']), 'ids in the used vocabulary')
    o2 = e.run(A, PA, max_new=32)
    show('same request again', o2)
    check(o2['cached'] == len(PA) - 1, 'the same request resumes at the full prefix (%d)' % o2['cached'])
    check(o2['ids'] == o1['ids'], 'the same request gives the same tokens')
    # 3: a fork at a shorter prefix = the same prompt cold
    PB = PA[:1500] + synth_ids(2, 300)
    ob = e.run(0xB000, PB, base=A, max_new=24)
    od = e.run(0xD000, PB, max_new=24)
    show('fork of A at 1500', ob)
    show('the same prompt cold', od)
    check(ob['cached'] == 1500 and od['cached'] == 0, 'the fork reuses its base\'s common prefix (%d)' % ob['cached'])
    check(ob['ids'] == od['ids'], 'a restart at 1500 gives the tokens of a cold prefill')
    # 4: a turn: the session's history and a suffix; only the suffix is prefilled
    P3 = PA + list(o2['ids']) + synth_ids(3, 60)
    o3 = e.run(A, P3, max_new=24)
    show('a turn on A', o3)
    check(o3['cached'] == len(PA) + len(o2['ids']), 'the turn prefills only its suffix (cached %d, expected %d)' % (
        o3['cached'], len(PA) + len(o2['ids'])))
    o3b = e.run(A, P3, max_new=24)
    check(o3b['cached'] == len(P3) - 1 and o3b['ids'] == o3['ids'], 'the turn again: the full prefix, the same tokens')
    # 5: evict and restore: two more sessions push A out of the pool to the disk tier
    for s in (0xB000, 0xD000):
        e.release(s)
    oe = e.run(0xE000, synth_ids(5, 5500), max_new=8)
    of = e.run(0xF000, synth_ids(6, 5500), max_new=8)
    show('filler E', oe)
    show('filler F', of)
    o4 = e.run(A, P3, max_new=24)
    show('A after eviction', o4)
    check(o4['cached'] == len(P3) - 1, 'after eviction: resumed at the full prefix (%d)' % o4['cached'])
    check(o4['ids'] == o3['ids'], 'after eviction: the same tokens')
    # 6: seeded sampling with logprobs
    PS = synth_ids(7, 400)
    s1 = e.run(0x5100, PS, max_new=24, temp=1.0, top_p=0.95, seed=7, n_top=5, keep=False)
    s2 = e.run(0x5200, PS, max_new=24, temp=1.0, top_p=0.95, seed=7, n_top=5, keep=False)
    show('sampled, seed 7', s1)
    check(s1['ids'] == s2['ids'] and s1['lp'] == s2['lp'], 'seeded sampling repeats (ids and logprobs)')
    check(len(s1['lp']) == len(s1['ids']) and all(lp <= 0 for lp in s1['lp']) and all(len(t) == 5 for t in s1['top']),
          'logprobs and 5 top logprobs a token')
    check(all(0 <= i < VOCAB for t in s1['top'] for i, _ in t), 'top logprob ids in the used vocabulary')
    # 7: cancel, a bad request
    oc = e.run(0xC000, synth_ids(8, 300), max_new=400, cancel_after=3)
    show('cancelled', oc)
    check(oc['reason'] == 'cancelled' and oc['completion'] < 400, 'cancel ends the request')
    ox = e.run(0xC100, [5, VOCAB, 7], max_new=4)
    check(ox['reason'] == 'error', 'an id >= 154856 is refused: %s' % ox['error'])
    for s in (0xC000, 0xE000, 0xF000):
        e.release(s)
    json.dump({'prompt': P3, 'ids': list(o3['ids']), 'max_new': 24}, open(a.state, 'w'))
    e.close()


if __name__ == '__main__':
    main()
    print('serve_test: %s' % ('passed' if not FAILS else 'FAILED: ' + '; '.join(FAILS)), flush=True)
    sys.exit(1 if FAILS else 0)
