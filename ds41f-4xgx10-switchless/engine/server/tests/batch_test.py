#!/usr/bin/env python3
"""Batched decoding (docs/batching.md, B2) through rank 0's socket, for a server with --slots N (N >= 2):
requests that decode together, a row of each per step, must each give what they give alone, bit for bit
(greedy and seeded sampling, tokens and log-probabilities); a request that comes while others decode gets
its first token after its own prefill, not after their decode; more requests than slots take turns
(--slice > 0) or wait for a slot (--slice 0); a cancel and a stop id inside a batch end only their own
request; a kept session whose slot went to another continues from the store.

  python3 engine/server/tests/batch_test.py --socket S --slots 4
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_test import Mux, ttft  # noqa: E402
from serve_test import load_npy_i32  # noqa: E402

S0 = 0x6261746300000000  # 'batc'

fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def same(name, rec, ref):
    d = next((i for i, (a, b) in enumerate(zip(rec['ids'], ref['ids'])) if a != b), None)
    check(rec['ids'] == ref['ids'], '%s: its tokens equal its run alone (%d tokens, %s)%s' % (
        name, len(rec['ids']), rec['done']['reason'], '' if rec['ids'] == ref['ids'] else
        ': first difference at %s of %d' % (d, len(ref['ids']))))
    check(rec['lp'] == ref['lp'], '%s: and so do its log-probabilities' % name)


def overlap(e, rids):
    """For each request, how many of its tokens came while each other request was between its first token
    and its end."""
    out = {}
    for r in rids:
        n = []
        for o in rids:
            if o == r:
                continue
            t0, t1 = e.recs[o]['t_tok'][0], e.recs[o]['done']['t']
            n.append(sum(t0 < t < t1 for t in e.recs[r]['t_tok']))
        out[r] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--slots', type=int, default=4, help="the server's --slots")
    a = ap.parse_args()
    T = load_npy_i32(a.tokens)
    e = Mux(a.socket)
    print('engine: %s, context %d, %d slots' % (e.name, e.max_ctx, a.slots))
    # six requests: greedy and sampled, with log-probabilities (C with the top 5 too); no stop ids (Mux)
    R = [('A', dict(prompt=T[:1000], max_new=300, n_top=0)),
         ('B', dict(prompt=T[20000:24000], max_new=240, n_top=0, temp=0.7, seed=7)),
         ('C', dict(prompt=T[40000:42000], max_new=200, n_top=5)),
         ('D', dict(prompt=T[60000:61500], max_new=320, n_top=0, temp=0.9, top_p=0.95, seed=11)),
         ('E', dict(prompt=T[80000:80800], max_new=260, n_top=0)),
         ('F', dict(prompt=T[100000:103000], max_new=180, n_top=0, temp=0.6, seed=3))]
    K = dict(R)
    used = []

    def start(name, sid, **over):
        kw = dict(K[name])
        kw.update(over)
        used.append(S0 + sid)
        return e.start(name, S0 + sid, kw.pop('prompt'), **kw)

    print('1. each alone, in a fresh session')
    ref = {}
    for i, (name, kw) in enumerate(R):
        r = start(name, 1 + i)
        e.pump(lambda: e.finished(r))
        ref[name] = e.recs[r]
        d = ref[name]['done']
        print('  %s alone: prompt %d, %d tokens, %s; first token after %.3f s, decode %.1f tok/s' % (
            name, d['prompt'], d['completion'], d['reason'], ttft(ref[name]),
            (d['completion'] - 1) / max(1e-9, d['decode_us'] / 1e6)))
    alone_s = sum(ref[n]['done']['decode_us'] / 1e6 for n, _ in R[:a.slots])
    alone_tok = sum(ref[n]['done']['completion'] - 1 for n, _ in R[:a.slots])

    n_tog = min(a.slots, len(R))
    names = [n for n, _ in R[:n_tog]]
    print('2. %s together (as many as there are slots), submitted at once' % ', '.join(names))
    t0 = time.time()
    rids = [start(n, 11 + i) for i, n in enumerate(names)]
    e.pump(lambda: e.finished(*rids))
    wall = max(e.recs[r]['done']['t'] for r in rids) - t0
    for r in rids:
        same(e.recs[r]['name'], e.recs[r], ref[e.recs[r]['name']])
    ov = overlap(e, rids)
    check(all(min(n) > 0 for n in ov.values()), 'each decoded while each other was in flight: %s' % (
        '; '.join('%s %s' % (e.recs[r]['name'], ov[r]) for r in rids)))
    tok = sum(e.recs[r]['done']['completion'] - 1 for r in rids)
    print('  together: %d tokens in %.2f s from the first submit (%.1f tok/s); alone, one after another: %.1f tok/s '
          'decoding (%d tokens in %.2f s)' % (tok, wall, tok / wall, alone_tok / max(1e-9, alone_s), alone_tok, alone_s))
    for r in rids:
        rec = e.recs[r]
        print('    %s: first token after %.3f s, done after %.2f s' % (rec['name'], ttft(rec), rec['done']['t'] - t0))

    print('3. staggered: B comes 30 tokens into A\'s decode, C at B\'s first token, D at C\'s')
    stag = names[:4]
    rids = [start(stag[0], 21)]
    e.pump(lambda: e.ntok(rids[0]) >= 30)
    for i, n in enumerate(stag[1:]):
        prev = rids[-1]
        if i > 0:
            e.pump(lambda: e.ntok(prev) >= 1)
        rids.append(start(n, 22 + i))
    e.pump(lambda: e.finished(*rids))
    for r in rids:
        same(e.recs[r]['name'], e.recs[r], ref[e.recs[r]['name']])
    for r in rids[1:]:
        rec = e.recs[r]
        lag = ttft(rec) - ttft(ref[rec['name']])
        check(lag < 1.0, "%s's first token %.3f s after its submit, %.3f s later than alone" % (rec['name'], ttft(rec), lag))
    first = e.recs[rids[0]]
    during = sum(e.recs[rids[1]]['t_tok'][0] < t < e.recs[rids[1]]['done']['t'] for t in first['t_tok'])
    check(during > 0, '%s decoded while %s was in flight: %d of its tokens' % (first['name'], e.recs[rids[1]]['name'], during))

    print('4. all %d at once: more than the %d slots' % (len(R), a.slots))
    t0 = time.time()
    rids = [start(n, 31 + i) for i, (n, _) in enumerate(R)]
    e.pump(lambda: e.finished(*rids))
    for r in rids:
        same(e.recs[r]['name'], e.recs[r], ref[e.recs[r]['name']])
    for r in rids:
        rec = e.recs[r]
        print('    %s: first token after %.3f s, done after %.2f s' % (rec['name'], ttft(rec), rec['done']['t'] - t0))

    print('5. a cancel and a stop id inside a batch: A, B and C together; B cancelled at its 40th token; C stops')
    cref = ref['C']['ids']
    sid = cref[60]
    k = cref.index(sid)
    r_a = start('A', 41)
    r_b = start('B', 42)
    r_c = start('C', 43, stop=(sid,))
    e.pump(lambda: e.ntok(r_b) >= 40)
    e.cancel(r_b)
    e.pump(lambda: e.finished(r_a, r_b, r_c))
    same('A', e.recs[r_a], ref['A'])
    db, dc = e.recs[r_b]['done'], e.recs[r_c]['done']
    check(db['reason'] == 'cancelled' and e.recs[r_b]['ids'] == ref['B']['ids'][:e.ntok(r_b)],
          'B %s after %d tokens, its tokens B\'s alone as far as they go' % (db['reason'], e.ntok(r_b)))
    check(dc['reason'] == 'stop' and e.recs[r_c]['ids'] == cref[:k + 1],
          'C stopped at its stop id %d: %d tokens (%s; alone it is at %d)' % (sid, e.ntok(r_c), dc['reason'], k))

    print('6. a kept session whose slot went to another continues from the store, beside another\'s turn')
    a2 = list(K['A']['prompt']) + list(ref['A']['ids']) + T[50000:50064]
    r1 = e.start('A+', S0 + 11, a2, max_new=100, n_top=0)
    b2 = list(K['B']['prompt']) + list(ref['B']['ids']) + T[52000:52032]
    r2 = e.start('B+', S0 + 12, b2, max_new=100, n_top=0)
    e.pump(lambda: e.finished(r1, r2))
    # the session holds its prompt and its tokens (the last one's KV too, or not: base - 1); a restart on an
    # odd position may step back one more
    for r, base in ((r1, len(K['A']['prompt']) + len(ref['A']['ids'])), (r2, len(K['B']['prompt']) + len(ref['B']['ids']))):
        d = e.recs[r]['done']
        check(d['cached'] >= base - 2 and d['reason'] == 'length' and d['completion'] == 100,
              '%s: cached %d of %d (its session holds %d), %d tokens, %s' % (
                  e.recs[r]['name'], d['cached'], d['prompt'], base, d['completion'], d['reason']))

    for s in sorted(set(used)):
        e.release(s)
    print('batch_test: %d failed' % fails)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
