#!/usr/bin/env python3
"""sparkDash 1.8.8 against the engine: sdrun.mjs's records joined with rank 0's log (a request's prompt and
output token counts find its line, in order), as tables.
  sdreport.py OUT        (OUT/decode/{sd.jsonl,run4.log}, OUT/prefill/{sd.jsonl,run4.log})"""
import json
import os
import re
import statistics as st
import sys

D = sys.argv[1]


def engine(path):
    """rank 0's requests in order: prompt, reused, prefill ms, tokens, decode ms, verify steps, accepted per
    step, GPU ms per step (the mean over steps of the step's rows' mean)"""
    reqs, drafts = [], {}
    if not os.path.exists(path):
        return reqs
    for line in open(path, errors='replace'):
        m = re.search(r'serve: req (\d+) drafts: (\d+) verify steps, ([\d.]+) drafts verified and ([\d.]+) accepted per '
                      r'step.*GPU ms by rows:(.*)', line)
        if m:
            rows = re.findall(r'(\d+): ([\d.]+) \((\d+)\)', m.group(5))
            n = sum(int(c) for _, _, c in rows)
            gpu = sum(float(ms) * int(c) for _, ms, c in rows) / n if n else None
            drafts[int(m.group(1))] = dict(v=int(m.group(2)), verified=float(m.group(3)), acc=float(m.group(4)), gpu=gpu)
            continue
        m = re.search(r'serve: req (\d+) session \w+: prompt (\d+), reused (\d+) \((\w+)\), prefill ([\d.]+) ms '
                      r'\(([\d.]+) tok/s\), (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\), (\w+)', line)
        if m:
            r = int(m.group(1))
            reqs.append(dict(req=r, prompt=int(m.group(2)), reused=int(m.group(3)), how=m.group(4),
                             pf_ms=float(m.group(5)), pf_tps=float(m.group(6)), tokens=int(m.group(7)),
                             dec_ms=float(m.group(8)), dec_tps=float(m.group(9)), why=m.group(10), **drafts.get(r, {})))
    return reqs


def join(recs, reqs, match_tokens=True):
    """each record's engine request: the next one with its prompt length (and output length)"""
    i = 0
    for x in recs:
        x['eng'] = None
        if not x.get('promptTokens'):
            continue
        for j in range(i, len(reqs)):
            e = reqs[j]
            if e['prompt'] == x['promptTokens'] and (not match_tokens or e['tokens'] == x.get('completionTokens')):
                x['eng'], i = e, j + 1
                break


def med(v):
    return st.median(v) if v else float('nan')


def load(mode):
    p = os.path.join(D, mode, 'sd.jsonl')
    recs = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    reqs = engine(os.path.join(D, mode, 'run4.log'))
    return recs, reqs


def steps(v):
    """the engine's view of a set of requests: tokens per verify step, wall ms per verify step (the decode's
    time from the first token to the end, over its verify steps), GPU ms per step"""
    e = [x['eng'] for x in v if x.get('eng') and x['eng'].get('v')]
    if not e:
        return dict(tps=float('nan'), wall=float('nan'), gpu=float('nan'), acc=float('nan'), n=0)
    V = sum(x['v'] for x in e)
    return dict(tps=sum((x['tokens'] - 1) for x in e) / V, wall=sum(x['dec_ms'] for x in e) / V,
                gpu=sum(x['gpu'] * x['v'] for x in e) / V, acc=sum(x['acc'] * x['v'] for x in e) / V, n=len(e),
                eng_tps=st.mean(x['dec_tps'] for x in e))


recs, reqs = load('decode')
join(recs, reqs)
if recs:
    print('## decode: sparkDash c1 (256 tokens, greedy, thinking off), %d records, %d engine requests, %d joined' % (
        len(recs), len(reqs), sum(1 for x in recs if x.get('eng'))))
    w = [x for x in recs if x['phase'] == 'warm']
    print('warm-ups (discarded): ' + ', '.join('%.2f' % x['decodeTps'] for x in w))
    print('| type | sparkDash tok/s, median of %d (min-max) | first token ms (median) | texts | engine: tokens/step | '
          'ms per verify step (wall) | GPU ms per step | engine tok/s |' % max(
              1, len([x for x in recs if x['phase'] == 'c1' and x['kind'] == 'prose'])))
    print('|---|---|---|---|---|---|---|---|')
    for k in ('prose', 'code', 'structured', 'json'):
        v = [x for x in recs if x['phase'] == 'c1' and x['kind'] == k]
        if not v:
            continue
        t = [x['decodeTps'] for x in v]
        s = steps(v)
        print('| %s | %.1f (%.1f-%.1f) | %.0f | %d distinct | %.2f | %.2f | %.2f | %.1f |' % (
            k, med(t), min(t), max(t), med([x['ttftMs'] for x in v]), len({x['text'] for x in v}), s['tps'],
            s['wall'], s['gpu'], s.get('eng_tps', float('nan'))))
    print('per run: ' + '; '.join('%s %s' % (k, ' '.join('%.2f' % x['decodeTps'] for x in recs
                                                          if x['phase'] == 'c1' and x['kind'] == k))
                                  for k in ('prose', 'code', 'structured', 'json')))
    errs = [x for x in recs if x.get('error')]
    if errs:
        print('errors: %d: %s' % (len(errs), errs[0]['error']))
    print()
    print('## catalogs at c1 (each prompt: the 32-token warm-up, then 256 tokens through runStreamingRequest)')
    print('| catalog | prompts | tok/s: mean of prompts | median | min-max | sum of tokens / sum of time | '
          'tokens/step | ms per verify step | GPU ms per step |')
    print('|---|---|---|---|---|---|---|---|---|')
    allv = []
    for k in ('prose', 'structural', 'code'):
        v = [x for x in recs if x['phase'] == 'catalog' and x['kind'] == k and not x.get('error')]
        if not v:
            continue
        allv += v
    for k, v in [(k, [x for x in allv if x['kind'] == k]) for k in ('prose', 'structural', 'code')] + [
            ('prose + structural', [x for x in allv if x['kind'] != 'code']), ('all', allv)]:
        if not v:
            continue
        t = [x['decodeTps'] for x in v]
        agg = sum(x['decodeTokens'] for x in v) / sum(x['decodeMs'] for x in v) * 1e3
        s = steps(v)
        print('| %s | %d | %.1f | %.1f | %.1f-%.1f | %.1f | %.2f | %.2f | %.2f |' % (
            k, len(v), st.mean(t), med(t), min(t), max(t), agg, s['tps'], s['wall'], s['gpu']))
    print()

recs, reqs = load('prefill')
join(recs, reqs, match_tokens=False)
if recs:
    print('## cold prefill (8 new tokens; sparkDash tok/s = prompt tokens / time to the first token), %d records, '
          '%d engine requests, %d joined' % (len(recs), len(reqs), sum(1 for x in recs if x.get('eng'))))
    print('| run | target | prompt tokens | first token ms | sparkDash tok/s | engine: prefill ms | engine tok/s | '
          'reused |')
    print('|---|---|---|---|---|---|---|---|')
    for x in recs:
        e = x.get('eng') or {}
        name = '%s %s' % (x['phase'], x.get('pass', x.get('kind', '')))
        print('| %s | %s | %s | %.0f | %.0f | %s | %s | %s |' % (
            name, x.get('target'), x.get('promptTokens'), x.get('ttftMs') or 0, x.get('prefillTps') or 0,
            '%.0f' % e['pf_ms'] if e else '-', '%.0f' % e['pf_tps'] if e else '-',
            '%d (%s)' % (e['reused'], e['how']) if e else '-'))
