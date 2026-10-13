#!/usr/bin/env python3
"""Revisiting a long session through the front-end (docs/engine.md, Serving: the disk tier). For each
size: a conversation of one long user message (this repository's sources) and its answer; other
conversations after it, so that the server evicts it (with a disk tier: spills it); then one more short
user turn, timed at the client to its first token. The same conversation sent again with another first
line, so that the front-end finds no session to continue, gives the cold prefill to compare with.

  python3 engine/server/tests/revisit.py --key-file ~/dsgb10-data/dev-api-key [--sizes 32000,150000]

Run it against a server whose store keeps little (e.g. --session-gib 0.5) so that one session pushes the
other out; rank 0's log says where each revisit came from ("back from disk", "reused N (end)").
"""
import argparse
import http.client
import json
import os
import sys
import time
import urllib.parse

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))


def corpus():
    out = []
    for d in ('engine', 'docs', 'frontend/src', 'ref'):
        for base, _, files in sorted(os.walk(os.path.join(ROOT, d))):
            for f in sorted(files):
                if f.endswith(('.cu', '.h', '.cpp', '.md', '.rs', '.py')):
                    try:
                        out.append('// %s\n%s' % (os.path.relpath(os.path.join(base, f), ROOT), open(os.path.join(base, f)).read()))
                    except UnicodeDecodeError:
                        pass
    return '\n'.join(out)


class Client:
    def __init__(self, url, key):
        u = urllib.parse.urlparse(url)
        self.host, self.port, self.key = u.hostname, u.port, key

    def chat(self, messages, max_tokens=16):
        """A streamed greedy request: (seconds to the first content, seconds in all, text, usage)."""
        body = {'model': 'deepseek-v4.1-flash', 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0,
                'stream': True, 'stream_options': {'include_usage': True}}
        c = http.client.HTTPConnection(self.host, self.port, timeout=1800)
        t0 = time.time()
        c.request('POST', '/v1/chat/completions', body=json.dumps(body),
                  headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError('HTTP %d: %s' % (r.status, r.read()[:300]))
        t_first, text, usage = None, '', None
        for line in r:
            line = line.strip()
            if not line.startswith(b'data: ') or line == b'data: [DONE]':
                continue
            ev = json.loads(line[6:])
            if ev.get('usage'):
                usage = ev['usage']
            for ch in ev.get('choices') or []:
                d = ch.get('delta') or {}
                piece = (d.get('content') or '') + (d.get('reasoning_content') or '')
                if piece and t_first is None:
                    t_first = time.time() - t0
                text += d.get('content') or ''
        c.close()
        return t_first, time.time() - t0, text, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:18893')
    ap.add_argument('--key-file', required=True)
    ap.add_argument('--sizes', default='32000,150000', help='approximate prompt tokens')
    ap.add_argument('--chars-per-token', type=float, default=3.3)
    a = ap.parse_args()
    cl = Client(a.url, open(os.path.expanduser(a.key_file)).read().strip())
    text = corpus()
    sizes = [int(x) for x in a.sizes.split(',')]
    convs = []
    for i, n in enumerate(sizes):
        doc = text[(i * 7919) % 100000:][:int(n * a.chars_per_token)]
        convs.append([{'role': 'system', 'content': 'Conversation %d.' % i},
                      {'role': 'user', 'content': 'A file from a repository:\n\n' + doc + '\n\nWhat is it? One line.'}])
    rows = []
    for i, m in enumerate(convs):  # first visits, in turn: each pushes the one before out of the store
        tf, tt, ans, u = cl.chat(m)
        print('first visit %d: prompt %s, cached %s, first token %.2f s' % (
            i, u and u.get('prompt_tokens'), u and (u.get('prompt_tokens_details') or {}).get('cached_tokens'), tf))
        m.append({'role': 'assistant', 'content': ans})
        sys.stdout.flush()
    tf, tt, _, u = cl.chat([{'role': 'user', 'content': 'Say hello.'}])  # pushes the last one out
    for i, m in enumerate(convs):
        m2 = m + [{'role': 'user', 'content': 'And in one word?'}]
        tf, tt, _, u = cl.chat(m2)
        cached = u and (u.get('prompt_tokens_details') or {}).get('cached_tokens')
        print('revisit %d: prompt %s, cached %s, first token %.3f s' % (i, u and u.get('prompt_tokens'), cached, tf))
        cold = [{'role': 'system', 'content': 'Conversation %d, again.' % i}] + m2[1:]
        tf2, _, _, u2 = cl.chat(cold)
        print('  the same cold: prompt %s, cached %s, first token %.3f s' % (
            u2 and u2.get('prompt_tokens'), u2 and (u2.get('prompt_tokens_details') or {}).get('cached_tokens'), tf2))
        rows.append((u and u.get('prompt_tokens'), cached, tf, tf2))
        sys.stdout.flush()
    for p, c, t, t2 in rows:
        print('revisit\t%s\t%s\t%.3f\t%.3f' % (p, c, t, t2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
