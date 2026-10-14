#!/usr/bin/env python3
"""altarfront against the real engine (docs/serving.md): the OpenAI surface over HTTP on a test port, with rank 0
behind it in place of the mock. Greedy chat completions, streamed and not: the same conversation twice gives the
same text with the prompt reused (cached_tokens), a second turn reuses the first (cached_tokens > 0), a client
that disconnects mid-stream is cancelled and the next request is served. With synthetic routed experts the text is
meaningless; the checks are about the engine behind the front-end, not the words.

  front_test.py http://127.0.0.1:18895 KEY_FILE
"""
import http.client
import json
import sys
import urllib.parse

BASE = urllib.parse.urlparse(sys.argv[1])
KEY = open(sys.argv[2]).read().strip()
FAILS = []


def check(cond, what):
    print(('ok   ' if cond else 'FAIL ') + what, flush=True)
    if not cond:
        FAILS.append(what)


def post(body, stream=False, read_events=None):
    c = http.client.HTTPConnection(BASE.hostname, BASE.port, timeout=600)
    c.request('POST', '/v1/chat/completions', body=json.dumps(dict(body, stream=stream)),
              headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + KEY})
    r = c.getresponse()
    if not stream:
        d = json.loads(r.read())
        c.close()
        return r.status, d
    out = {'reasoning': '', 'content': '', 'usage': None, 'events': 0}
    while True:
        line = r.fp.readline()
        if not line:
            break
        line = line.decode().strip()
        if not line.startswith('data: '):
            continue
        if line == 'data: [DONE]':
            break
        e = json.loads(line[6:])
        out['events'] += 1
        if e.get('usage'):
            out['usage'] = e['usage']
        for ch in e.get('choices', []):
            out['reasoning'] += ch['delta'].get('reasoning_content') or ''
            out['content'] += ch['delta'].get('content') or ''
        if read_events and out['events'] >= read_events:
            break
    c.close()
    return r.status, out


def text(m):
    return (m.get('reasoning_content') or '') + '|' + (m.get('content') or '')


def main():
    msgs = [{'role': 'system', 'content': 'You are a coding agent. ' + 'Be careful. ' * 200},
            {'role': 'user', 'content': 'List the files of the repository and say what each one does.'}]
    body = {'model': 'altar-1', 'messages': msgs, 'max_tokens': 24, 'temperature': 0}
    s1, b1 = post(body)
    check(s1 == 200 and b1['choices'][0]['message'] is not None, 'a chat completion: 200')
    u1 = b1.get('usage', {})
    print('  usage', u1, flush=True)
    s2, b2 = post(body)
    u2 = b2.get('usage', {})
    c2 = (u2.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
    check(s2 == 200 and text(b2['choices'][0]['message']) == text(b1['choices'][0]['message']),
          'the same request again: the same text')
    check(c2 >= u2.get('prompt_tokens', 0) - 1 > 0, 'the same request again: the prompt reused (%d of %d)' % (
        c2, u2.get('prompt_tokens', 0)))
    s3, o3 = post(body, stream=True)
    check(s3 == 200 and o3['reasoning'] + '|' + o3['content'] == text(b1['choices'][0]['message']),
          'streamed: the same text')
    m1 = b1['choices'][0]['message']
    msgs2 = msgs + [{'role': 'assistant', 'content': m1.get('content') or '', 'reasoning_content': m1.get('reasoning_content')},
                    {'role': 'user', 'content': 'Now the tests.'}]
    s4, b4 = post(dict(body, messages=msgs2))
    u4 = b4.get('usage', {})
    c4 = (u4.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
    print('  turn 2 usage', u4, flush=True)
    check(s4 == 200 and c4 >= u1.get('prompt_tokens', 0), 'a second turn reuses the first (cached %d, first prompt %d)' % (
        c4, u1.get('prompt_tokens', 0)))
    post(dict(body, max_tokens=400), stream=True, read_events=3)  # the client goes away
    s5, b5 = post(body)
    check(s5 == 200 and text(b5['choices'][0]['message']) == text(b1['choices'][0]['message']),
          'after a client went away mid-stream: served, the same text')


if __name__ == '__main__':
    main()
    print('front_test: %s' % ('passed' if not FAILS else 'FAILED: ' + '; '.join(FAILS)), flush=True)
    sys.exit(1 if FAILS else 0)
