#!/usr/bin/env python3
"""The front-end's smoke test (frontend/tools/smoke_test.py) checks the mock engine's replies; these are
its model-dependent checks restated for the real engine behind dsfront: a non-streamed answer, streamed
reasoning and content with reasoning-token usage, a tool call and the answer after its result (reusing
the previous turn), a stop string, an answer right after a client disconnects, concurrent streams.

  python3 engine/server/tests/frontend_e2e.py http://127.0.0.1:18893 ~/dsgb10-data/dev-api-key
"""
import http.client, json, sys, threading, time, urllib.parse

BASE = urllib.parse.urlparse(sys.argv[1])
KEY = open(sys.argv[2]).read().strip()
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def conn():
    return http.client.HTTPConnection(BASE.hostname, BASE.port, timeout=600)


def request(body):
    c = conn()
    c.request("POST", "/v1/chat/completions", body=json.dumps(body),
              headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"{}")


def stream(body, read_events=None):
    c = conn()
    c.request("POST", "/v1/chat/completions", body=json.dumps(dict(body, stream=True)),
              headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    r = c.getresponse()
    ev = []
    while True:
        line = r.fp.readline()
        if not line:
            break
        line = line.decode().strip()
        if not line.startswith("data: "):
            continue
        d = line[6:]
        ev.append(d if d == "[DONE]" else json.loads(d))
        if d == "[DONE]" or (read_events and len(ev) >= read_events):
            break
    if read_events:
        c.close()
    return ev


def collect(ev):
    o = {"content": "", "reasoning": "", "calls": {}, "finish": None, "usage": None}
    for e in ev:
        if e == "[DONE]":
            continue
        if e.get("usage"):
            o["usage"] = e["usage"]
        for ch in e.get("choices", []):
            d = ch.get("delta", {})
            o["content"] += d.get("content") or ""
            o["reasoning"] += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                c = o["calls"].setdefault(tc["index"], {"id": None, "name": "", "arguments": ""})
                if tc.get("id"):
                    c["id"] = tc["id"]
                f = tc.get("function") or {}
                c["name"] += f.get("name") or ""
                c["arguments"] += f.get("arguments") or ""
            if ch.get("finish_reason"):
                o["finish"] = ch["finish_reason"]
    return o


SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]

t0 = time.time()
s, b = request({"messages": [SYSTEM, {"role": "user", "content": "Say hello in five words."}], "max_tokens": 64})
m = b["choices"][0]["message"] if s == 200 else {}
check(s == 200 and m.get("content") and b["choices"][0]["finish_reason"] in ("stop", "length"),
      f"non-streaming answer: {m.get('content')!r} ({time.time() - t0:.1f} s)")

body = {"messages": [SYSTEM, {"role": "user", "content": "What is 17 * 23? Answer briefly."}], "max_completion_tokens": 600,
        "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": True}}
o = collect(stream(body))
check(o["reasoning"] and o["content"], f"streamed reasoning ({len(o['reasoning'])} chars) and content: {o['content'][:80]!r}")
u = o["usage"] or {}
check(u.get("completion_tokens_details", {}).get("reasoning_tokens", 0) > 0, f"usage with reasoning tokens: {u.get('completion_tokens_details')}")

msgs = [SYSTEM, {"role": "user", "content": "Use the read tool to read notes.txt, then tell me what it says."}]
body = {"messages": msgs, "tools": TOOLS, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": True},
        "max_tokens": 800}
o = collect(stream(body))
call = o["calls"].get(0)
check(o["finish"] == "tool_calls" and call and call["name"] == "read", f"tool call: {call}")
if call:
    first_total = o["usage"]["prompt_tokens"] + o["usage"]["completion_tokens"]
    msgs2 = msgs + [{"role": "assistant", "content": o["content"] or None, "reasoning_content": o["reasoning"],
                     "tool_calls": [{"id": call["id"], "type": "function", "function": {"name": "read", "arguments": call["arguments"]}}]},
                    {"role": "tool", "tool_call_id": call["id"], "content": "line one\nline two"}]
    o2 = collect(stream(dict(body, messages=msgs2)))
    check(o2["content"] and o2["finish"] == "stop", f"answer after the tool result: {o2['content'][:100]!r}")
    cached = o2["usage"]["prompt_tokens_details"]["cached_tokens"]
    check(cached >= first_total - 1,
          f"tool-result turn reuses the previous turn (cached {cached} of {o2['usage']['prompt_tokens']}, previous turn {first_total})")

s, b = request({"messages": [{"role": "user", "content": "Count from one to ten in words, separated by spaces."}], "stop": ["six"],
                "max_tokens": 64})
c = b["choices"][0]["message"]["content"] if s == 200 else ""
check(s == 200 and "six" not in c and "five" in c and b["choices"][0]["finish_reason"] == "stop", f"stop string cuts the output: {c!r}")

ev = stream({"messages": [{"role": "user", "content": "Write a long story about a lighthouse."}], "max_tokens": 3000}, read_events=20)
time.sleep(0.3)
t0 = time.time()
s, b = request({"messages": [{"role": "user", "content": "Still alive? Answer yes or no."}], "max_tokens": 16})
check(s == 200 and b["choices"][0]["message"]["content"],
      f"answers after a disconnect: {b['choices'][0]['message']['content']!r} ({time.time() - t0:.1f} s)")

res = [None] * 4


def worker(i):
    res[i] = collect(stream({"messages": [{"role": "user", "content": f"Name a colour. ({i})"}], "max_tokens": 32}))


th = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
t0 = time.time()
[t.start() for t in th]
[t.join() for t in th]
check(all(r and r["finish"] in ("stop", "length") and r["content"] for r in res),
      f"4 concurrent streams ({time.time() - t0:.1f} s): {[r['content'][:20] for r in res if r]}")
print(f"{len(FAILS)} failures")
sys.exit(1 if FAILS else 0)
