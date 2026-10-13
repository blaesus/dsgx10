#!/usr/bin/env python3
"""End-to-end checks of `dsfront serve` against `dsfront mock-engine` (stdlib only).

    dsfront mock-engine --socket S &
    dsfront serve --engine-socket S --api-key-file K &
    smoke_test.py http://127.0.0.1:18890 K [LOG]     # LOG: dsfront serve's stderr, for its "plan:" lines

Covers the OpenAI surface pi uses: models, auth, streaming and not, reasoning_content, tool calls
and a tool-result turn (with prefix reuse reported in cached_tokens), usage, stop strings,
max_tokens, the loop guard, bad requests, a client that disconnects, and concurrent requests.
"""
import http.client
import json
import sys
import threading
import time
import urllib.parse

BASE = urllib.parse.urlparse(sys.argv[1])
KEY = open(sys.argv[2]).read().strip()
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def conn():
    return http.client.HTTPConnection(BASE.hostname, BASE.port, timeout=60)


def request(method, path, body=None, key=KEY):
    c = conn()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    c.request(method, path, body=body if isinstance(body, (bytes, str)) or body is None else json.dumps(body), headers=headers)
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, data


def stream(body, read_events=None):
    """Streamed chat: (events, raw lines). Stops after read_events events if given (and drops the connection)."""
    c = conn()
    c.request("POST", "/v1/chat/completions", body=json.dumps(dict(body, stream=True)),
              headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    r = c.getresponse()
    assert r.status == 200, (r.status, r.read())
    events = []
    while True:
        line = r.fp.readline()
        if not line:
            break
        line = line.decode().rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            events.append("[DONE]")
            break
        events.append(json.loads(data))
        if read_events and len(events) >= read_events:
            c.close()
            return events
    c.close()
    return events


def collect(events):
    out = {"reasoning": "", "content": "", "calls": {}, "finish": None, "usage": None}
    for e in events:
        if e == "[DONE]" or "error" in e:
            continue
        if e.get("usage"):
            out["usage"] = e["usage"]
        for ch in e["choices"]:
            d = ch["delta"]
            out["reasoning"] += d.get("reasoning_content") or ""
            out["content"] += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                slot = out["calls"].setdefault(tc["index"], {"id": None, "name": "", "arguments": ""})
                slot["id"] = slot["id"] or tc.get("id")
                slot["name"] += tc["function"].get("name") or ""
                slot["arguments"] += tc["function"].get("arguments") or ""
            if ch.get("finish_reason"):
                out["finish"] = ch["finish_reason"]
    return out


TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
SYSTEM = {"role": "system", "content": "You are a coding agent. " + "Be careful. " * 50}


def main():
    s, b = request("GET", "/v1/models")
    check(s == 200 and b["data"][0]["id"], "models lists the served model")
    s, b = request("GET", "/v1/models", key="wrong")
    check(s == 401, "a wrong key is rejected")
    s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, key=None)
    check(s == 401, "a missing key is rejected")

    s, b = request("POST", "/v1/chat/completions", {"messages": [SYSTEM, {"role": "user", "content": "Hello there"}]})
    check(s == 200 and b["choices"][0]["message"]["content"] == "Mock reply to: Hello there", "non-streaming chat answer")
    check(b["choices"][0]["finish_reason"] == "stop" and b["usage"]["completion_tokens"] > 0, "finish_reason stop and usage")
    check("reasoning_content" not in b["choices"][0]["message"], "no reasoning_content when thinking is off")

    # thinking, streamed, pi-style kwargs
    body = {"messages": [SYSTEM, {"role": "user", "content": "Think first"}], "max_completion_tokens": 200,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}}
    ev = stream(body)
    o = collect(ev)
    check(o["reasoning"].startswith("Mock reasoning") and o["content"] == "Mock reply to: Think first", "streamed reasoning and content")
    check(o["finish"] == "stop" and o["usage"] and o["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0, "stream usage with reasoning tokens")
    check(ev[-1] == "[DONE]" and ev[0]["choices"][0]["delta"].get("role") == "assistant", "stream starts with the role and ends with [DONE]")

    # an agent turn: tool call, then the tool result, reusing the prefix
    msgs = [SYSTEM, {"role": "user", "content": 'Please read it. MOCK_TOOL read {"path": "notes.txt"}'}]
    body = {"messages": msgs, "tools": TOOLS, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": True}}
    o = collect(stream(body))
    call = o["calls"].get(0)
    check(o["finish"] == "tool_calls" and call and call["name"] == "read" and json.loads(call["arguments"]) == {"path": "notes.txt"},
          "streamed tool call with finish_reason tool_calls")
    check(call and call["id"].startswith("call_"), "tool call has an id")
    first_prompt = o["usage"]["prompt_tokens"]
    first_total = first_prompt + o["usage"]["completion_tokens"]
    msgs2 = msgs + [{"role": "assistant", "content": o["content"] or None, "reasoning_content": o["reasoning"],
                     "tool_calls": [{"id": call["id"], "type": "function", "function": {"name": "read", "arguments": call["arguments"]}}]},
                    {"role": "tool", "tool_call_id": call["id"], "content": "line one\nline two"}]
    o2 = collect(stream(dict(body, messages=msgs2)))
    check(o2["content"] == "The tool returned: line one line two" and o2["finish"] == "stop", "answer after the tool result")
    cached = o2["usage"]["prompt_tokens_details"]["cached_tokens"]
    check(cached >= first_total - 1, f"tool-result turn reuses the whole previous turn (cached {cached} of {o2['usage']['prompt_tokens']}, previous turn {first_total})")

    # stop strings, max_tokens, loop guard
    s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello stopword here"}], "stop": ["stopword"]})
    check(b["choices"][0]["message"]["content"] == "Mock reply to: Hello " and b["choices"][0]["finish_reason"] == "stop", "stop string cuts the output")
    s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "MOCK_LONG 500"}], "max_tokens": 5})
    check(b["choices"][0]["finish_reason"] == "length" and b["usage"]["completion_tokens"] == 5, "max_tokens ends with length")
    t0 = time.time()
    s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "MOCK_LOOP"}], "max_tokens": 5000})
    n = b["usage"]["completion_tokens"]
    check(b["choices"][0]["finish_reason"] == "stop" and n < 400, f"loop guard stops a repeating output ({n} tokens, {time.time() - t0:.1f} s)")

    # bad requests
    for what, body in [("invalid JSON", b"{"), ("image input", {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}),
                       ("n = 2", {"messages": [{"role": "user", "content": "x"}], "n": 2}), ("no messages", {"messages": []}),
                       ("unknown role", {"messages": [{"role": "robot", "content": "x"}]})]:
        s, b = request("POST", "/v1/chat/completions", body)
        check(s == 400 and "error" in b, f"400 for {what}")

    # a client that goes away mid-stream, then the server still answers
    ev = stream({"messages": [{"role": "user", "content": "MOCK_LONG 3000"}]}, read_events=20)
    check(len(ev) == 20, "read part of a long stream, then disconnect")
    time.sleep(0.3)
    s, b = request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "still alive?"}]})
    check(s == 200 and b["choices"][0]["message"]["content"] == "Mock reply to: still alive?", "server answers after a disconnect")

    # where a request starts: two pi-like conversations whose system text differs only in its last
    # line (the time) share all before it; dsfront's log (the third argument) says so, in counts only
    sysp = "You are an expert coding assistant. Read files, run commands and edit code.\n" * 40

    def pi(when, msg):
        return {"messages": [{"role": "system", "content": f"{sysp}Current date and time: Monday, October 12, 2026 at {when}"},
                             {"role": "user", "content": msg}], "max_tokens": 8}

    s1, b1 = request("POST", "/v1/chat/completions", pi("09:14:08 AM", "fix the build"))
    s2, b2 = request("POST", "/v1/chat/completions", pi("09:15:31 AM", "write the tests"))
    u = b2.get("usage") or {}
    c2, p2 = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0), u.get("prompt_tokens", 0)
    check(s1 == 200 and s2 == 200 and 0.9 * p2 < c2 < p2,
          f"a conversation with another's system text but a later time reuses what comes before the time (cached {c2} of {p2})")
    if len(sys.argv) > 3:
        time.sleep(0.2)
        plans = [l for l in open(sys.argv[3]).read().splitlines() if " plan: " in l]
        check(len(plans) >= 2 and ", new: " in plans[-2] and ", a fork of " in plans[-1] and " ids in common" in plans[-1],
              f"the log says where each started: {plans[-2:]}")

    # concurrency
    results = [None] * 8

    def worker(i):
        results[i] = collect(stream({"messages": [{"role": "user", "content": f"parallel {i} MOCK_LONG 100"}], "max_tokens": 1000}))

    th = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    t0 = time.time()
    [t.start() for t in th]
    [t.join() for t in th]
    check(all(r and r["finish"] == "stop" and r["content"].count("word") == 100 for r in results), f"8 concurrent streams ({time.time() - t0:.1f} s)")

    print(f"{len(FAILS)} failures")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
