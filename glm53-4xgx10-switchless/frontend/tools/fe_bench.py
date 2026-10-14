#!/usr/bin/env python3
"""Front-end overhead against `glmfront mock-engine --delay-ms 0 --prefill-tps 0` (stdlib only).

    fe_bench.py http://127.0.0.1:18890 KEYFILE

Measures (1) streamed decode throughput through the whole front-end (SSE parsing included), and
(2) time to first event for a long agent conversation: cold (render, tokenise everything) and
warm (the next turn of the same conversation, which reuses the session's ids and tokenises only
the new tool result).
"""
import http.client
import json
import random
import sys
import time
import urllib.parse

BASE = urllib.parse.urlparse(sys.argv[1])
KEY = open(sys.argv[2]).read().strip()


def stream(body):
    c = http.client.HTTPConnection(BASE.hostname, BASE.port, timeout=600)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body=json.dumps(dict(body, stream=True, stream_options={"include_usage": True})),
              headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    r = c.getresponse()
    first = None
    usage = None
    out = {"content": "", "reasoning": "", "calls": []}
    while True:
        line = r.fp.readline()
        if not line:
            break
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        e = json.loads(data)
        if e.get("usage"):
            usage = e["usage"]
        for ch in e.get("choices", []):
            d = ch["delta"]
            if first is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                first = time.time() - t0
            out["content"] += d.get("content") or ""
            out["reasoning"] += d.get("reasoning_content") or ""
            out["calls"] += d.get("tool_calls") or []
    c.close()
    return first, time.time() - t0, usage, out


TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def main():
    # (1) decode throughput
    first, total, usage, _ = stream({"messages": [{"role": "user", "content": "MOCK_LONG 20000"}], "max_tokens": 32768})
    n = usage["completion_tokens"]
    print(f"decode: {n} tokens streamed in {total - first:.2f} s = {n / (total - first):.0f} tok/s through the front-end")

    # (2) a long agent conversation: 40 tool rounds with ~2.5k-token results each
    rng = random.Random(1)
    words = "def class return import self value index error result config path file data line".split()

    def blob(n):
        return "\n".join(" ".join(rng.choice(words) for _ in range(12)) for _ in range(n))

    msgs = [{"role": "system", "content": "You are a coding agent.\n" + blob(200)}, {"role": "user", "content": "Fix the bug."}]
    for i in range(40):
        cid = f"call_{i}"
        msgs.append({"role": "assistant", "content": None, "reasoning_content": f"Step {i}: read the next file.",
                     "tool_calls": [{"id": cid, "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": f"src/f{i}.py"})}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": blob(250)})
    body = {"messages": msgs, "tools": TOOLS, "max_tokens": 64, "chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}}
    first, total, usage, out = stream(body)
    print(f"cold: {usage['prompt_tokens']} prompt tokens, first event after {first * 1000:.0f} ms (cached {usage['prompt_tokens_details']['cached_tokens']})")
    msgs.append({"role": "assistant", "content": out["content"] or None, "reasoning_content": out["reasoning"]})
    msgs.append({"role": "user", "content": "And now? " + blob(20)})
    first, total, usage, out = stream(dict(body, messages=msgs))
    print(f"warm: {usage['prompt_tokens']} prompt tokens, first event after {first * 1000:.0f} ms (cached {usage['prompt_tokens_details']['cached_tokens']})")


if __name__ == "__main__":
    main()
