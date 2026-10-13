#!/usr/bin/env python3
"""OpenAI-compatible benchmark for DeepSeek-V4.1-Flash engines (stdlib only).

Modes:
  decode   : single-stream greedy decode rate on fixed prompts (prose/code/agentic)
  prefill  : cold prefill (unique prefix) TTFT and tok/s at several prompt lengths
  agentic  : multi-turn session growing by tool outputs; TTFT per turn (prefix-cache reuse)
  edit     : pi-style coding tasks whose read/edit/write calls really apply to scratch files (editbench.py)
Usage: bench.py --url http://127.0.0.1:8888/v1 --key-file F --model M [--modes decode,prefill,agentic]
"""
import argparse, json, os, random, sys, time, urllib.request, glob

def post_stream(url, key, body, timeout=3600):
    body = dict(body); body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})})
    t0 = time.perf_counter(); t_first = None; usage = None; text = []; reasoning = []; n_chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            ev = json.loads(payload)
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                c = (d.get("content") or "") + (d.get("reasoning_content") or "")
                tc = d.get("tool_calls")
                if c or tc:
                    if t_first is None: t_first = time.perf_counter()
                    n_chunks += 1
                    text.append(d.get("content") or ""); reasoning.append(d.get("reasoning_content") or "")
    t1 = time.perf_counter()
    return dict(ttft=(t_first or t1) - t0, total=t1 - t0, t_first=t_first, t0=t0, t1=t1, usage=usage,
                text="".join(text), reasoning="".join(reasoning), chunks=n_chunks)

DECODE_PROMPTS = {
  "prose": "Write a detailed, vivid short story (about 600 words) about a lighthouse keeper who discovers that the light is being used to send messages to ships that do not exist.",
  "code": "Write a complete, well-commented Python module implementing a thread-safe LRU cache with TTL expiry, including a small CLI demo and unit tests using unittest.",
  "agentic": "You are refactoring a large codebase. Explain step by step, with concrete shell commands and code diffs, how you would migrate a Flask application with SQLAlchemy models to FastAPI with async SQLAlchemy, preserving all endpoints and tests.",
  "json": "Produce a JSON array of 25 fictional employees. Each object must have id, name, email, department, title, salary, start_date (ISO), skills (array of 3 strings) and manager_id (or null).",
}

def corpus_text():
    files = sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
    random.Random(1234).shuffle(files)
    for f in files:
        try:
            yield open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            pass

def make_doc(n_chars, seed):
    out = []; tot = 0
    rng = random.Random(seed)
    files = sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
    rng.shuffle(files)
    for f in files:
        try: t = open(f, encoding="utf-8", errors="ignore").read()
        except Exception: continue
        out.append(f"# FILE: {f}\n{t}\n"); tot += len(t)
        if tot >= n_chars: break
    return "".join(out)[:n_chars]

def run_decode(a, base):
    res = {}
    for name, p in DECODE_PROMPTS.items():
        rates = []
        for rep in range(a.reps):
            body = dict(model=a.model, messages=[{"role": "user", "content": p}], max_tokens=a.max_tokens,
                        temperature=0, chat_template_kwargs={"thinking": False})
            r = post_stream(base, a.key, body)
            ct = r["usage"]["completion_tokens"] if r["usage"] else None
            dec_t = r["t1"] - (r["t_first"] or r["t1"])
            rate = (ct - 1) / dec_t if ct and dec_t > 0 else float("nan")
            rates.append(rate)
            print(f"decode {name:8s} rep{rep}: completion={ct} ttft={r['ttft']*1000:.0f}ms decode={rate:.1f} tok/s  e2e={ct/r['total']:.1f} tok/s", flush=True)
        res[name] = rates
    return res

def run_prefill(a, base):
    res = {}
    for n_k in a.prefill_lens:
        chars = int(n_k * 1000 * 3.6)  # ~3.6 chars/token for python source
        doc = make_doc(chars, seed=n_k)
        nonce = f"Session {random.random():.12f} {time.time()}\n"
        msg = nonce + "Here is a source dump:\n" + doc + "\n\nIn one sentence, what is the first file about?"
        body = dict(model=a.model, messages=[{"role": "user", "content": msg}], max_tokens=1, temperature=0,
                    chat_template_kwargs={"thinking": False})
        r = post_stream(base, a.key, body)
        pt = r["usage"]["prompt_tokens"] if r["usage"] else None
        cached = (r["usage"] or {}).get("prompt_tokens_details") or {}
        print(f"prefill ~{n_k}k: prompt_tokens={pt} cached={cached.get('cached_tokens')} total={r['total']:.2f}s -> {pt/r['total']:.0f} tok/s", flush=True)
        res[n_k] = (pt, r["total"])
    return res

def run_agentic(a, base):
    """Simulate an agent: big system prompt + repo context, then N turns each adding a tool result."""
    sysmsg = "You are a coding agent. Use the provided context. Answer briefly.\n" + make_doc(int(a.agent_ctx_k*1000*3.6), seed=99)
    nonce = f"[run {random.random():.10f}]"
    msgs = [{"role": "system", "content": nonce + sysmsg}, {"role": "user", "content": "List the first three function names you see."}]
    out = []
    for turn in range(a.turns):
        body = dict(model=a.model, messages=msgs, max_tokens=a.agent_out, temperature=0, chat_template_kwargs={"thinking": False})
        r = post_stream(base, a.key, body)
        u = r["usage"] or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        ct = u.get("completion_tokens")
        dec_t = r["t1"] - (r["t_first"] or r["t1"])
        print(f"agentic turn {turn}: prompt={u.get('prompt_tokens')} cached={cached} ttft={r['ttft']*1000:.0f}ms completion={ct} decode={(ct-1)/dec_t if ct and dec_t>0 else 0:.1f} tok/s", flush=True)
        out.append((u.get("prompt_tokens"), cached, r["ttft"]))
        msgs.append({"role": "assistant", "content": r["text"]})
        tool = make_doc(int(a.tool_k * 1000 * 3.6), seed=1000 + turn)
        msgs.append({"role": "user", "content": f"Tool output (cat file):\n{tool}\nWhat changed? One line."})
    return out

def run_edit(a, base):
    import editbench
    return editbench.run(a, base)

def main():
    import editbench
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--key-file", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--modes", default="decode,prefill,agentic")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prefill-lens", default="1,4,16,32,64")
    ap.add_argument("--agent-ctx-k", type=float, default=16)
    ap.add_argument("--tool-k", type=float, default=2)
    ap.add_argument("--agent-out", type=int, default=64)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--out", default="")
    editbench.add_args(ap)
    a = ap.parse_args()
    a.key = open(a.key_file).read().strip() if a.key_file else ""
    a.prefill_lens = [float(x) if "." in x else int(x) for x in a.prefill_lens.split(",")]
    results = {}
    for m in a.modes.split(","):
        results[m] = {"decode": run_decode, "prefill": run_prefill, "agentic": run_agentic, "edit": run_edit}[m](a, a.url)
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1, default=str)

if __name__ == "__main__":
    main()
