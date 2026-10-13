"""Greedy completions with top-5 logprobs for four fixed prompts (96 tokens, thinking off).

  python3 golden_sglang.py KEY_FILE [--url URL] [--model NAME] [--out FILE] [--force]

Recorded from the SGLang kit into golden_sglang.json on 2026-10-11 with the defaults below. --url,
--model and --out send a later recording to another server or model name and into a new file; an
existing --out file is kept unless --force is given. Each entry is {prompt, content, usage, logprobs,
finish_reason, served_model}; the 2026-10-11 recording predates the last two.
"""
import argparse
import json
import os
import sys
import urllib.request

prompts = {
 "math": "What is 17 * 23? Answer with just the number.",
 "prose": "Write a detailed, vivid short story (about 600 words) about a lighthouse keeper who discovers that the light is being used to send messages to ships that do not exist.",
 "code": "Write a complete, well-commented Python module implementing a thread-safe LRU cache with TTL expiry, including a small CLI demo and unit tests using unittest.",
 "fact": "Explain in three sentences why the sky is blue.",
}

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("key_file")
ap.add_argument("--url", default="http://127.0.0.1:8888/v1")
ap.add_argument("--model", default="deepseek-v4.1-flash-uncensored")
ap.add_argument("--out", default="golden_sglang.json")
ap.add_argument("--force", action="store_true", help="overwrite an existing --out file")
a = ap.parse_args()
if os.path.exists(a.out) and not a.force:
    sys.exit(f"{a.out} exists; give another --out (or --force to overwrite it)")
key = open(os.path.expanduser(a.key_file)).read().strip()

out = {}
for name, p in prompts.items():
    body = dict(model=a.model, messages=[{"role":"user","content":p}], max_tokens=96,
                temperature=0, logprobs=True, top_logprobs=5, chat_template_kwargs={"thinking": False})
    req = urllib.request.Request(a.url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+key})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    ch = r["choices"][0]
    out[name] = dict(prompt=p, content=ch["message"]["content"], usage=r["usage"], logprobs=ch.get("logprobs"),
                     finish_reason=ch.get("finish_reason"), served_model=r.get("model"))
    print(name, r["usage"], ch.get("finish_reason"), repr(ch["message"]["content"][:120]))
json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out)
