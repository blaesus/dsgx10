#!/usr/bin/env python3
"""G8 (docs/engine.md, "End to end on real weights"): real text through glmfront. The four golden conversations of
ref/tokens (short, prose, python, and agentic's first six messages, ending in a tool result, so the prompt stays
under 8k tokens), each greedy and then seeded (temperature 1.0, top-p 0.95, seed 1234), up to MAX new tokens, one
request at a time; then two sessions at once (python with prose, short with agentic, greedy). Per request: the
time to the first streamed token (TTFT), the decode rate after it, the text; the ids come from glmfront's
--dump-dir (N.out.json), matched by order. Writes OUT (JSON); compare two configurations' files with --compare.

  g8_front.py run URL KEY_FILE REF_TOKENS_DIR DUMP_DIR OUT.json [--max 512] [--phases single,pair]
  g8_front.py compare A.json B.json ...      the same ids request by request (the G7 rule on real text)
"""
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse

PROMPTS = ["short", "prose", "python", "agentic"]
SAMPLINGS = [("greedy", {"temperature": 0}), ("seeded", {"temperature": 1.0, "top_p": 0.95, "seed": 1234})]
PAIRS = [("python", "prose"), ("short", "agentic")]


def conversation(ref, name):
    d = json.load(open(os.path.join(ref, name + ".messages.json")))
    msgs = d["messages"]
    if name == "agentic":
        msgs = msgs[:6]
    body = {"messages": msgs}
    if d.get("tools"):
        body["tools"] = d["tools"]
    return body


def post(url, key, body):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body=json.dumps(dict(body, stream=True, stream_options={"include_usage": True})),
              headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    r = c.getresponse()
    out = {"status": r.status, "reasoning": "", "content": "", "tool_calls": [], "usage": None, "events": 0,
           "t_first": None, "finish": None}
    while True:
        line = r.fp.readline()
        if not line:
            break
        line = line.decode().strip()
        if not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            break
        e = json.loads(line[6:])
        out["events"] += 1
        if e.get("usage"):
            out["usage"] = e["usage"]
        for ch in e.get("choices", []):
            d = ch.get("delta") or {}
            piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
            if (piece or d.get("tool_calls")) and out["t_first"] is None:
                out["t_first"] = time.time() - t0
            out["reasoning"] += d.get("reasoning_content") or ""
            out["content"] += d.get("content") or ""
            if d.get("tool_calls"):
                out["tool_calls"].append(d["tool_calls"])
            if ch.get("finish_reason"):
                out["finish"] = ch["finish_reason"]
    out["t_total"] = time.time() - t0
    c.close()
    u_ = out["usage"] or {}
    n = u_.get("completion_tokens", 0)
    dt = out["t_total"] - (out["t_first"] or 0)
    out["decode_tps"] = (n - 1) / dt if n > 1 and dt > 0 else 0.0
    return out


def newest_dump(dump, seen):
    """the out.json files not seen before, in order of their number"""
    fs = sorted((f for f in os.listdir(dump) if f.endswith(".out.json") and f not in seen),
                key=lambda f: int(f.split(".")[0]) if f.split(".")[0].isdigit() else 0)
    return fs


def run(url, keyf, ref, dump, out, mx, phases):
    key = open(keyf).read().strip()
    res = {"requests": []}
    seen = set(f for f in os.listdir(dump)) if os.path.isdir(dump) else set()

    def ids_of_new():
        time.sleep(0.3)
        got = []
        for f in newest_dump(dump, seen):
            seen.add(f)
            try:
                d = json.load(open(os.path.join(dump, f)))
                rq = os.path.join(dump, f.replace(".out.json", ".json"))
                if os.path.exists(rq):
                    d["request"] = json.load(open(rq)).get("request")
                got.append(d)
            except Exception as e:  # noqa
                got.append({"error": str(e)})
        return got

    if "single" in phases:
        for p in PROMPTS:
            base = conversation(ref, p)
            for sname, sp in SAMPLINGS:
                o = post(url, key, dict(base, max_tokens=mx, **sp))
                dumps = ids_of_new()
                o.update(prompt=p, sampling=sname, sessions=1, dumps=dumps)
                u = o["usage"] or {}
                print("%-8s %-7s status %d prompt %5d cached %5d completion %4d  TTFT %6.3f s  decode %5.1f tok/s  %s" % (
                    p, sname, o["status"], u.get("prompt_tokens", 0),
                    (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0), u.get("completion_tokens", 0),
                    o["t_first"] or -1, o["decode_tps"], o["finish"]), flush=True)
                res["requests"].append(o)
    if "pair" in phases:
        for a, b in PAIRS:
            outs = {}

            def go(name):
                outs[name] = post(url, key, dict(conversation(ref, name), max_tokens=mx, temperature=0))

            t0 = time.time()
            th = [threading.Thread(target=go, args=(n,)) for n in (a, b)]
            for t in th:
                t.start()
            for t in th:
                t.join()
            wall = time.time() - t0
            dumps = ids_of_new()
            tot = 0
            for n in (a, b):
                o = outs[n]
                u = o["usage"] or {}
                tot += u.get("completion_tokens", 0)
                o.update(prompt=n, sampling="greedy", sessions=2, pair=a + "+" + b,
                         dumps=[d for d in dumps if d.get("request", {}).get("messages") == conversation(ref, n)["messages"]])
                print("pair %-16s %-8s completion %4d  TTFT %6.3f s  decode %5.1f tok/s  %s" % (
                    a + "+" + b, n, u.get("completion_tokens", 0), o["t_first"] or -1, o["decode_tps"], o["finish"]), flush=True)
                res["requests"].append(o)
            res.setdefault("pairs", []).append({"pair": a + "+" + b, "wall_s": wall, "tokens": tot, "tps": tot / wall})
            print("pair %-16s together %d tokens in %.1f s: %.1f tok/s" % (a + "+" + b, tot, wall, tot / wall), flush=True)
    json.dump(res, open(out, "w"), indent=1)


def out_ids(o):
    for d in o.get("dumps", []):
        if isinstance(d.get("ids"), list):
            return d["ids"][:d.get("kept", len(d["ids"]))]
    return None


def compare(files):
    runs = [json.load(open(f)) for f in files]
    base = {(o["prompt"], o["sampling"], o["sessions"]): o for o in runs[0]["requests"]}
    bad = 0
    for f, r in zip(files[1:], runs[1:]):
        for o in r["requests"]:
            k = (o["prompt"], o["sampling"], o["sessions"])
            if k not in base:
                continue
            a, b = out_ids(base[k]), out_ids(o)
            ta = base[k]["reasoning"] + "|" + base[k]["content"]
            tb = o["reasoning"] + "|" + o["content"]
            same = (a == b) if (a is not None and b is not None) else (ta == tb)
            bad += not same
            print("%-40s %-8s %-7s %d: %s (%s)" % (os.path.basename(f), *k, "same" if same else "DIFFER",
                                                   "ids" if a is not None and b is not None else "text"))
    print("compare: %s" % ("all the same" if not bad else "%d differ" % bad))
    return bad


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 6 and a[0] == "run":
        opt = {"--max": "512", "--phases": "single,pair"}
        for i in range(6, len(a), 2):
            opt[a[i]] = a[i + 1]
        run(a[1], a[2], a[3], a[4], a[5], int(opt["--max"]), opt["--phases"].split(","))
    elif len(a) >= 3 and a[0] == "compare":
        sys.exit(1 if compare(a[1:]) else 0)
    else:
        sys.exit(__doc__)
