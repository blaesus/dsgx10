#!/usr/bin/env python3
"""Statistics of bench_record.py sessions, for one server or two side by side.

  python3 bench_record_report.py --side kit k1.jsonl k2.jsonl k3.jsonl --side ours o1.jsonl o2.jsonl o3.jsonl \
      [--dsfront ~/dsgb10/prodbin/dsfront --tokenizer /data/models/DeepSeek-V4.1-Flash/tokenizer.json]

Each file is one session of `bench.py --modes decode,prefill,agentic` at its defaults. Per side:
  decode   tok/s as bench.py computes it, (completion_tokens - 1) / (t_end - t_first): each session's mean of
           its requests per prompt type, the mean of the session means, the range over all requests and the
           spread (largest - smallest) / mean; tokens in the first content chunk and content chunks a request
  TTFT     the decode requests' time to the first content: median and range, in ms
  prefill  prompt_tokens / total per length: mean and range over sessions
  agentic  turn 0's TTFT; turns 1-5's TTFT (median and range) and the new tokens each turn (prompt - cached)
  tokens   with --dsfront: every text re-tokenised (`dsfront tokenize`) against usage.completion_tokens (a
           stop's end-of-sentence id counts in completion_tokens on both servers, so that one is added back)
  texts    whether each prompt type's text is the same in every session, and, with two sides, where the two
           servers' greedy texts first differ (in characters, and in tokens with --dsfront)
With two sides, the decode ratio (second / first) with its range from the extreme requests.
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import DECODE_PROMPTS  # noqa: E402

TYPES = list(DECODE_PROMPTS)


def load(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    s = {"decode": {t: [] for t in TYPES}, "prefill": [], "agentic": []}
    for r in recs:
        b = r.get("body") or {}
        msgs = b.get("messages") or []
        res = r["result"]
        if b.get("max_tokens") == 1:
            s["prefill"].append(r)
        elif msgs and msgs[0].get("role") == "system":
            s["agentic"].append(r)
        else:
            t = next((k for k, p in DECODE_PROMPTS.items() if msgs and msgs[0].get("content") == p), None)
            if t:
                s["decode"][t].append(r)
        res["rate"] = rate(res)
    return s


def rate(res):
    u = res.get("usage") or {}
    ct = u.get("completion_tokens")
    if not ct or res.get("t_first") is None:
        return float("nan")
    dt = res["t1"] - res["t_first"]
    return (ct - 1) / dt if dt > 0 else float("nan")


def content_chunks(r):
    """(arrival s since open, text) of each data event that carried content, reasoning or tool calls."""
    out = []
    for t, line in r["lines"]:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        p = line[5:].strip()
        if p == "[DONE]":
            break
        try:
            ev = json.loads(p)
        except ValueError:
            continue
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            c = (d.get("content") or "") + (d.get("reasoning_content") or "")
            if c or d.get("tool_calls"):
                out.append((t, c))
    return out


def comments(r):
    return sum(1 for _, l in r["lines"] if l.startswith(":"))


class Tok:
    def __init__(self, fe, tokj):
        self.fe, self.tokj = fe, tokj
        self.dir = tempfile.mkdtemp(prefix="brr")

    def count(self, text):
        if not text:
            return 0
        p = os.path.join(self.dir, "t.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        out = subprocess.run([self.fe, "tokenize", p, "--tokenizer", self.tokj], capture_output=True, text=True, check=True)
        return len(json.loads(out.stdout.strip()))


def rng(xs, fmt="%.1f"):
    return (fmt + "-" + fmt) % (min(xs), max(xs)) if xs else "-"


def spread(xs):
    return 100.0 * (max(xs) - min(xs)) / st.mean(xs) if xs else float("nan")


def side_report(name, sessions, tok):
    print(f"\n## {name}: {len(sessions)} session(s)\n")
    print("| decode | session means (tok/s) | mean | requests | spread | first chunk tokens | content chunks a request |")
    print("|---|---|---|---|---|---|---|")
    summary = {}
    for t in TYPES:
        sm = [st.mean(r["result"]["rate"] for r in s["decode"][t]) for s in sessions if s["decode"][t]]
        rr = [r["result"]["rate"] for s in sessions for r in s["decode"][t]]
        first = []
        nchunks = []
        for s in sessions:
            for r in s["decode"][t]:
                cc = content_chunks(r)
                nchunks.append(len(cc))
                if tok and cc:
                    first.append(tok.count(cc[0][1]))
        summary[t] = (st.mean(sm) if sm else float("nan"), rr)
        print(f"| {t} | {' / '.join('%.2f' % x for x in sm)} | {st.mean(sm):.2f} | {rng(rr, '%.2f')} | {spread(rr):.1f} % | "
              f"{rng(first, '%d') if first else '-'} | {rng(nchunks, '%d')} |")
    tt = [1e3 * r["result"]["ttft"] for s in sessions for t in TYPES for r in s["decode"][t]]
    if tt:
        print(f"\nTTFT, the decode requests ({len(tt)}): median {st.median(tt):.0f} ms, {rng(tt, '%.0f')} ms")
        first = [s["decode"][TYPES[0]][0] for s in sessions if s["decode"][TYPES[0]]]
        rest = [1e3 * r["result"]["ttft"] for s in sessions for t in TYPES for r in s["decode"][t]
                if not any(r is f for f in first)]
        if first and rest:
            print(f"  each session's first request: {' / '.join('%.0f' % (1e3 * r['result']['ttft']) for r in first)} ms; "
                  f"the others ({len(rest)}): median {st.median(rest):.0f} ms, {rng(rest, '%.0f')} ms")
    byn = {}
    for s in sessions:
        for r in s["prefill"]:
            u = r["result"]["usage"] or {}
            pt, cached = u.get("prompt_tokens"), ((u.get("prompt_tokens_details") or {}).get("cached_tokens"))
            byn.setdefault(round(pt / 1000.0) if pt else 0, []).append((pt, pt / r["result"]["total"], cached))
    if byn:
        print("\n| prefill, cold | prompt tokens | tok/s by session | mean | spread | cached |")
        print("|---|---|---|---|---|---|")
        for k in sorted(byn):
            v = byn[k]
            xs = [x[1] for x in v]
            print(f"| ~{k}k | {rng([x[0] for x in v], '%d')} | {' / '.join('%.0f' % x for x in xs)} | {st.mean(xs):.0f} | "
                  f"{spread(xs):.1f} % | {sorted(set(x[2] for x in v), key=str)} |")
    a0 = [r["result"]["ttft"] for s in sessions for r in s["agentic"][:1]]
    an = [r["result"]["ttft"] for s in sessions for r in s["agentic"][1:]]
    new = []
    for s in sessions:
        for r in s["agentic"][1:]:
            u = r["result"]["usage"] or {}
            c = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
            if u.get("prompt_tokens") is not None and c is not None:
                new.append(u["prompt_tokens"] - c)
    if a0:
        print(f"\nagentic: turn 0 TTFT {' / '.join('%.3f' % x for x in a0)} s; turns 1+ ({len(an)}) median {st.median(an):.3f} s, "
              f"{rng(an, '%.3f')} s; new tokens a turn {rng(new, '%d') if new else '-'}")
    if tok:
        n = eq = 0
        worst = []
        for s in sessions:
            recs = [r for t in TYPES for r in s["decode"][t]] + s["agentic"]
            for r in recs:
                u = r["result"]["usage"] or {}
                ct = u.get("completion_tokens")
                if ct is None:
                    continue
                stop = any('"finish_reason": "stop"' in l or '"finish_reason":"stop"' in l for _, l in r["lines"])
                want = ct - 1 if stop else ct
                got = tok.count(r["result"]["text"] + r["result"]["reasoning"])
                n += 1
                eq += got == want
                if got != want:
                    worst.append((want, got))
        print(f"\ntoken accounting: {eq} of {n} texts re-tokenise to completion_tokens (less a stop's end id); "
              f"differences (expected, re-tokenised): {worst[:8]}")
    same = []
    for t in TYPES:
        texts = [r["result"]["text"] for s in sessions for r in s["decode"][t]]
        same.append(f"{t} {'same' if texts and all(x == texts[0] for x in texts) else 'DIFFERS'} ({len(texts)})")
    print("\ngreedy texts across this side's requests: " + ", ".join(same))
    cm = sum(comments(r) for s in sessions for t in TYPES for r in s["decode"][t])
    print(f"SSE comment lines in the decode requests: {cm}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", nargs="+", action="append", required=True, metavar=("NAME", "FILE"))
    ap.add_argument("--dsfront", default="")
    ap.add_argument("--tokenizer", default="")
    a = ap.parse_args()
    tok = Tok(os.path.expanduser(a.dsfront), a.tokenizer) if a.dsfront and a.tokenizer else None
    sides = [(s[0], [load(p) for p in s[1:]]) for s in a.side]
    sums = [(n, side_report(n, ss, tok)) for n, ss in sides]
    if len(sides) == 2:
        (n1, s1), (n2, s2) = sums
        print(f"\n## {n2} / {n1}\n")
        for t in TYPES:
            m1, r1 = s1[t]
            m2, r2 = s2[t]
            print(f"- {t}: {m2 / m1:.3f} (from {min(r2) / max(r1):.3f} to {max(r2) / min(r1):.3f})")
        print()
        for t in TYPES:
            a1 = next((r["result"]["text"] for s in sides[0][1] for r in s["decode"][t]), "")
            a2 = next((r["result"]["text"] for s in sides[1][1] for r in s["decode"][t]), "")
            i = next((k for k, (x, y) in enumerate(zip(a1, a2)) if x != y), min(len(a1), len(a2)))
            where = "identical" if a1 == a2 else f"first differ at character {i}"
            if tok and a1 != a2:
                where += f" (after {tok.count(a1[:i])} tokens of {n1}'s text)"
            print(f"- {t}: the two servers' first texts are {where}")


if __name__ == "__main__":
    main()
