#!/usr/bin/env python3
"""Tokens per step and time per step of bench_record.py's decode requests, from each server's own log.

  python3 steps.py kit  DSV41_LOG K1.jsonl [K2.jsonl ...]   [--utc-offset-h H]
  python3 steps.py ours RUN4_LOG O1.jsonl [RUN4_LOG O2.jsonl ...]

kit   First the conditions while the recorded requests ran: the requests in the kit's access lines by
      client address and path, and the most running and queued requests on any batch line.
      Then the decode-batch lines, `[YYYY-MM-DD HH:MM:SS TP0] Decode batch, #running-req: R, ...
      accept len: A, ... gen throughput (token/s): G, ... #queue-req: Q`, inside each decode request's
      decode interval, t_wall + [t_first, t1] (the kit's container logs UTC, to the second; --utc-offset-h
      if a log is in another zone): a line stamped second s counts only if [s, s + 1) is inside. Each
      interval's first line is dropped (its throughput spans time before the request), and every line
      kept must show R = 1 and Q = 0. A line covers a fixed number of steps, so tokens per step is the
      mean of A over the lines, and a step 1000 A / G ms (median). Only these numbers are read from the
      log, and no line of it is printed. The lines sample the text (a few per request), so tokens per
      step / step time agrees with bench.py's rate only to a few per cent.
ours  Rank 0's `serve: req N drafts: S verify steps, ... A accepted per step` and `serve: req N session
      ...: prompt P, ... T tokens in X ms` lines, one RUN4_LOG per session, each followed by that
      session's records. A request is matched to its prompt type by P against the records'
      usage.prompt_tokens. Tokens per step is 1 + A, a step X / S ms.
Per prompt type: tokens per step, ms per step (kit: median over its log lines; ours: mean over
requests), their quotient in tok/s, and bench.py's own rate for the same requests.
"""
import argparse
import datetime as dt
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import DECODE_PROMPTS  # noqa: E402

TYPES = list(DECODE_PROMPTS)


def decode_records(path):
    """[(type, record)] of a bench_record.py file's decode requests, in order."""
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        msgs = (r.get("body") or {}).get("messages") or []
        if (r.get("body") or {}).get("max_tokens") == 1 or not msgs or msgs[0].get("role") == "system":
            continue
        t = next((k for k, p in DECODE_PROMPTS.items() if msgs[0].get("content") == p), None)
        if t:
            out.append((t, r))
    return out


def rate(r):
    res = r["result"]
    ct = (res.get("usage") or {}).get("completion_tokens")
    if not ct or res.get("t_first") is None or res["t1"] <= res["t_first"]:
        return float("nan")
    return (ct - 1) / (res["t1"] - res["t_first"])


KIT_LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: TP0)?\] (Decode batch|Prefill batch|INFO: )[,.]?\s*(.*)$")
ACCESS = re.compile(r"^([\d.]+|\[[\da-f:]+\]):\d+ - \"([A-Z]+) (\S+) HTTP/[\d.]+\" (\d+)")


def num(pat, s):
    m = re.search(pat, s)
    return float(m.group(1)) if m else None


def kit_lines(path, offset_h):
    """From the kit's log, numbers only: decode-batch lines as (epoch s, running, queue, accept len,
    gen tok/s), prefill-batch lines as (epoch s, running, queue), access lines as (epoch s, client
    address, method, path, status)."""
    tz = dt.timezone(dt.timedelta(hours=offset_h))
    dec, pre, acc = [], [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = KIT_LINE.match(line)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz).timestamp()
        kind, rest = m.group(2), m.group(3)
        if kind == "Decode batch":
            dec.append((t, num(r"#running-req: (\d+)", rest), num(r"#queue-req: (\d+)", rest),
                        num(r"accept len: ([\d.]+)", rest), num(r"gen throughput \(token/s\): ([\d.]+)", rest)))
        elif kind == "Prefill batch":
            pre.append((t, num(r"#running-req: (\d+)", rest), num(r"#queue-req: (\d+)", rest)))
        else:
            a = ACCESS.match(rest)
            if a:
                acc.append((t, a.group(1), a.group(2), a.group(3).split("?")[0][:40], a.group(4)))
    return dec, pre, acc


def conditions(dec, pre, acc, records):
    """Who else used the kit while the sessions ran: requests by client and path, and the most
    running and queued requests any prefill or decode line showed, between the first request's start
    and the last one's end."""
    recs = [json.loads(l) for p in records for l in open(p, encoding="utf-8") if l.strip()]
    lo = min(r["t_wall"] for r in recs) - 1.0
    hi = max(r["t_wall"] + r["result"]["t1"] for r in recs) + 1.0
    by = {}
    for t, ip, meth, path, status in acc:
        if lo <= t <= hi:
            by[(ip, meth, path, status)] = by.get((ip, meth, path, status), 0) + 1
    run = [x[1] for x in dec + pre if lo <= x[0] <= hi and x[1] is not None]
    que = [x[2] for x in dec + pre if lo <= x[0] <= hi and x[2] is not None]
    span = dt.datetime.fromtimestamp(lo + 1.0, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"## kit: conditions from {span} UTC for {hi - lo - 2.0:.0f} s ({len(recs)} recorded requests)\n")
    for k, v in sorted(by.items()):
        print(f"- {v} x {k[1]} {k[2]} {k[3]} from {k[0]}")
    print(f"- most running requests on a log line: {int(max(run)) if run else '-'}; "
          f"most queued: {int(max(que)) if que else '-'}\n")


def kit(a):
    rows, pre, acc = kit_lines(a.log, a.utc_offset_h)
    conditions(rows, pre, acc, a.records)
    by = {t: {"al": [], "ms": [], "rate": [], "lines": []} for t in TYPES}
    for path in a.records:
        for t, r in decode_records(path):
            res = r["result"]
            lo, hi = r["t_wall"] + (res["t_first"] or 0.0), r["t_wall"] + res["t1"]
            # a line stamped second s was written in [s, s + 1): keep it only if all of that is inside
            win = [x for x in rows if lo <= x[0] and x[0] + 1.0 <= hi][1:]
            ok = [x for x in win if x[1] == 1 and x[2] == 0 and x[3] and x[4]]
            by[t]["lines"].append((len(win), len(ok)))
            by[t]["al"] += [x[3] for x in ok]
            by[t]["ms"] += [1000.0 * x[3] / x[4] for x in ok]
            by[t]["rate"].append(rate(r))
    report("kit", by, "median of the log lines")


RUN4_DRAFTS = re.compile(r"^serve: req (\d+) drafts: (\d+) verify steps, [\d.]+ drafts verified and ([\d.]+) accepted per step")
RUN4_DONE = re.compile(r"^serve: req (\d+) session \S+: prompt (\d+), .* (\d+) tokens in ([\d.]+) ms")


def ours(a):
    args = [a.log] + a.records
    sessions, cur = [], None
    for p in args:
        if p.endswith(".jsonl"):
            if cur is None:
                sys.exit("give each session's RUN4_LOG before its records")
            cur[1].append(p)
        else:
            cur = (p, [])
            sessions.append(cur)
    by = {t: {"al": [], "ms": [], "rate": [], "lines": []} for t in TYPES}
    for log, recs in sessions:
        drafts, done = {}, {}
        for line in open(log, encoding="utf-8", errors="replace"):
            m = RUN4_DRAFTS.match(line)
            if m:
                drafts[int(m.group(1))] = (int(m.group(2)), float(m.group(3)))
            m = RUN4_DONE.match(line)
            if m:
                done[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
        dec = [(t, r) for p in recs for t, r in decode_records(p)]
        ptype = {}
        for t, r in dec:
            pt = (r["result"].get("usage") or {}).get("prompt_tokens")
            if pt is not None:
                ptype.setdefault(pt, set()).add(t)
        for n in sorted(done):
            pt, ntok, ms = done[n]
            ts = ptype.get(pt, set())
            if len(ts) != 1 or n not in drafts or ntok < 2:
                continue
            t = next(iter(ts))
            steps, acc = drafts[n]
            by[t]["al"].append(1.0 + acc)
            by[t]["ms"].append(ms / steps)
            by[t]["lines"].append((1, 1))
        for t, r in dec:
            by[t]["rate"].append(rate(r))
    report("ours", by, "mean of the requests")


def report(name, by, how):
    print(f"## {name}: tokens and time per step ({how})\n")
    print("| type | requests | log lines kept / in window | tokens per step | ms per step | tokens / step time | bench.py tok/s |")
    print("|---|---|---|---|---|---|---|")
    for t in TYPES:
        b = by[t]
        kept = f"{sum(k for _, k in b['lines'])} / {sum(w for w, _ in b['lines'])}"
        rates = [x for x in b["rate"] if x == x]
        bench_rate = f"{st.mean(rates):.1f}" if rates else "-"
        if not b["al"]:
            print(f"| {t} | {len(b['rate'])} | {kept} | - | - | - | {bench_rate} |")
            continue
        al = st.mean(b["al"])
        ms = st.median(b["ms"]) if name == "kit" else st.mean(b["ms"])
        print(f"| {t} | {len(b['rate'])} | {kept} | {al:.2f} | {ms:.1f} | {1000.0 * al / ms:.1f} | {bench_rate} |")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("side", choices=["kit", "ours"])
    ap.add_argument("log")
    ap.add_argument("records", nargs="+")
    ap.add_argument("--utc-offset-h", type=float, default=0.0)
    a = ap.parse_args()
    kit(a) if a.side == "kit" else ours(a)


if __name__ == "__main__":
    main()
