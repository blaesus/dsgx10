#!/usr/bin/env python3
"""A soak of the service through the front-end (docs/engine.md, "Soak"): agentic sessions replayed and
grown for a fixed time, with what real use does to a server: forks, regenerates, cancels mid-stream,
client disconnects (during a prefill, and with a reset mid-stream), more sessions than the session store
holds (so it evicts), a few sessions grown to ~150k tokens, compaction-like turns (the history replaced,
the system prompt kept), the golden agentic transcript replayed turn by turn, streamed and plain
requests, greedy and sampled, thinking on and off. Several clients at once, so requests queue.

Records in --out: requests.jsonl (each request: kind, session, times, usage, finish, checks, error),
server.jsonl (rank 0's and the front-end's per-request lines and the store's evictions, timestamped as
they are read), memory.jsonl (each rank's MemAvailable, resident set and GPU memory every --sample-s).
The summary (per --window-min window, then the pass criteria) is printed at the end and by --report.

  python3 engine/server/tests/soak.py --url http://127.0.0.1:18893 --key-file ~/dsgb10-data/dev-api-key \\
      --minutes 60 --out DIR [--clients 3] [--seed 1] [--logs ~/dsgb10/service/logs]
  python3 engine/server/tests/soak.py --report DIR

Pass: no errors (every response well formed: status 200, every event JSON, one finish reason, [DONE],
usage, tool calls with known names and JSON arguments); memory flat (each rank's GPU memory and
resident set within 256 MiB of where they started, MemAvailable within 1 GiB); latency flat (decode
tok/s and time to first token of comparable requests in the last window within 10 % of the first); the
store evicted; and, after it, a clean stop (service.sh stop; checked outside this script).

--disk loads the store's disk tier (docs/engine.md, "The disk tier"): more and longer sessions, so that
the store and the file fill; branches of long sessions that are likely spilled (fork_long); revisits of
the least recently used sessions the front-end still holds, some right after a branch has pushed them
out; and every few minutes a revisit against the same prompt cold (the system prompt changed at its
start). Every session put aside gets a reference: its last prompt twice, greedy and short (a probe),
the second restarting from the live ring at the prompt's last position. A revisit is that probe again,
and must answer the same wherever rank 0 restarted it at the same position (from the store or the
disk). Also recorded: rank 0's spills, flushes (the copy into staging timed on its own), read-backs
(each block checked against its checksum) and drops; each rank's pinned memory, the page cache the
spill file holds (cachestat; O_DIRECT should keep it at none) and the box's dirty pages.
Pass, with --disk records: the above, every comparable check equal (at least one from the disk),
read-backs, and the spill file's page cache under 64 MiB.
"""
import base64
import argparse
import collections
import glob
import hashlib
import http.client
import json
import math
import os
import random
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
BOXES = [("rank 0", None), ("rank 1", "10.0.0.4"), ("rank 2", "10.0.0.3"), ("rank 3", "10.0.0.2")]  # this fleet's workers
CHARS_PER_TOKEN = 3.4  # code and prose in the corpus below, by the DeepSeek tokenizer (roughly)
FE_TABLE = (256, 16 << 20)  # the front-end's session table: sessions and token ids (frontend/src/server.rs)
GOLDEN = json.load(open(os.path.join(REPO, "golden", "long.messages.json")))
TOOLS = GOLDEN["tools"]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
# the golden transcript's system prompt with a longer workspace preamble, as agent harnesses have
SYSTEM = GOLDEN["messages"][0]["content"] + "\n\n# Workspace notes\n\n" + open(os.path.join(REPO, "README.md")).read()[:7000]
TASKS = [
    "Find why {f} is slow on large inputs and propose a fix with a benchmark.",
    "Review {f} for error handling gaps and write tests for the ones you find.",
    "Port the parts of {f} that we use to a small module of our own, with the same behaviour.",
    "Explain how {f} works to a new team member, with the call graph of its main entry points.",
    "Add type hints to {f} and check them with the tools in the workspace.",
]


def load_corpus():
    files = sorted(glob.glob("/usr/lib/python3.12/*.py")) + sorted(glob.glob(os.path.join(REPO, "docs", "*.md")))
    files += sorted(glob.glob(os.path.join(REPO, "engine", "*", "*.cu")))
    out = []
    for f in files:
        try:
            if os.path.getsize(f) > 4000:
                out.append((f, open(f, errors="replace").read()))
        except OSError:
            pass
    return out


CORPUS = load_corpus()


def tool_text(rng, tokens):
    """About `tokens` tokens of real files, as a tool would return them."""
    want = int(tokens * CHARS_PER_TOKEN)
    parts, got = [], 0
    while got < want:
        path, text = rng.choice(CORPUS)
        start = rng.randrange(0, max(1, len(text) - 2000))
        piece = text[start:start + want - got]
        parts.append(f"==> {path} (offset {start}) <==\n{piece}\n")
        got += len(piece) + len(path) + 24
    return "".join(parts)


def tool_size(rng, long=False):
    if long:
        return rng.randint(10000, 15000)
    if rng.random() < 0.5:
        return int(math.exp(rng.uniform(math.log(200), math.log(2000))))
    return rng.randint(2000, 15000)


def est_mib(n):
    """A session's state in the store (MiB) by its length: pages of 4096 positions, window snapshots as
    thinned (every 1024 over the last 8192, four per doubling beyond, the prompt's end), the end snapshot."""
    if n <= 0:
        return 0.0
    snaps = min(n, 8192) / 1024 + (4 * math.log2(n / 8192) if n > 8192 else 0) + 1
    return 3.48 * math.ceil(n / 4096) + 2.81 * snaps + 6.1


class Recorder:
    def __init__(self, path):
        self.f = open(path, "a")
        self.lock = threading.Lock()

    def write(self, rec):
        with self.lock:
            self.f.write(json.dumps(rec) + "\n")
            self.f.flush()


# ------------------------------------------------------------------ HTTP


class Client:
    def __init__(self, url, key):
        u = urllib.parse.urlparse(url)
        self.host, self.port, self.key = u.hostname, u.port, key

    def _send(self, body, timeout=1800):
        c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        c.request("POST", "/v1/chat/completions", body=json.dumps(body),
                  headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        return c

    def call(self, body, cancel_after=None, disconnect_after=None, reset=False):
        """One request. Streamed: read events (to [DONE], or cancel_after content events, then close; with
        reset, close with an RST). disconnect_after: close that many seconds after sending, reading nothing.
        Returns the raw outcome; check() judges it."""
        o = {"t_send": time.time(), "t_first": None, "t_end": None, "status": None, "events": [], "body": None,
             "aborted": None, "error": None}
        c = None
        try:
            c = self._send(body)
            if disconnect_after is not None:
                time.sleep(disconnect_after)
                c.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                c.close()
                o["aborted"] = "disconnect"
                return o
            r = c.getresponse()
            o["status"] = r.status
            if not body.get("stream"):
                raw = r.read()
                o["t_end"] = time.time()
                try:
                    o["body"] = json.loads(raw)
                except ValueError:
                    o["error"] = f"not JSON: {raw[:200]!r}"
                return o
            if r.status != 200:
                o["error"] = f"HTTP {r.status}: {r.read()[:300]!r}"
                return o
            seen = 0
            while True:
                line = r.fp.readline()
                if not line:
                    break
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                d = line[6:]
                if d == "[DONE]":
                    o["events"].append(d)
                    break
                try:
                    e = json.loads(d)
                except ValueError:
                    o["events"].append({"_bad": d[:200]})
                    continue
                o["events"].append(e)
                for ch in e.get("choices") or []:
                    dl = ch.get("delta") or {}
                    if dl.get("content") or dl.get("reasoning_content") or dl.get("tool_calls"):
                        if o["t_first"] is None:
                            o["t_first"] = time.time()
                        seen += 1
                if cancel_after is not None and seen >= cancel_after:
                    if reset:
                        c.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    c.close()
                    o["aborted"] = "reset" if reset else "cancel"
                    break
            o["t_end"] = time.time()
        except Exception as e:  # a connection that fails is an error unless we caused it
            o["error"] = f"{type(e).__name__}: {e}"
            o["t_end"] = time.time()
        finally:
            if c is not None and o["aborted"] is None:
                c.close()
        return o


def collect(events):
    r = {"content": "", "reasoning": "", "calls": {}, "finish": [], "usage": None, "errors": [], "bad": 0}
    for e in events:
        if e == "[DONE]":
            continue
        if "_bad" in e:
            r["bad"] += 1
            continue
        if "error" in e:
            r["errors"].append(e["error"])
            continue
        if e.get("usage"):
            r["usage"] = e["usage"]
        for ch in e.get("choices") or []:
            d = ch.get("delta") or {}
            r["content"] += d.get("content") or ""
            r["reasoning"] += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                c = r["calls"].setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                c["id"] = tc.get("id") or c["id"]
                f = tc.get("function") or {}
                c["name"] += f.get("name") or ""
                c["arguments"] += f.get("arguments") or ""
            if ch.get("finish_reason"):
                r["finish"].append(ch["finish_reason"])
    return r


def check_calls(calls):
    """The front-end's part of a tool call: an id, and arguments that parse as JSON. (A call of a tool the
    request did not offer is the model's choice, not a malformed response: the summary counts them apart.)"""
    bad = []
    for c in calls:
        if not c.get("id"):
            bad.append(f"tool call without an id: {c['name']!r}")
        try:
            json.loads(c["arguments"] or "{}")
        except ValueError:
            bad.append(f"arguments not JSON: {c['arguments'][:80]!r}")
    return bad


def answer(rec):
    """What the model said: the text and the calls, without the calls' ids (random per response)."""
    return [rec.get("content"), [[c["name"], c["arguments"]] for c in rec.get("calls") or []]]


def judge(body, o):
    """What the request returned, and what is wrong with it (empty: well formed)."""
    rec = {"status": o["status"], "aborted": o["aborted"], "t_send": o["t_send"],
           "ttft": (o["t_first"] - o["t_send"]) if o["t_first"] else None,
           "total": (o["t_end"] - o["t_send"]) if o["t_end"] else None}
    bad = []
    if o["aborted"] == "disconnect":
        return rec, bad
    if o["error"]:
        bad.append(o["error"])
        return rec, bad
    if not body.get("stream"):
        b = o["body"] or {}
        if o["status"] != 200:
            bad.append(f"HTTP {o['status']}: {json.dumps(b)[:300]}")
            return rec, bad
        try:
            ch = b["choices"][0]
            msg, finish, usage = ch["message"], ch["finish_reason"], b["usage"]
        except (KeyError, IndexError, TypeError):
            bad.append(f"malformed: {json.dumps(b)[:300]}")
            return rec, bad
        calls = [{"id": c.get("id"), "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                 for c in msg.get("tool_calls") or []]
        rec.update(finish=finish, usage=usage, content=msg.get("content") or "", calls=calls)
        bad += check_calls(calls)
        if finish not in ("stop", "length", "tool_calls"):
            bad.append(f"finish {finish}")
        return rec, bad
    r = collect(o["events"])
    calls = [r["calls"][k] for k in sorted(r["calls"])]
    rec.update(finish=r["finish"][0] if r["finish"] else None, usage=r["usage"], content=r["content"],
               reasoning=len(r["reasoning"]), calls=calls)
    if r["bad"]:
        bad.append(f"{r['bad']} events not JSON")
    if r["errors"]:
        bad.append(f"error events: {r['errors'][:2]}")
    if o["aborted"]:
        return rec, bad
    if not o["events"] or o["events"][-1] != "[DONE]":
        bad.append("no [DONE]")
    if len(r["finish"]) != 1 or r["finish"][0] not in ("stop", "length", "tool_calls"):
        bad.append(f"finish reasons {r['finish']}")
    u = r["usage"]
    if not u or u.get("prompt_tokens", 0) <= 0 or u.get("prompt_tokens_details", {}).get("cached_tokens", 0) > u["prompt_tokens"]:
        bad.append(f"usage {u}")
    bad += check_calls(calls)
    if r["finish"] == ["tool_calls"] and not calls:
        bad.append("finish tool_calls without a call")
    return rec, bad


# ------------------------------------------------------------------ sessions


class Session:
    def __init__(self, name, messages, target, kind):
        self.name, self.messages, self.target, self.kind = name, messages, target, kind
        self.tokens, self.turns, self.pending, self.busy, self.done, self.t_last = 0, 0, 0, False, False, 0.0
        self.last = None  # the messages of its last completed turn (to regenerate)
        self.ref, self.ref_n = None, 0  # --disk: the answer a probe of it must give, and which reference
        self.lock = threading.Lock()


class Pool:
    LONG_EVERY = 90  # seconds between a long session's turns, so that they grow through the run

    def __init__(self, rng, n_active, n_long, long_target=150000, keep=64):
        self.lock = threading.RLock()
        self.rng = rng
        self.n = 0
        self.long_target, self.keep = long_target, keep
        self.active, self.retired, self.long = [], [], []
        for _ in range(n_long):
            self.long.append(self.new("long", long_target))
        self.n_active = n_active

    def new(self, kind, target=None, messages=None):
        with self.lock:
            self.n += 1
            name = f"{kind}{self.n}"
            if messages is None:
                f = os.path.basename(self.rng.choice(CORPUS)[0])
                task = self.rng.choice(TASKS).format(f=f)
                messages = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": f"[{name}] {task}"}]
            return Session(name, messages, target or self.rng.randint(12000, 60000), kind)

    def take(self, which):
        """A session to work on (marked busy), or None."""
        with self.lock:
            if which == "long":  # a finished one is replaced by a new one
                for i, s in enumerate(self.long):
                    if s.done and not s.busy:
                        self.retired.append(s)
                        self.long[i] = self.new("long", self.long_target)
                c = [s for s in self.long if not s.busy and time.time() - s.t_last >= self.LONG_EVERY]
            elif which == "retired":
                c = [s for s in self.retired if not s.busy]
            else:
                live = [s for s in self.active if not s.done]
                while len(live) < self.n_active:
                    s = self.new("s")
                    self.active.append(s)
                    live.append(s)
                c = [s for s in live if not s.busy]
                if which == "big":  # a pending tool output large enough to cut its prefill off
                    c = [s for s in c if s.pending >= 6000]
                elif which == "grown":
                    c = [s for s in c if s.tokens >= 24000]
                elif which == "forkable":
                    c = [s for s in c if s.turns >= 3]
                elif which == "regen":
                    c = [s for s in c if s.last is not None]
            if not c:
                return None
            s = self.rng.choice(c)
            s.busy = True
            return s

    def give(self, s):
        with self.lock:
            s.busy = False
            if s.done and s in self.active:
                self.active.remove(s)
                self.retired.append(s)
                del self.retired[:-self.keep]

    def add(self, s):
        with self.lock:
            if s.done:
                self.retired.append(s)
                del self.retired[:-self.keep]
            else:
                self.active.append(s)

    def claim(self, s):
        """s itself, marked busy, unless another client has it."""
        with self.lock:
            if s.busy:
                return False
            s.busy = True
            return True

    def known(self):
        with self.lock:
            return self.active + self.long + self.retired


def body_for(rng, messages, stream=True, max_tokens=None, greedy=None, thinking=None):
    b = {"model": "deepseek-v4.1-flash-uncensored", "messages": messages, "tools": TOOLS,
         "max_tokens": max_tokens or rng.choice([64, 128, 160, 256])}
    if stream:
        b.update(stream=True, stream_options={"include_usage": True})
    if greedy if greedy is not None else rng.random() < 0.3:
        b["temperature"] = 0
    else:
        b.update(temperature=0.7, top_p=0.95)
    if thinking if thinking is not None else rng.random() < 0.15:
        b["chat_template_kwargs"] = {"thinking": True}
    return b


def grow(rng, s, rec, long=False):
    """Append the model's reply and the tool output(s) it asked for (or a user message with one)."""
    n = tool_size(rng, long)
    calls = rec.get("calls") or []
    if calls:
        ids = [f"call_{s.name}_{s.turns}_{i}" for i in range(len(calls))]
        s.messages.append({"role": "assistant", "content": rec.get("content") or "",
                           "tool_calls": [{"id": i, "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                          for i, c in zip(ids, calls)]})
        for i, c in zip(ids, calls):
            s.messages.append({"role": "tool", "tool_call_id": i, "content": tool_text(rng, n // len(calls))})
    else:
        s.messages.append({"role": "assistant", "content": rec.get("content") or "(no answer)"})
        s.messages.append({"role": "user", "content": "Here is what run_command printed:\n" + tool_text(rng, n) + "\nGo on."})
    s.pending = n


class Soak:
    def __init__(self, a):
        self.a = a
        self.cl = Client(a.url, open(os.path.expanduser(a.key_file)).read().strip())
        self.req = Recorder(os.path.join(a.out, "requests.jsonl"))
        self.rng = random.Random(a.seed)
        self.pool = Pool(random.Random(a.seed + 1), a.active, a.long, a.long_target, a.keep)
        self.stop_at = time.time() + 60 * a.minutes
        self.next_golden = time.time() + 120
        self.glock = threading.Lock()
        self.next_cold = time.time() + a.cold_every
        self.clock = threading.Lock()
        self.n_cold = 0
        self.extra = []  # sessions this harness made outside the pool (cold runs, golden replays), for tiers()

    def record(self, kind, s, body, o, extra=None):
        rec, bad = judge(body, o)
        u = rec.get("usage") or {}
        out = {"t": rec["t_send"], "kind": kind, "session": s.name if s else None, "stream": bool(body.get("stream")),
               "greedy": body.get("temperature") == 0, "thinking": "chat_template_kwargs" in body,
               "max_tokens": body["max_tokens"], "status": rec["status"], "aborted": rec["aborted"],
               "ttft": rec["ttft"], "total": rec["total"], "finish": rec.get("finish"),
               "prompt": u.get("prompt_tokens"), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "completion": u.get("completion_tokens"), "calls": len(rec.get("calls") or []),
               "unknown_tools": sum(1 for c in rec.get("calls") or [] if c["name"] not in TOOL_NAMES), "bad": bad}
        if out["completion"] and rec["ttft"] is not None and rec["total"] and rec["total"] > rec["ttft"]:
            out["tok_s"] = out["completion"] / (rec["total"] - rec["ttft"])
        if not rec["aborted"] and not bad:  # to compare greedy answers to the same prompt
            out["out_hash"] = hashlib.sha1(json.dumps(answer(rec)).encode()).hexdigest()[:12]
        if extra:
            out.update(extra)
        self.req.write(out)
        if bad:
            print(f"{time.strftime('%T')} {kind} {out['session']}: {bad}", flush=True)
        return rec, bad

    # one turn of a session; the kinds of request below are variations of it
    def turn(self, s, kind="turn", stream=True, long=False, **kw):
        body = body_for(self.rng, s.messages, stream=stream)
        s.ref = None  # its state moves on (or is cut short)
        o = self.cl.call(body, **kw)
        s.t_last = time.time()
        rec, bad = self.record(kind, s, body, o, {"tokens": s.tokens})
        if not bad and not o["aborted"]:
            u = rec.get("usage") or {}
            s.last = list(s.messages)
            s.tokens = (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
            s.turns += 1
            if s.tokens >= s.target:
                s.done = True
                self.take_ref(s)  # put aside: to be revisited
            else:
                grow(self.rng, s, rec, long)
        return rec, bad

    # --disk: probes of a session's state, before it is put aside and when it comes back
    def probe(self, s, role, extra=None):
        """The session's last prompt again, greedy, 64 tokens, thinking off: the answer's hash, or None."""
        body = body_for(self.rng, s.last, greedy=True, thinking=False, max_tokens=64)
        o = self.cl.call(body)
        s.t_last = time.time()
        rec, bad = self.record("probe", s, body, o, dict({"role": role, "ref": s.ref_n, "tokens": s.tokens}, **(extra or {})))
        return None if bad or o["aborted"] else hashlib.sha1(json.dumps(answer(rec)).encode()).hexdigest()[:12]

    def take_ref(self, s):
        """Two probes: the first may restart anywhere; the second restarts from the live ring at the prompt's
        last position, as a later probe does from the state the session comes back with."""
        if not self.a.disk or s.last is None:
            return
        s.ref_n += 1
        s.ref = None
        if self.probe(s, "ref1") is not None:
            s.ref = self.probe(s, "ref2")

    def in_table(self):
        """The sessions the front-end's table should still hold: the most recently used, within its limits
        less a margin (it also holds the golden replays, the cold runs and the answers)."""
        out, n = set(), 0
        for i, s in enumerate(sorted(self.pool.known(), key=lambda s: -s.t_last)):
            n += s.tokens + 2048
            if i >= FE_TABLE[0] - 40 or n > 0.8 * FE_TABLE[1]:
                break
            out.add(s.name)
        return out

    def tiers(self):
        """Where each session's state should be, by last use and estimated size: the store holds the most
        recent up to its budget, the file the next ones up to its size, the rest was dropped."""
        out, acc = {}, 0.0
        with self.pool.lock:
            known = self.pool.known() + self.extra
        for s in sorted(known, key=lambda s: -s.t_last):
            acc += est_mib(s.tokens)
            out[s.name] = ("store" if acc <= 1024 * self.a.store_gib else
                           "disk" if acc <= 1024 * (self.a.store_gib + self.a.disk_gib) else "gone")
        return out

    def put_aside(self, recent, min_tokens=0, largest=False):
        """A session put aside that the front-end still holds and that should be in the file (marked busy):
        the most recently spilled (recent), the largest, or any; else any the front-end holds."""
        tab, tier = self.in_table(), self.tiers()
        with self.pool.lock:
            c = [s for s in self.pool.retired if not s.busy and s.last and s.name in tab and s.tokens >= min_tokens]
            d = [s for s in c if tier.get(s.name) == "disk"] or c
            if not d:
                return None
            s = max(d, key=lambda s: s.t_last) if recent else max(d, key=lambda s: s.tokens) if largest else self.rng.choice(d)
            s.busy = True
            return s

    def act(self):
        r = self.rng.random()
        if time.time() >= self.next_golden and self.glock.acquire(blocking=False):
            try:
                self.next_golden = time.time() + 600
                return self.golden()
            finally:
                self.glock.release()
        if self.a.disk and time.time() >= self.next_cold and self.clock.acquire(blocking=False):
            try:
                self.next_cold = time.time() + self.a.cold_every
                return self.cold_pair()
            finally:
                self.clock.release()
        if r < 0.22:
            s = self.pool.take("long")
            if s:
                try:
                    probe = self.a.disk and s.ref is not None
                    if probe:  # its reference from the turn before: the state it comes back with
                        self.probe(s, "check")
                    rec = self.turn(s, "long", long=True)
                    if self.a.disk and not s.done and self.rng.random() < 0.5:
                        self.take_ref(s)
                    return rec
                finally:
                    self.pool.give(s)
        kinds = [("turn", 0.55), ("json", 0.05), ("regenerate", 0.06), ("fork", 0.06), ("cancel", 0.06),
                 ("disconnect", 0.05), ("reset", 0.03), ("compact", 0.04), ("revisit", 0.05)]
        if self.a.disk:  # more state, and more of it brought back
            kinds = [("turn", 0.36), ("json", 0.04), ("regenerate", 0.05), ("fork", 0.05), ("fork_long", 0.12),
                     ("cancel", 0.05), ("disconnect", 0.04), ("reset", 0.03), ("compact", 0.04), ("revisit", 0.22)]
        r = self.rng.random() * sum(w for _, w in kinds)
        for k, w in kinds:
            r -= w
            if r < 0:
                break
        getattr(self, "do_" + k)()

    def with_session(self, which, f):
        s = self.pool.take(which)
        if s is None:
            s = self.pool.take("any")
            if s is None:
                return
            which = "any"
        try:
            f(s, which)
        finally:
            self.pool.give(s)

    def do_turn(self):
        self.with_session("any", lambda s, w: self.turn(s))

    def do_json(self):
        self.with_session("any", lambda s, w: self.turn(s, "json", stream=False))

    def do_cancel(self):
        self.with_session("any", lambda s, w: self.turn(s, "cancel", cancel_after=self.rng.randint(1, 30)))

    def do_reset(self):
        self.with_session("any", lambda s, w: self.turn(s, "reset", cancel_after=self.rng.randint(1, 30), reset=True))

    def do_disconnect(self):
        # the client goes away during the prefill of a large tool output; the next turn sends it again
        def f(s, which):
            if which != "big":
                return self.turn(s, "cancel", cancel_after=self.rng.randint(1, 10))
            return self.turn(s, "disconnect", disconnect_after=self.rng.uniform(0.15, 0.8))
        self.with_session("big", f)

    def do_regenerate(self):
        def f(s, which):
            if which != "regen":
                return self.turn(s)
            body = body_for(self.rng, s.last, greedy=True, thinking=False, max_tokens=64)
            r1, _ = self.record("regenerate", s, body, self.cl.call(body))
            o2 = self.cl.call(body)
            r2, _ = judge(body, o2)
            self.record("regenerate", s, body, o2, {"same": answer(r1) == answer(r2)})
        self.with_session("regen", f)

    def do_fork(self):
        # a branch from an earlier point of a session: its history to a tool result, a different result
        def f(s, which):
            if which != "forkable":
                return self.turn(s)
            cuts = [i for i, m in enumerate(s.messages) if m["role"] in ("tool", "user") and i >= 2]
            if not cuts:
                return self.turn(s)
            i = self.rng.choice(cuts)
            m = dict(s.messages[i], content=tool_text(self.rng, tool_size(self.rng)))
            t = self.pool.new("fork", messages=[dict(x) for x in s.messages[:i]] + [m])
            t.target = self.rng.randint(8000, 40000)
            self.turn(t, "fork")
            self.pool.add(t)
        self.with_session("forkable", f)

    def do_compact(self):
        # the history replaced by a summary, the system prompt kept: a new session that shares only it
        def f(s, which):
            if which != "grown":
                return self.turn(s)
            text = "\n".join(m["content"] for m in s.messages[1:] if isinstance(m.get("content"), str))
            k = int(self.rng.randint(2000, 6000) * CHARS_PER_TOKEN)
            at = self.rng.randrange(0, max(1, len(text) - k))
            m = {"role": "user", "content": f"[{s.name} compacted] The conversation so far, summarised:\n{text[at:at + k]}\n\nContinue the task."}
            t = self.pool.new("compact", messages=[s.messages[0], m])
            s.done = True
            self.turn(t, "compact")
            self.pool.add(t)
        self.with_session("grown", f)

    def do_fork_long(self):
        """A branch of a long session, or of one put aside with at least 32k tokens (likely spilled): its
        history to a tool result in its later half, then a different result. The branch is put aside after
        its first turn (with a reference), to be revisited."""
        with self.pool.lock:
            c = [s for s in self.pool.long + self.pool.retired if not s.busy and s.tokens >= 32000 and s.turns >= 2]
            s = self.rng.choice(c) if c else None
            if s:
                s.busy = True
        if s is None:
            return self.do_fork()
        try:
            cuts = [i for i, m in enumerate(s.messages) if m["role"] in ("tool", "user") and i >= max(2, len(s.messages) // 2)]
            if not cuts:
                return
            i = self.rng.choice(cuts)
            m = dict(s.messages[i], content=tool_text(self.rng, tool_size(self.rng)))
            f = self.pool.new("lfork", messages=[dict(x) for x in s.messages[:i]] + [m])
            f.target = 1
            s.t_last = time.time()
        finally:
            self.pool.give(s)
        self.turn(f, "fork_long")
        self.pool.add(f)

    def do_revisit(self):
        if self.a.disk:
            # the least recently used session put aside (spilled, or the next to spill), some right after a
            # branch has added state, or any; its probe again
            push = self.rng.random() < 0.3
            if push:  # a branch adds state, the store spills: then the session it most likely pushed out
                self.do_fork_long()
            s = self.put_aside(recent=push or self.rng.random() < 0.5)
            if s is None:
                return self.do_turn()
            try:
                if s.ref is None:
                    self.take_ref(s)
                else:
                    self.probe(s, "check", {"pushed": push})
            finally:
                self.pool.give(s)
            return
        # a session put aside long ago (its state likely evicted from the store): one more turn
        s = self.pool.take("retired")
        if s is None:
            return self.do_turn()
        try:
            body = body_for(self.rng, s.last or s.messages)
            o = self.cl.call(body)
            self.record("revisit", s, body, o, {"tokens": s.tokens})
        finally:
            self.pool.give(s)

    def cold_pair(self):
        """A revisit (its probe: from the disk when it was spilled) against the same prompt cold, the system
        prompt changed at its start so that the front-end shares nothing: the larger sessions put aside, and
        the least recently used one, in turn."""
        self.n_cold += 1
        s = self.put_aside(recent=self.n_cold % 2 == 0, min_tokens=32000, largest=True)
        if s is None:
            return
        try:
            if s.ref is None:
                self.probe(s, "ref1", {"cold_pair": self.n_cold})
            else:
                self.probe(s, "check", {"cold_pair": self.n_cold})
            msgs = [dict(m) for m in s.last]
            msgs[0] = dict(msgs[0], content=f"[cold {self.n_cold}]\n" + msgs[0]["content"])
            body = body_for(self.rng, msgs, greedy=True, thinking=False, max_tokens=64)
            self.record("cold", s, body, self.cl.call(body), {"cold_pair": self.n_cold, "tokens": s.tokens})
            c = Session(f"cold{self.n_cold}", None, 0, "cold")
            c.tokens, c.t_last = s.tokens, time.time()
            self.extra.append(c)
        finally:
            self.pool.give(s)

    def golden(self):
        """The golden agentic transcript turn by turn: each request is the history up to an assistant turn."""
        msgs = GOLDEN["messages"]
        s = Session("golden", [], 0, "golden")
        g = Session(f"golden{len(self.extra)}", None, 0, "golden")  # its state, for tiers()
        self.extra.append(g)
        for i, m in enumerate(msgs):
            if m["role"] != "assistant" or time.time() >= self.stop_at:
                continue
            body = body_for(self.rng, msgs[:i], greedy=True, thinking=True, max_tokens=64)
            o = self.cl.call(body)
            rec, _ = self.record("golden", s, body, o, {"turn": i})
            g.tokens, g.t_last = (rec.get("usage") or {}).get("prompt_tokens") or g.tokens, time.time()

    def client(self, wid):
        while time.time() < self.stop_at:
            try:
                self.act()
            except Exception as e:  # the harness's own failure: recorded as an error, and on
                import traceback
                self.req.write({"t": time.time(), "kind": "harness", "bad": [f"{type(e).__name__}: {e}"],
                                "trace": traceback.format_exc()[-1500:]})
                print(f"harness error: {e}", flush=True)
                time.sleep(1)


# ------------------------------------------------------------------ monitoring

RE_REQ = re.compile(r"serve: req (\d+) session (\w+): prompt (\d+), reused (\d+) \((\w+)\), prefill ([\d.]+) ms "
                    r"\((\d+) tok/s\), (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\), (\w+); store ([\d.]+) MiB")
RE_EVICT = re.compile(r"serve: (evict session|spill session|session \w+ evicted)")
RE_SPILL = re.compile(r"state: session (\d+) spilled to disk \(([\d.]+) MiB; ([\d.]+) of ([\d.]+) GiB of the file in use"
                      r"(?:; host ([\d.]+) ms)?\)")
RE_FLUSH = re.compile(r"spill: ([\d.]+) MiB to staging in ([\d.]+) ms on the stream \(([\d.]+) GB/s; checksums before it "
                      r"([\d.]+) ms\), then to the file in ([\d.]+) ms \(([\d.]+) GB/s\)")
RE_BACK = re.compile(r"state: session (\d+) back from disk: ([\d.]+) MiB in ([\d.]+) ms \(([\d.]+) GB/s; ([\d.]+) ms waiting "
                     r"for its writes(?:; (\d+) blocks checked, (\w+))?\)")
RE_SBACK = re.compile(r"serve: session (\w+) back from disk: ([\d.]+) MiB, (\d+) positions, in ([\d.]+) ms")
RE_DROP = re.compile(r"state: session (\d+) dropped from the disk tier")
# the spill file's pages in the page cache (cachestat(2), 451): O_DIRECT should keep it at none
CACHESTAT = base64.b64encode(b"""import ctypes, glob, os
libc = ctypes.CDLL(None, use_errno=True)
class R(ctypes.Structure): _fields_ = [("off", ctypes.c_uint64), ("len", ctypes.c_uint64)]
class C(ctypes.Structure): _fields_ = [(n, ctypes.c_uint64) for n in ("cache", "dirty", "wb", "ev", "rev")]
tot = 0
for p in glob.glob("/data/dsgb10/spill/*.spill") + glob.glob(os.path.expanduser("~/dsgb10/spill/*.spill")):
    try:
        fd = os.open(p, os.O_RDONLY)
    except OSError:
        continue
    c = C()
    if libc.syscall(451, fd, ctypes.byref(R(0, 0)), ctypes.byref(c), 0) == 0:
        tot += c.cache * os.sysconf("SC_PAGE_SIZE")
    os.close(fd)
print(tot)
""").decode()
RE_FE = re.compile(r"req (\d+) session (\d+): prompt (\d+) \(cached (\d+)\), completion (\d+), (\w+); ttft ([\d.]+) s, ([\d.]+) tok/s")
# dsfront's session table (frontend/src/server.rs: SessionTable::new(256, 16 << 20)), and what a token
# it holds may cost: its id (4 bytes), its text's end (8) and its text (about 4), with the vectors'
# spare capacity up to as much again
TABLE_SESSIONS, TABLE_TOKENS, TABLE_BYTES = 256, 16 << 20, 32


class Monitor:
    def __init__(self, a):
        self.a = a
        self.mem = Recorder(os.path.join(a.out, "memory.jsonl"))
        self.srv = Recorder(os.path.join(a.out, "server.jsonl"))
        self.files = {}
        self.stop = threading.Event()
        self.back, self.back_by = None, {}  # a read-back's numbers, until its request's line

    def tail(self, name, path):
        path = os.path.expanduser(path)
        try:
            st = os.stat(path)
        except OSError:
            return []
        pos, ino = self.files.get(name, (None, None))
        if pos is None:
            pos = st.st_size  # from now on
        if ino is not None and (st.st_ino != ino or st.st_size < pos):
            pos = 0  # rotated or truncated
        with open(path, "rb") as f:
            f.seek(pos)
            data = f.read()
        nl = data.rfind(b"\n") + 1
        self.files[name] = (pos + nl, st.st_ino)
        return data[:nl].decode("utf-8", "replace").splitlines()

    def logs(self):
        t = time.time()
        for line in self.tail("rank0", os.path.join(self.a.logs, "rank.0.log")):
            m = RE_REQ.search(line)
            if m:
                g = m.groups()
                rec = {"t": t, "src": "engine", "req": int(g[0]), "session": g[1], "prompt": int(g[2]),
                       "reused": int(g[3]), "how": g[4], "prefill_ms": float(g[5]), "prefill_tok_s": int(g[6]),
                       "tokens": int(g[7]), "decode_ms": float(g[8]), "tok_s": float(g[9]), "why": g[10],
                       "store_mib": float(g[11])}
                rec.update(self.back_by.pop(g[1], {}))
                self.srv.write(rec)
                continue
            m = RE_SPILL.search(line)
            if m:
                g = m.groups()
                self.srv.write({"t": t, "src": "spill", "h": int(g[0]), "mib": float(g[1]), "file_gib": float(g[2]),
                                "cap_gib": float(g[3]), "host_ms": float(g[4]) if g[4] else None})
                continue
            m = RE_FLUSH.search(line)
            if m:
                g = [float(x) for x in m.groups()]
                self.srv.write({"t": t, "src": "flush", "mib": g[0], "copy_ms": g[1], "copy_gbs": g[2], "sum_ms": g[3],
                                "write_ms": g[4], "write_gbs": g[5]})
                continue
            m = RE_BACK.search(line)
            if m:
                g = m.groups()
                self.back = {"disk_mib": float(g[1]), "disk_ms": float(g[2]), "disk_gbs": float(g[3]),
                             "disk_wait_ms": float(g[4]), "blocks": int(g[5]) if g[5] else None, "fp": g[6]}
                self.srv.write(dict({"t": t, "src": "back", "h": int(g[0])}, **self.back))
                continue
            m = RE_SBACK.search(line)
            if m:
                self.back_by[m.group(1)] = dict(self.back or {}, from_disk=True, back_ms=float(m.group(4)))
                self.back = None
                continue
            m = RE_DROP.search(line)
            if m:
                self.srv.write({"t": t, "src": "drop", "h": int(m.group(1))})
                continue
            if RE_EVICT.search(line):
                self.srv.write({"t": t, "src": "evict", "line": line.strip()})
            elif re.search(r"error|fail|abort|core|watchdog", line, re.I) and "drafts" not in line:
                self.srv.write({"t": t, "src": "engine_error", "line": line.strip()})
        for line in self.tail("front", os.path.join(self.a.logs, "front.log")):
            m = RE_FE.search(line)
            if m:
                g = m.groups()
                self.srv.write({"t": t, "src": "front", "req": int(g[0]), "prompt": int(g[2]), "cached": int(g[3]),
                                "completion": int(g[4]), "finish": g[5], "ttft": float(g[6]), "tok_s": float(g[7])})
            elif "repetition loop" in line:
                self.srv.write({"t": t, "src": "front_loop", "line": line.strip()})
            elif re.search(r"error|panic|lost|refus", line, re.I):
                self.srv.write({"t": t, "src": "front_error", "line": line.strip()})

    def sample(self):
        # the Engram row cache is an anonymous mapping of --engram-cache-mib that fills as rows are first
        # read: its resident part is counted apart from the rest of the rank's resident set
        kb = self.a.engram_cache_mib * 1024
        cmd = ("ps -o pid=,rss= -C dsgb10_rank; echo --; nvidia-smi --query-compute-apps=pid,used_memory "
               "--format=csv,noheader,nounits; echo --; awk '/^MemAvailable:/ {print $2}' /proc/meminfo; echo --; "
               "for p in $(pgrep -x dsgb10_rank); do awk '/^Size:/ {sz=$2} /^Rss:/ {if (sz==" + str(kb) +
               ") c+=$2} END {print c+0}' /proc/$p/smaps; done; echo --; "
               # pinned host memory (cudaHostAlloc: shared mappings of /dev/zero), the disk tier's staging among it
               "for p in $(pgrep -x dsgb10_rank); do awk '/^[0-9a-f]+-[0-9a-f]+ / {z = ($2 ~ /s/ && /\\/dev\\/zero/)} "
               "/^Rss:/ {if (z) c+=$2} END {print c+0}' /proc/$p/smaps; done; echo --; "
               "python3 -c \"$(echo " + CACHESTAT + " | base64 -d)\"; echo --; "
               "awk '/^(Dirty|Writeback):/ {s+=$2} END {print s+0}' /proc/meminfo")
        t = time.time()
        for rank, (box, addr) in enumerate(BOXES):
            try:
                argv = ["bash", "-c", cmd] if addr is None else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", addr, cmd]
                out = subprocess.run(argv, capture_output=True, text=True, timeout=20).stdout
                ps, gpu, avail, cache, pinned, pcache, dirty = (out.split("--\n") + [""] * 7)[:7]
                pids = {int(p): int(r) for p, r in (l.split() for l in ps.split("\n") if l.strip())}
                gmib = sum(int(m) for p, m in (l.split(",") for l in gpu.split("\n") if l.strip()) if int(p) in pids)
                rec = {"t": t, "rank": rank, "box": box, "pids": sorted(pids), "rss_mib": sum(pids.values()) / 1024,
                       "cache_mib": sum(int(c) for c in cache.split()) / 1024,
                       "gpu_mib": gmib, "avail_gib": int(avail.strip() or 0) / 1048576,
                       "pinned_mib": sum(int(c) for c in pinned.split()) / 1024,
                       "spill_cache_mib": int(pcache.strip() or 0) / 1048576 if pcache.strip().isdigit() else None,
                       "dirty_mib": int(dirty.strip() or 0) / 1024 if dirty.strip().isdigit() else None}
                if addr is None:
                    rec["front_mib"] = self.front_mib()
                self.mem.write(rec)
            except Exception as e:
                self.mem.write({"t": t, "rank": rank, "box": box, "error": f"{type(e).__name__}: {e}"})

    def front_mib(self):
        """The front-end's resident set (MiB), by the service's pid file; None without one."""
        try:
            pid = open(os.path.expanduser(self.a.front_pid_file)).read().split()[0]
            if open(f"/proc/{pid}/comm").read().strip() != "dsfront":
                return None
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
        except (OSError, IndexError, ValueError):
            pass
        return None

    def run(self):
        last = 0
        while not self.stop.is_set():
            self.logs()
            if time.time() - last >= self.a.sample_s:
                last = time.time()
                self.sample()
            self.stop.wait(5)
        self.logs()
        self.sample()


# ------------------------------------------------------------------ report


def load(path):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except OSError:
        return []


def med(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if v else None


def pct(v, p):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def fmt(x, f="{:.0f}"):
    return "-" if x is None else f.format(x)


def report(out, window_min, engram_mib=4096, store_gib=11.25):
    req, srv, mem = (load(os.path.join(out, f)) for f in ("requests.jsonl", "server.jsonl", "memory.jsonl"))
    if not req:
        print("no requests recorded")
        return False
    t0 = min(r["t"] for r in req + mem)
    w = 60 * window_min
    nwin = int((max(r["t"] for r in req + mem + srv) - t0) // w) + 1
    win = lambda r: int((r["t"] - t0) // w)
    eng = [r for r in srv if r.get("src") == "engine"]
    fe = [r for r in srv if r.get("src") == "front"]
    print(f"{len(req)} requests over {(max(r['t'] for r in req) - t0) / 60:.1f} min; by kind: "
          + ", ".join(f"{k} {n}" for k, n in collections.Counter(r["kind"] for r in req).most_common()))
    print("window | reqs  bad | decode tok/s (engine, >=32 tok, ctx<32k) | ttft s (front, <1k new) med p90 | "
          "prefill tok/s (>=4k new) | store MiB, evictions | per rank: GPU MiB / RSS MiB / MemAvailable GiB | front-end RSS MiB")
    rows = []
    for i in range(nwin):
        R = [r for r in req if win(r) == i]
        E = [r for r in eng if win(r) == i]
        F = [r for r in fe if win(r) == i]
        M = [r for r in mem if win(r) == i and "error" not in r]
        dec = [r["tok_s"] for r in E if r["tokens"] >= 32 and r["prompt"] < 32768]
        small = [r["ttft"] for r in F if r["prompt"] - r["cached"] < 1000]
        pf = [r["prefill_tok_s"] for r in E if r["prompt"] - r["reused"] >= 4096]
        st = [r["store_mib"] for r in E]
        ev = sum(1 for r in srv if r.get("src") == "evict" and win(r) == i)
        per = []
        for k in range(4):
            mk = [m for m in M if m["rank"] == k]
            per.append(f"{fmt(max((m['gpu_mib'] for m in mk), default=None))}/{fmt(max((m['rss_mib'] for m in mk), default=None))}"
                       f"/{fmt(min((m['avail_gib'] for m in mk), default=None), '{:.1f}')}")
        bad = sum(1 for r in R if r.get("bad"))
        rows.append({"dec": med(dec), "ttft": med(small), "pf": med(pf)})
        print(f"{i * window_min:3g}-{(i + 1) * window_min:<3g}| {len(R):4d} {bad:4d} | {fmt(med(dec), '{:.1f}'):>6} (n {len(dec):3d})"
              f"                    | {fmt(med(small), '{:.3f}')} {fmt(pct(small, 0.9), '{:.3f}')} (n {len(small)}) | "
              f"{fmt(med(pf))} (n {len(pf)}) | {fmt(max(st, default=None))}, {ev} | " + "  ".join(per)
              + f" | {fmt(max((m['front_mib'] for m in M if m.get('front_mib') is not None), default=None))}")
    # pass criteria
    ok = True
    bad = [r for r in req if r.get("bad")]
    errs = [r for r in srv if r.get("src") in ("engine_error", "front_error")]
    print(f"\nerrors: {len(bad)} requests not well formed or failed; {len(errs)} error lines in the logs")
    for r in bad[:8]:
        print(f"  {time.strftime('%T', time.localtime(r['t']))} {r['kind']} {r.get('session')}: {r['bad'][:2]}")
    for r in errs[:8]:
        print(f"  log: {r['line'][:160]}")
    ok &= not bad and not errs
    print(f"tool calls: {sum(r.get('calls') or 0 for r in req)}, of tools not offered: {sum(r.get('unknown_tools') or 0 for r in req)}; "
          f"repetition loops stopped by the front-end: {sum(1 for r in srv if r.get('src') == 'front_loop')}; regenerates "
          f"identical (greedy): {sum(1 for r in req if r.get('same'))} of {sum(1 for r in req if 'same' in r)}")
    for k in range(4):
        mk = [m for m in mem if m.get("rank") == k and "error" not in m and m["pids"]]
        if not mk:
            print(f"rank {k}: no memory samples")
            ok = False
            continue
        g = [m["gpu_mib"] for m in mk]
        rs = [m["rss_mib"] for m in mk]
        av = [m["avail_gib"] for m in mk]
        pids = {tuple(m["pids"]) for m in mk}
        # the Engram row cache fills up to its size: the rest of the resident set must stay flat
        if all("cache_mib" in m for m in mk):
            rest = [m["rss_mib"] - m["cache_mib"] for m in mk]
            cache = f", of which the Engram cache {mk[0]['cache_mib']:.0f} -> {mk[-1]['cache_mib']:.0f} and the rest {rest[0]:.0f} -> max {max(rest):.0f}"
            rss_ok = max(rest) - rest[0] <= 256
        else:  # older records: only the bound
            cache = f" (the Engram cache's {engram_mib} MiB fill included)"
            rss_ok = max(rs) - rs[0] <= engram_mib + 256
        # MemAvailable is the box's, and other jobs share the boxes: its median over the first window
        # against the last (it may fall by what the rank grew, plus 1 GiB), the lowest shown
        grown = (max(rs) - rs[0]) / 1024
        av0 = med([m["avail_gib"] for m in mk if m["t"] < mk[0]["t"] + w])
        av1 = med([m["avail_gib"] for m in mk if m["t"] > mk[-1]["t"] - w])
        flat = max(g) - g[0] <= 256 and rss_ok and av0 - av1 <= grown + 1.0
        print(f"rank {k}: GPU {g[0]:.0f} -> max {max(g):.0f} MiB; RSS {rs[0]:.0f} -> max {max(rs):.0f} MiB{cache}; MemAvailable "
              f"{av0:.1f} -> {av1:.1f} GiB (lowest {min(av):.1f}); {len(pids)} process{'es' if len(pids) > 1 else ''}: "
              f"{'flat' if flat else 'NOT FLAT'}")
        ok &= flat and len(pids) == 1
    # latency: the first two windows against the last two, on comparable requests
    early = lambda r: win(r) < 2
    late = lambda r: win(r) >= nwin - 2
    for name, rs, val, higher in (
            ("decode tok/s (engine; >= 32 tokens, context < 32k)", eng, lambda r: r["tok_s"] if r["tokens"] >= 32 and r["prompt"] < 32768 else None, True),
            # (a session read back from the disk counts its reading in its prefill: shown in the disk tier's part)
            ("prefill ms of turns adding < 1k tokens (engine)", eng, lambda r: r["prefill_ms"] if r["prompt"] - r["reused"] < 1000 and not r.get("from_disk") else None, False),
            ("ttft s of turns adding < 1k tokens (front-end, with queueing)", fe, lambda r: r["ttft"] if r["prompt"] - r["cached"] < 1000 else None, False),
            ("prefill tok/s of turns adding >= 4k tokens (engine)", eng, lambda r: r["prefill_tok_s"] if r["prompt"] - r["reused"] >= 4096 else None, True)):
        a, b = med([val(r) for r in rs if early(r)]), med([val(r) for r in rs if late(r)])
        if a and b and nwin >= 4:
            change = b / a - 1
            flat = (change >= -0.10) if higher else (change <= 0.10)
            print(f"{name}: first 2 windows {a:.3f}, last 2 {b:.3f} ({100 * change:+.1f} %): {'flat' if flat else 'NOT FLAT'}")
            ok &= flat or "front-end" in name  # queueing makes the client's view noisy: shown, not judged
    # the front-end's session table is bounded, so its memory grows to a level. What it holds is estimated
    # from rank 0's records (each session's latest prompt and output). Once the table was full by the
    # middle of the run: the second half's peak against the first half's. While it was still filling (a
    # short run): the resident set against what the table held, at TABLE_BYTES per token. Either way
    # never above a full table's.
    fm = sorted((m for m in mem if m.get("front_mib") is not None), key=lambda m: m["t"])
    if fm:
        held, i, pts = {}, 0, []
        es = sorted(eng, key=lambda r: r["t"])
        for m in fm:
            while i < len(es) and es[i]["t"] <= m["t"]:
                held[es[i]["session"]] = es[i]["prompt"] + es[i]["tokens"]
                i += 1
            pts.append((m["t"], m["front_mib"], len(held), sum(held.values())))
        half = (fm[0]["t"] + fm[-1]["t"]) / 2
        a = max(p[1] for p in pts if p[0] <= half)
        b = max((p[1] for p in pts if p[0] > half), default=a)
        mid = [p for p in pts if p[0] <= half][-1]
        rss0 = pts[0][1]
        limit = lambda tokens: rss0 + TABLE_BYTES * tokens / 2**20 + 64
        ceiling = limit(TABLE_TOKENS)
        if mid[2] >= TABLE_SESSIONS or mid[3] >= TABLE_TOKENS:
            flat = b - a <= 64
            how = f"the table full by the middle; the peak of the first half {a:.0f}, of the second {b:.0f}"
        else:
            over = max(p[1] - limit(p[3]) for p in pts)
            flat = over <= 0
            how = (f"the table still filling ({pts[-1][2]} sessions and {pts[-1][3] / 1e6:.1f}M tokens held at the end, of "
                   f"{TABLE_SESSIONS} and {TABLE_TOKENS / 2**20:.0f}M): {rss0:.0f} MiB + {TABLE_BYTES} bytes per token held "
                   f"+ 64 MiB, {'within it by ' + format(-over, '.0f') + ' MiB' if flat else format(over, '.0f') + ' MiB over it'}")
        n = len(pts)
        mx, my = sum(p[3] for p in pts) / n, sum(p[1] for p in pts) / n
        k = sum((p[3] - mx) * (p[1] - my) for p in pts) / max(1e-9, sum((p[3] - mx) ** 2 for p in pts))
        flat &= max(p[1] for p in pts) <= ceiling
        print(f"front-end: RSS {fm[0]['front_mib']:.0f} -> {fm[-1]['front_mib']:.0f} MiB; {how}; a full table's ceiling "
              f"{ceiling:.0f} MiB (fit: {my - k * mx:.0f} MiB + {k * 2**20:.0f} bytes per token held): {'flat' if flat else 'NOT FLAT'}")
        ok &= flat
    nev = sum(1 for r in srv if r.get("src") == "evict")
    peak = max((r['store_mib'] for r in eng), default=0)
    print(f"store: {nev} evictions, at most {peak:.0f} MiB; sessions "
          f"{len({r['session'] for r in eng})}; longest prompt {max((r['prompt'] for r in eng), default=0)} tokens")
    # the store must evict once full; a run too short to fill it (under 90 % of its budget between
    # requests, --store-gib) has nothing to evict, and is not judged on it
    if nev or peak >= 0.9 * 1024 * store_gib:
        ok &= nev > 0
    else:
        print(f"  no evictions to expect: the store peaked at {peak / 1024:.1f} GiB of its {store_gib:g} (--store-gib)")
    # the golden transcript, replayed every 10 minutes with the same prompts (greedy): the cleanest probe
    # of drift, matched to rank 0's records by prompt length and time
    gold = [r for r in req if r["kind"] == "golden" and r.get("prompt")]
    if gold:
        print("golden replays (the same prompts, greedy, 64 tokens each): engine decode tok/s over the replay, "
              "prefill ms summed, tokens reused")
        reps = collections.defaultdict(list)
        g0 = gold[0]["t"]
        for r in gold:
            reps[int((r["t"] - g0) // 300)].append(r)
        for k in sorted(reps):
            es = []
            for r in reps[k]:
                c = [e for e in eng if e["prompt"] == r["prompt"] and abs(e["t"] - r["t"]) < 150]
                if c:
                    es.append(min(c, key=lambda e: abs(e["t"] - r["t"])))
            dec = sum(e["tokens"] for e in es) / max(1e-9, sum(e["decode_ms"] for e in es) / 1e3)
            hashes = "".join(r.get("out_hash", "?")[:1] for r in reps[k])
            print(f"  {time.strftime('%T', time.localtime(reps[k][0]['t']))}: {len(reps[k])} turns, {dec:.1f} tok/s, "
                  f"prefill {sum(e['prefill_ms'] for e in es):.0f} ms, reused {sum(e['reused'] for e in es)} of "
                  f"{sum(e['prompt'] for e in es)} tokens; answers {hashes}")
    if any(r.get("src") in ("spill", "back") for r in srv) or any(r["kind"] == "probe" for r in req):
        ok &= disk_report(req, srv, mem, eng, t0, w, nwin)
    print("PASS" if ok else "FAIL")
    return ok


def match_engine(recs, eng):
    """Each request's rank-0 line: the same prompt length, read after it was sent, the first not taken (rank
    0 runs requests one at a time and logs them in order)."""
    by = collections.defaultdict(list)
    for e in eng:
        by[e["prompt"]].append(e)
    used = set()
    out = {}
    for r in sorted(recs, key=lambda r: r["t"]):
        for e in by.get(r.get("prompt"), []):
            if id(e) not in used and r["t"] - 1 <= e["t"] <= r["t"] + (r.get("total") or 0) + 30:
                used.add(id(e))
                out[id(r)] = e
                break
    return out


def disk_report(req, srv, mem, eng, t0, w, nwin):
    ok = True
    win = lambda r: int((r["t"] - t0) // w)
    sp = [r for r in srv if r.get("src") == "spill"]
    fl = [r for r in srv if r.get("src") == "flush"]
    bk = [r for r in srv if r.get("src") == "back"]
    dr = [r for r in srv if r.get("src") == "drop"]
    mins = max(1e-9, (max(r["t"] for r in req) - t0) / 60)
    print(f"\ndisk tier: {len(sp)} spills ({sum(r['mib'] for r in sp) / 1024:.1f} GiB), {len(bk)} read back "
          f"({sum(r['disk_mib'] for r in bk) / 1024:.1f} GiB), {len(dr)} dropped from the file; per minute {len(sp) / mins:.1f} "
          f"spills, {len(bk) / mins:.1f} read-backs; the file at most {max((r['file_gib'] for r in sp), default=0):.2f} of "
          f"{max((r['cap_gib'] for r in sp), default=0):.2f} GiB")
    if bk:
        big = [r for r in bk if r["disk_mib"] >= 32]
        waited = [r for r in bk if r["disk_wait_ms"] > 0]
        print(f"  read-backs of >= 32 MiB: {len(big)}, median {fmt(med([r['disk_ms'] for r in big]), '{:.0f}')} ms for "
              f"{fmt(med([r['disk_mib'] for r in big]), '{:.0f}')} MiB, {fmt(med([r['disk_gbs'] for r in big]), '{:.2f}')} GB/s "
              f"(p10 {fmt(pct([r['disk_gbs'] for r in big], 0.1), '{:.2f}')}); {len(waited)} waited for their own writes "
              f"(at most {fmt(max((r['disk_wait_ms'] for r in waited), default=None), '{:.0f}')} ms); "
              f"{sum(r['blocks'] or 0 for r in bk)} blocks checked against their checksums, "
              f"{sum(1 for r in bk if r.get('fp') == '0' * 16)} read-backs with a zero fingerprint")
    if fl:
        print(f"  copy into staging (rank 0, on the stream): median {fmt(med([r['copy_ms'] for r in fl]), '{:.2f}')} ms for "
              f"{fmt(med([r['mib'] for r in fl]), '{:.1f}')} MiB, {fmt(med([r['copy_gbs'] for r in fl]), '{:.1f}')} GB/s "
              f"(p90 {fmt(pct([r['copy_ms'] for r in fl], 0.9), '{:.2f}')} ms, at most {max(r['copy_ms'] for r in fl):.2f}); "
              f"checksums before it {fmt(med([r['sum_ms'] for r in fl]), '{:.2f}')} ms; host side of a spill "
              f"{fmt(med([r['host_ms'] for r in sp if r.get('host_ms') is not None]), '{:.2f}')} ms (median); "
              f"to the file {fmt(med([r['write_gbs'] for r in fl]), '{:.2f}')} GB/s")
    print("  window | spills GiB | read back | dropped | file GiB | read-back ms (median) | staging copy ms (median)")
    for i in range(nwin):
        S = [r for r in sp if win(r) == i]
        B = [r for r in bk if win(r) == i]
        F = [r for r in fl if win(r) == i]
        print(f"  {i}: {len(S):4d} {sum(r['mib'] for r in S) / 1024:5.1f} | {len(B):4d} | {sum(1 for r in dr if win(r) == i):4d} | "
              f"{fmt(max((r['file_gib'] for r in S), default=None), '{:.2f}')} | {fmt(med([r['disk_ms'] for r in B]), '{:.0f}')} | "
              f"{fmt(med([r['copy_ms'] for r in F]), '{:.2f}')}")
    ok &= len(sp) > 0 and len(bk) > 0
    # probes: a check against its reference's second probe, where rank 0 restarted both at the same position
    pr = [r for r in req if r["kind"] in ("probe", "cold")]
    m = match_engine(pr, eng)
    refs = {}
    for r in pr:
        if r["kind"] == "probe" and r.get("role") == "ref2":
            refs[(r["session"], r["ref"])] = r
    stats = collections.Counter()
    for r in pr:
        if r["kind"] != "probe" or r.get("role") != "check":
            continue
        e = m.get(id(r))
        ref = refs.get((r["session"], r["ref"]))
        er = m.get(id(ref)) if ref else None
        src = "unmatched" if e is None else "disk" if e.get("from_disk") else "cold" if e["how"] == "cold" else "store"
        if ref is None or er is None or e is None or r.get("out_hash") is None or ref.get("out_hash") is None:
            stats[(src, "no reference")] += 1
        elif e["reused"] != er["reused"]:
            stats[(src, "restarted elsewhere")] += 1
        elif r["out_hash"] == ref["out_hash"]:
            stats[(src, "equal")] += 1
        else:
            stats[(src, "DIFFERENT")] += 1
            print(f"  probe differs: {r['session']} reference {r['ref']}, prompt {r.get('prompt')}, reused {e['reused']}")
    print(f"probes: {sum(1 for r in pr if r.get('role') == 'ref2')} references; checks: "
          + "; ".join(f"from {s} {k} {n}" for (s, k), n in sorted(stats.items())))
    ok &= not any(k == "DIFFERENT" for _, k in stats) and stats[("disk", "equal")] > 0
    # revisits against cold
    pairs = collections.defaultdict(dict)
    for r in pr:
        if r.get("cold_pair"):
            pairs[r["cold_pair"]]["cold" if r["kind"] == "cold" else "revisit"] = r
    if pairs:
        print("revisit against cold: prompt | revisit: from | engine ms to the first token (of it reading back) | client ttft s | "
              "cold: engine ms | client ttft s")
        for k in sorted(pairs):
            a, c = pairs[k].get("revisit"), pairs[k].get("cold")
            ea, ec = m.get(id(a)) if a else None, m.get(id(c)) if c else None
            frm = "-" if not ea else "disk" if ea.get("from_disk") else ea["how"]
            print(f"  {fmt(a and a.get('prompt'))} | {frm} | {fmt(ea and ea['prefill_ms'])} ({fmt(ea and ea.get('disk_ms'))}) | "
                  f"{fmt(a and a.get('ttft'), '{:.3f}')} | {fmt(ec and ec['prefill_ms'])} | {fmt(c and c.get('ttft'), '{:.3f}')}")
    # memory the disk tier adds: pinned (flat), the spill file's page cache (none), dirty pages
    for k in range(4):
        mk = [x for x in mem if x.get("rank") == k and "error" not in x and x.get("pinned_mib") is not None]
        if not mk:
            continue
        pc = [x["spill_cache_mib"] for x in mk if x.get("spill_cache_mib") is not None]
        dt = [x["dirty_mib"] for x in mk if x.get("dirty_mib") is not None]
        flat = max(x["pinned_mib"] for x in mk) - mk[0]["pinned_mib"] <= 16 and max(pc, default=0) < 64
        print(f"rank {k}: pinned {mk[0]['pinned_mib']:.0f} -> max {max(x['pinned_mib'] for x in mk):.0f} MiB; the spill "
              f"file in the page cache at most {fmt(max(pc, default=None), '{:.1f}')} MiB; the box's dirty and writeback "
              f"pages at most {fmt(max(dt, default=None), '{:.0f}')} MiB: {'ok' if flat else 'NOT OK'}")
        ok &= flat
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18893")
    ap.add_argument("--key-file", default="~/dsgb10-data/dev-api-key")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--out", default=None)
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--active", type=int, default=6, help="ordinary sessions in play at once")
    ap.add_argument("--long", type=int, default=3, help="sessions grown to ~150k tokens")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--logs", default="~/dsgb10/service/logs")
    ap.add_argument("--front-pid-file", default="~/dsgb10/service/run/front.pid", help="to sample the front-end's memory")
    ap.add_argument("--sample-s", type=float, default=30)
    ap.add_argument("--window-min", type=float, default=5)
    ap.add_argument("--engram-cache-mib", type=int, default=4096, help="the server's, to tell its row cache in the resident set")
    ap.add_argument("--report", default=None, help="print the summary of an earlier run's --out")
    ap.add_argument("--disk", action="store_true", help="load the store's disk tier (see the top)")
    ap.add_argument("--long-target", type=int, default=150000, help="tokens a long session grows to")
    ap.add_argument("--keep", type=int, default=64, help="sessions put aside that stay candidates for revisits")
    ap.add_argument("--cold-every", type=float, default=300, help="--disk: seconds between revisits against cold")
    ap.add_argument("--store-gib", type=float, default=11.25, help="the store's budget between requests (--disk: to guess "
                    "tiers; a run whose store stays under 90 %% of it is not judged on evictions)")
    ap.add_argument("--disk-gib", type=float, default=32, help="--disk: the file's size (to guess tiers)")
    a = ap.parse_args()
    if a.report:
        sys.exit(0 if report(a.report, a.window_min, a.engram_cache_mib, a.store_gib) else 1)
    a.out = a.out or time.strftime("soak-%Y%m%d-%H%M%S")
    a.logs = os.path.expanduser(a.logs)
    os.makedirs(a.out, exist_ok=True)
    mon = Monitor(a)
    mt = threading.Thread(target=mon.run, daemon=True)
    mt.start()
    soak = Soak(a)
    print(f"soak: {a.minutes} min, {a.clients} clients, records in {a.out}", flush=True)
    th = [threading.Thread(target=soak.client, args=(i,)) for i in range(a.clients)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    mon.stop.set()
    mt.join()
    sys.exit(0 if report(a.out, a.window_min, a.engram_cache_mib, a.store_gib) else 1)


if __name__ == "__main__":
    main()
