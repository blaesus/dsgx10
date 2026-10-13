#!/usr/bin/env python3
"""bench.py, unchanged, with every request it sends recorded.

  python3 bench_record.py --record FILE.jsonl <bench.py's arguments>

bench.py measures exactly as it always has: this wrapper imports it and taps urllib's response, so each
SSE line reaches bench.py at the same moment, with one time.perf_counter() call and a list append added
per line. A request's record is written after bench.py's post_stream has taken its end time, so the
writing is outside every interval bench.py measures. One JSON line per request:

  t_wall   time.time() when the request was sent (to match the server's own log)
  url      where it went
  body     the request; a message content longer than 4096 characters is replaced by its length, its
           sha256 and its first 160 characters (bench.py's prefill and agentic prompts are rebuilt from
           its code and a nonce, and the nonce is in those characters)
  lines    [[seconds since the request was opened, the line], ...]: every line as received, SSE comments
           and blank lines included
  result   bench.py's post_stream result: ttft, total, t_first and t1 relative to its t0, usage, the
           content chunks it counted, text and reasoning

The texts let usage.completion_tokens be checked against a re-tokenisation and two servers' greedy texts
be compared; the lines give each content chunk's arrival and size (bench_record_report.py does both).
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

_real_urlopen = urllib.request.urlopen
_pending = []
_out = None


def _summary(body):
    b = json.loads(body)
    for m in b.get("messages", []):
        c = m.get("content")
        if isinstance(c, str) and len(c) > 4096:
            m["content"] = {"chars": len(c), "sha256": hashlib.sha256(c.encode()).hexdigest(), "head": c[:160]}
    return b


class _Tap:
    """The response, with each line's arrival time noted as bench.py reads it."""

    def __init__(self, resp, rec):
        self._resp = resp
        self._rec = rec

    def __enter__(self):
        self._resp.__enter__()
        return self

    def __exit__(self, *exc):
        return self._resp.__exit__(*exc)

    def __iter__(self):
        t_open, lines = self._rec["_t_open"], self._rec["lines"]
        for line in self._resp:
            lines.append((time.perf_counter() - t_open, line))
            yield line

    def __getattr__(self, name):
        return getattr(self._resp, name)


def _urlopen(req, *a, **kw):
    # only references and two clocks here: this runs inside bench.py's t0 -> t_first interval
    is_req = isinstance(req, urllib.request.Request)
    rec = {"t_wall": time.time(), "url": req.full_url if is_req else str(req),
           "_data": req.data if is_req else kw.get("data"), "lines": [], "_t_open": time.perf_counter()}
    resp = _real_urlopen(req, *a, **kw)
    _pending.append(rec)
    return _Tap(resp, rec)


_real_post_stream = bench.post_stream


def _post_stream(url, key, body, timeout=3600):
    r = _real_post_stream(url, key, body, timeout)
    # bench.py's t1 is taken: write the record now, outside its intervals
    rec = _pending.pop(0) if _pending else {"lines": [], "_t_open": None, "_data": None}
    _pending.clear()
    data = rec.pop("_data")
    rec["body"] = _summary(data) if data else None
    t0 = r["t0"]
    rec["lines"] = [[round(t, 6), ln.decode("utf-8", "replace")] for t, ln in rec["lines"]]
    rec["open_minus_t0"] = None if rec["_t_open"] is None else round(rec["_t_open"] - t0, 6)
    del rec["_t_open"]
    rec["result"] = {"ttft": r["ttft"], "total": r["total"],
                     "t_first": None if r["t_first"] is None else r["t_first"] - t0, "t1": r["t1"] - t0,
                     "usage": r["usage"], "chunks": r["chunks"], "text": r["text"], "reasoning": r["reasoning"]}
    _out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _out.flush()
    return r


def main():
    global _out
    argv = sys.argv[1:]
    if "--record" not in argv or argv.index("--record") + 1 >= len(argv):
        sys.exit("usage: bench_record.py --record FILE.jsonl <bench.py's arguments>")
    i = argv.index("--record")
    path = argv[i + 1]
    del argv[i:i + 2]
    _out = open(path, "a", encoding="utf-8")
    urllib.request.urlopen = _urlopen
    bench.post_stream = _post_stream
    sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")] + argv
    try:
        bench.main()
    finally:
        _out.close()


if __name__ == "__main__":
    main()
