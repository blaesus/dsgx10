"""Compare rows of a checkpoint tensor with the same rows on the Hugging Face Hub, by HTTP range
requests only: the 8-byte header length, the JSON header, then each row's bytes. Never fetches more
than it asked for (every response must be a 206 of the requested length).

  python3 tools/hf_rows.py --local /data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8 \
      --tensor layers.1.engram.embed.weight --tensor layers.1.engram.embed.scale --rows 5,77,123456

`--rows-file f.txt` reads more row ids from a text file (whitespace or comma separated).
"""
import argparse
import http.client
import json
import math
import os
import struct
import sys
import urllib.parse
import urllib.request

ELEM = {"F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "U8": 1, "I8": 1}


class RangeReader:
    """Range GETs on one Hub file over a kept-alive HTTPS connection to wherever it redirects."""

    def __init__(self, repo, fname, rev="main"):
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{fname}"
        req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
        with urllib.request.urlopen(req, timeout=60) as r:  # follows the redirect to the CDN
            assert r.status == 206, r.status
            self.head8 = r.read(8)
            self.url = r.geturl()
        u = urllib.parse.urlsplit(self.url)
        self.host, self.path = u.netloc, u.path + ("?" + u.query if u.query else "")
        self.conn = None
        self.fetched = 8

    def get(self, off, n):
        for attempt in range(3):
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPSConnection(self.host, timeout=60)
                self.conn.request("GET", self.path, headers={"Range": f"bytes={off}-{off + n - 1}"})
                r = self.conn.getresponse()
                if r.status != 206 or int(r.getheader("Content-Length", -1)) != n:
                    self.conn.close()  # never read a whole-file 200 body
                    raise RuntimeError(f"bad range response {r.status} {r.getheader('Content-Length')}")
                b = r.read()
                self.fetched += len(b)
                return b
            except (http.client.HTTPException, OSError):
                self.conn = None
                if attempt == 2:
                    raise


def local_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--rev", default="main")
    ap.add_argument("--local", required=True, help="local checkpoint directory")
    ap.add_argument("--tensor", action="append", required=True)
    ap.add_argument("--rows", default="", help="comma-separated row ids")
    ap.add_argument("--rows-file", default="")
    ap.add_argument("--max-bytes", type=int, default=8 << 20, help="refuse to fetch more than this")
    a = ap.parse_args()
    rows = [int(v) for v in a.rows.split(",") if v]
    if a.rows_file:
        rows += [int(v) for v in open(a.rows_file).read().replace(",", " ").split()]
    rows = sorted(set(rows))
    wmap = json.load(open(os.path.join(a.local, "model.safetensors.index.json")))["weight_map"]
    readers = {}
    for t in a.tensor:
        fname = wmap[t]
        lpath = os.path.join(a.local, fname)
        ln, lhdr = local_header(lpath)
        if fname not in readers:
            rr = readers[fname] = RangeReader(a.repo, fname, a.rev)
            rn = struct.unpack("<Q", rr.head8)[0]
            rr.hdr_n, rr.hdr = rn, json.loads(rr.get(8, rn))
        rr = readers[fname]
        e, le = rr.hdr[t], lhdr[t]
        row_bytes = ELEM[e["dtype"]] * math.prod(e["shape"][1:])
        same_meta = (e["dtype"], e["shape"]) == (le["dtype"], le["shape"])
        print(f"{t}: {fname}, hub {e['dtype']} {e['shape']} offset {e['data_offsets'][0]}, local "
              f"{le['dtype']} {le['shape']} offset {le['data_offsets'][0]}; header same: {same_meta and rr.hdr_n == ln}")
        if len(rows) * row_bytes + sum(r.fetched for r in readers.values()) > a.max_bytes:
            sys.exit(f"would fetch more than --max-bytes {a.max_bytes}")
        hub_base, loc_base = 8 + rr.hdr_n + e["data_offsets"][0], 8 + ln + le["data_offsets"][0]
        diffs = 0
        fd = os.open(lpath, os.O_RDONLY)
        for row in rows:
            hb = rr.get(hub_base + row * row_bytes, row_bytes)
            lb = os.pread(fd, row_bytes, loc_base + row * row_bytes)
            if hb != lb:
                diffs += 1
                if diffs <= 5:
                    print(f"  row {row} differs: hub {hb[:16].hex()} local {lb[:16].hex()}")
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        print(f"  {len(rows)} rows of {row_bytes} B compared: {len(rows) - diffs} identical, {diffs} differ")
    print("fetched from the Hub:", sum(r.fetched for r in readers.values()), "bytes")


if __name__ == "__main__":
    main()
