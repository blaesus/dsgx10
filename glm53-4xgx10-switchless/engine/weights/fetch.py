#!/usr/bin/env python3
"""Fetch single tensors from a Hugging Face safetensors repo by HTTP range requests (the shard's 8-byte length
and JSON header, then the tensor's bytes; nothing else is downloaded), as tools/provenance.py does.

    python -I fetch.py OUT --repo fp8|bf16 NAME [NAME ...]          (also usable as a module: fetch(...))

OUT/<repo>/<name> holds the raw bytes (written to .part, renamed when complete, so a rerun resumes);
OUT/<repo>/manifest.jsonl one line per tensor: name, shard, dtype, shape, bytes, SHA-256 of the bytes.
The repos are pinned to the revisions docs/quantisation.md § 3.0 and § 4 name. HF_TOKEN is used when set.
The bytes are data only: nothing fetched is executed; headers are parsed as JSON.
"""
import argparse, hashlib, json, os, struct, sys, threading, time, urllib.request

REPOS = {
    "fp8": ("zai-org/GLM-5.3", "c5087a76d148760811ffbdedfe6ff9f194e46e38"),
    "bf16": ("zai-org/GLM-5.3-BF16", "9d2398f478cab2de883137db3a36ad2c96205e24"),
}
_lock = threading.Lock()


def _req(base, path, a=None, b=None, tries=8):
    h = {}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    if a is not None:
        h["Range"] = f"bytes={a}-{b}"
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(base + path, headers=h), timeout=600) as f:
                data = f.read()
            if a is not None and len(data) != b - a + 1:
                raise IOError(f"short read {len(data)} of {b - a + 1}")
            return data
        except Exception as e:  # network: retry with backoff
            if t == tries - 1:
                raise
            print(f"retry {path} {a}: {e}", flush=True)
            time.sleep(min(60, 2 ** t))


class Repo:
    """One pinned repo's tensors under OUT/<repo id with _>."""

    def __init__(self, out, key):
        self.repo, self.rev = REPOS[key]
        self.key = key
        self.base = f"https://huggingface.co/{self.repo}/resolve/{self.rev}/"
        self.dir = os.path.join(os.path.expanduser(out), self.repo.replace("/", "_"))
        os.makedirs(os.path.join(self.dir, "headers"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "tensors"), exist_ok=True)
        ip = os.path.join(self.dir, "model.safetensors.index.json")
        if not os.path.exists(ip):
            open(ip + ".part", "wb").write(_req(self.base, "model.safetensors.index.json"))
            os.rename(ip + ".part", ip)
        self.wm = json.load(open(ip))["weight_map"]
        self.mp = os.path.join(self.dir, "manifest.jsonl")
        self.man = {}
        if os.path.exists(self.mp):
            for line in open(self.mp):
                r = json.loads(line)
                self.man[r["name"]] = r

    def has(self, name):
        return name in self.wm

    def header(self, shard):
        hp = os.path.join(self.dir, "headers", shard + ".json")
        if not os.path.exists(hp):
            n = struct.unpack("<Q", _req(self.base, shard, 0, 7))[0]
            json.dump({"n": n, "h": json.loads(_req(self.base, shard, 8, 8 + n - 1))}, open(hp + ".part", "w"))
            os.rename(hp + ".part", hp)
        return json.load(open(hp))

    def fetch(self, name):
        """The tensor's record (name, shard, dtype, shape, bytes, sha256, path); fetches it if missing."""
        if name in self.man and os.path.exists(self.man[name]["path"]):
            return self.man[name]
        shard = self.wm[name]
        hh = self.header(shard)
        v = hh["h"][name]
        a, b = v["data_offsets"]
        tp = os.path.join(self.dir, "tensors", name)
        if not os.path.exists(tp) or os.path.getsize(tp) != b - a:
            data = _req(self.base, shard, 8 + hh["n"] + a, 8 + hh["n"] + b - 1)
            open(tp + ".part", "wb").write(data)
            os.rename(tp + ".part", tp)
        else:
            data = open(tp, "rb").read()
        rec = {"name": name, "repo": self.repo, "rev": self.rev, "shard": shard, "dtype": v["dtype"],
               "shape": v["shape"], "bytes": b - a, "sha256": hashlib.sha256(data).hexdigest(), "path": tp}
        with _lock:
            open(self.mp, "a").write(json.dumps(rec) + "\n")
            self.man[name] = rec
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--repo", choices=sorted(REPOS), required=True)
    ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    r = Repo(a.out, a.repo)
    for n in a.names:
        rec = r.fetch(n)
        print(f"{n}: {rec['dtype']} {rec['shape']} {rec['bytes']} B sha256 {rec['sha256'][:16]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
