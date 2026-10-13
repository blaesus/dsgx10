"""The converter's inputs: NVIDIA's checkpoint (local safetensors shards; its BF16 tensors are the BF16 master,
docs/quantisation.md § 3.0) and tensors fetched by range request from zai's repos (fetch.py), each read with its
SHA-256 computed over the bytes as read. Reads drop their pages from the page cache afterwards (the box serves
production; nothing here should grow the cache by the ~35 GB a conversion reads)."""
import hashlib, json, os, struct

import numpy as np
import torch

import fetch

DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8}


def _read(path, off, n):
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = bytearray(n)
        mv, got = memoryview(buf), 0
        while got < n:
            r = os.preadv(fd, [mv[got:]], off + got)
            if r <= 0:
                raise IOError(f"short read {path} at {off + got}")
            got += r
        os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
        return buf
    finally:
        os.close(fd)


class Checkpoint:
    """A local safetensors checkpoint: name -> (shard, dtype, shape, offset)."""

    def __init__(self, d):
        self.dir = os.path.expanduser(d)
        self.wm = json.load(open(os.path.join(self.dir, "model.safetensors.index.json")))["weight_map"]
        self.meta = {}
        for shard in sorted(set(self.wm.values())):
            p = os.path.join(self.dir, shard)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            for k, v in h.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.meta[k] = (p, v["dtype"], tuple(v["shape"]), 8 + n + a, b - a)
        missing = set(self.wm) - set(self.meta)
        if missing:
            raise SystemExit(f"{len(missing)} indexed tensors not in the shards, e.g. {sorted(missing)[:3]}")

    def __contains__(self, name):
        return name in self.meta

    def names(self):
        return self.meta.keys()

    def shape(self, name):
        return self.meta[name][2]

    def get(self, name):
        """(torch tensor, sha256 hex of its bytes)."""
        p, dt, shape, off, n = self.meta[name]
        buf = _read(p, off, n)
        sha = hashlib.sha256(buf).hexdigest()
        t = torch.frombuffer(buf, dtype=DT[dt]).reshape(shape) if n else torch.empty(shape, dtype=DT[dt])
        return t, sha


class Fetched:
    """Tensors from a pinned Hugging Face repo, fetched on first use into a local cache directory."""

    def __init__(self, cache, key):
        self.repo = fetch.Repo(cache, key)

    def get(self, name):
        rec = self.repo.fetch(name)
        buf = bytearray(open(rec["path"], "rb").read())
        sha = hashlib.sha256(buf).hexdigest()
        if sha != rec["sha256"]:
            raise SystemExit(f"{rec['path']}: SHA-256 {sha} differs from the fetch record {rec['sha256']}")
        return torch.frombuffer(buf, dtype=DT[rec["dtype"]]).reshape(rec["shape"]), sha

    def record(self, name):
        return self.repo.fetch(name)
