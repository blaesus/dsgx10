"""The converter's input: AikidoSec/altar-1's local safetensors shards (revision 5d591cd9...), read with plain
preadv into numpy (no torch): BF16 as uint16 bit patterns, F32, I32 (the INT4 parts), I64 (`weight_shape`). Every
read computes the SHA-256 of the bytes as read and drops its pages from the page cache (the box serves production;
nothing here should grow the cache by the ~305 GiB a conversion reads).

A shard is used only once its whole-file SHA-256 equals the Hub's LFS SHA-256 in the manifest (`--manifest`
manifest.tsv: name, sha256, bytes; `--hashes` a `sha256sum` output over the shards, made on the box): the
checkpoint refuses to read any tensor of an unverified shard. For --plan, a shard may be absent if its header
prefix is present as NAME.hdr (tools/tally.py's header files): its tensors are then known but cannot be read.
The shards, index and manifest are untrusted data: parsed as JSON and bytes, never executed (python -I)."""
import hashlib, json, os, struct

import numpy as np

# safetensors dtype -> numpy (BF16 kept as its bit pattern)
DT = {"BF16": np.uint16, "F32": np.float32, "I32": np.int32, "I64": np.int64, "U8": np.uint8}


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


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 64 << 20:
            raise SystemExit(f"{path}: header of {n} B")
        raw = f.read(n)
    if len(raw) != n:
        raise SystemExit(f"{path}: short header")
    return n, json.loads(raw)


def read_manifest(path):
    """manifest.tsv: name \\t sha256 \\t bytes (sha256 empty for non-LFS files) -> {name: (sha256, bytes)}."""
    out = {}
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3 and f[1]:
            out[f[0]] = (f[1], int(f[2]))
    return out


def read_hashes(path, man=None):
    """`sha256sum` output -> {basename: sha256}; or a verify log's "OK <file>" lines (a box's own sha256sum compared
    with the manifest by the fetch scripts), which stand for the manifest's SHA-256 ("BAD" lines for none)."""
    out = {}
    for line in open(path):
        f = line.split()
        if len(f) == 2 and len(f[0]) == 64:
            out[os.path.basename(f[1].lstrip("*"))] = f[0]
        elif len(f) == 2 and f[0] == "OK" and man and os.path.basename(f[1]) in man:
            out[os.path.basename(f[1])] = man[os.path.basename(f[1])][0]
    return out


class Checkpoint:
    """name -> (shard path, dtype, shape, offset, bytes). With `manifest` and `hashes`, every shard read must have
    its measured SHA-256 equal to the manifest's (and its size); `self.shards` records each shard's state."""

    def __init__(self, d, index=None, manifest=None, hashes=None):
        self.dir = os.path.expanduser(d)
        ip = index or os.path.join(self.dir, "model.safetensors.index.json")
        self.index_sha = hashlib.sha256(open(ip, "rb").read()).hexdigest()
        self.wm = json.load(open(ip))["weight_map"]
        man = read_manifest(manifest) if manifest else None
        got = read_hashes(hashes, man) if hashes else {}
        if man and "model.safetensors.index.json" in man and man["model.safetensors.index.json"][0] != self.index_sha:
            raise SystemExit(f"{ip}: SHA-256 {self.index_sha} differs from the manifest's")
        self.meta, self.shards = {}, {}
        for shard in sorted(set(self.wm.values())):
            p = os.path.join(self.dir, shard)
            readable = os.path.exists(p)
            hp = p if readable else (p + ".hdr" if os.path.exists(p + ".hdr") else None)
            if hp is None:
                raise SystemExit(f"{shard}: neither the shard nor its header prefix in {self.dir}")
            n, h = _header(hp)
            state = "header only"
            if readable:
                if man is None:
                    state = "unverified (no manifest given)"
                else:
                    want = man.get(shard)
                    if want is None:
                        raise SystemExit(f"{shard}: not in the manifest")
                    if os.path.getsize(p) != want[1]:
                        state = f"size {os.path.getsize(p)} != {want[1]}"
                        readable = False
                    elif got.get(shard) != want[0]:
                        state = "SHA-256 not measured" if shard not in got else f"SHA-256 {got[shard]} != {want[0]}"
                        readable = False
                    else:
                        state = "verified"
            self.shards[shard] = {"state": state, "readable": readable,
                                  "sha256": man[shard][0] if man and shard in man else None}
            for k, v in h.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.meta[k] = (p, v["dtype"], tuple(v["shape"]), 8 + n + a, b - a, shard)
        missing = set(self.wm) - set(self.meta)
        if missing:
            raise SystemExit(f"{len(missing)} indexed tensors not in the shards, e.g. {sorted(missing)[:3]}")
        self.require_verified = man is not None

    def __contains__(self, name):
        return name in self.meta

    def names(self):
        return self.meta.keys()

    def shape(self, name):
        return self.meta[name][2]

    def dtype(self, name):
        return self.meta[name][1]

    def nbytes(self, name):
        return self.meta[name][4]

    def shard_of(self, name):
        return self.meta[name][5]

    def _check(self, name):
        sh = self.shards[self.meta[name][5]]
        if not sh["readable"]:
            raise SystemExit(f"{name}: shard {self.meta[name][5]} is not usable ({sh['state']})")

    def get(self, name):
        """(numpy array in its natural shape, sha256 hex of its bytes)."""
        self._check(name)
        p, dt, shape, off, n, _ = self.meta[name]
        buf = _read(p, off, n)
        sha = hashlib.sha256(buf).hexdigest()
        return np.frombuffer(buf, dtype=DT[dt]).reshape(shape), sha

    def get_rows(self, name, a, b):
        """Rows [a, b) of a 2-D tensor (one contiguous read): (array, sha256 of those bytes)."""
        self._check(name)
        p, dt, shape, off, n, _ = self.meta[name]
        row = n // shape[0]
        buf = _read(p, off + a * row, (b - a) * row)
        return np.frombuffer(buf, dtype=DT[dt]).reshape((b - a,) + tuple(shape[1:])), hashlib.sha256(buf).hexdigest()

    def usable(self):
        return all(s["readable"] for s in self.shards.values())

    def record(self):
        """The provenance block for a manifest: index SHA-256, every shard's name, LFS SHA-256 and state."""
        return {"index_sha256": self.index_sha,
                "shards": {k: {"sha256": v["sha256"], "state": v["state"]} for k, v in sorted(self.shards.items())}}
