#!/usr/bin/env python3
"""glw convert: cut altar-1's per-rank .glw files (every tensor but the routed experts, whose TOC entries are
reserved for experts_fill.py) from the local checkpoint, on the CPU, without requantising anything.

    python -I convert.py --ckpt DIR [--index J] --plan [--list] [--ranks R ...]     the plan, bytes a rank, coverage
    python -I convert.py --ckpt DIR --manifest manifest.tsv --hashes shards.sha256 --out-dir OUT
                         [--ranks 0 1 2 3] [--only SUBSTR ...] [--bf16-only] [--commit C] [--suffix S]

--manifest/--hashes: every shard read must have the Hub's LFS SHA-256 (manifest.tsv), as measured on this box
(`sha256sum` output); without them the run is a test (--unverified-ok) and the manifest says so. --bf16-only
leaves out the INT4 attention tiles (q_b, W_UV, o_proj of layers 4-76) when their packer has not landed; W_UK^T of
those layers is written in BF16 by the contract's dequantisation. --test-packers uses packers.py's test stand-ins
(never for rank files).

One pass over the sources writes every requested rank: each unit's sources are read once (SHA-256 while reading;
the embedding and head by the rank's rows only) and each rank's outputs cut from them. Files are written as
OUT.tmp and renamed once the TOC and header are in place; beside each, OUT.manifest.json (docs/layout.md § 5).
Peak memory is one unit's sources (the largest a dense MLP's three BF16 matrices, 453 MB) plus one rank's outputs.
"""
import argparse, collections, hashlib, json, os, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

import glw  # noqa: E402
import packers  # noqa: E402
import sources  # noqa: E402

GIB = 1 << 30
REVISION = "AikidoSec/altar-1@5d591cd98958e4e7e429517887e80ca3bb7ce2c4"


class RankFile:
    def __init__(self, path, rank, outs, toc_offset, data_offset, file_bytes):
        self.path, self.tmp, self.rank = path, path + ".tmp", rank
        self.outs, self.toc_offset, self.data_offset, self.file_bytes = outs, toc_offset, data_offset, file_bytes
        self.by_name = {o.name: o for o in outs}
        self.crc, self.sha, self.done = {}, {}, set()
        self.fd = os.open(self.tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.posix_fallocate(self.fd, 0, file_bytes)

    def write(self, name, arr):
        o = self.by_name[name]
        want = glw.NP_DTYPE[o.dtype]
        if arr.dtype != want or arr.nbytes != o.bytes:
            raise SystemExit(f"{name}: built {arr.dtype} {arr.shape} {arr.nbytes} B, planned {np.dtype(want)} "
                             f"{o.shape} {o.bytes} B")
        if arr.size != int(np.prod(o.shape)):
            raise SystemExit(f"{name}: element count {arr.size}, planned {o.shape}")
        b = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        os.pwrite(self.fd, memoryview(b), o.offset)
        pad = glw.align_up(o.bytes) - o.bytes
        if pad:
            os.pwrite(self.fd, bytes(pad), o.offset + o.bytes)
        self.crc[name] = glw.crc32c(b)
        self.sha[name] = hashlib.sha256(memoryview(b)).hexdigest()
        self.done.add(name)

    def flush(self):
        os.fdatasync(self.fd)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def finish(self, ck_id, tool, created):
        missing = [o.name for o in self.outs if o.kind != glw.RESERVED and o.name not in self.done]
        if missing:
            raise SystemExit(f"rank {self.rank}: {len(missing)} tensors not written, e.g. {missing[:3]}")
        toc = b"".join(o.entry(self.crc.get(o.name, 0)) for o in self.outs)
        toc_crc = glw.crc32c(toc)
        os.pwrite(self.fd, toc + bytes(self.data_offset - self.toc_offset - len(toc)), self.toc_offset)
        enc = lambda s, n: s.encode()[:n]  # noqa: E731
        hdr = glw.HDR.pack(glw.MAGIC, glw.VERSION, glw.HDR_BYTES, self.rank, glw.TP, len(self.outs), glw.ENT_BYTES,
                           self.toc_offset, self.data_offset, self.file_bytes - self.data_offset, self.file_bytes,
                           glw.ALIGN, toc_crc, enc("altar-1", 64), enc(ck_id, 256), enc(created, 32), enc(tool, 64))
        os.pwrite(self.fd, hdr + bytes(glw.HDR_BYTES - len(hdr)), 0)
        os.fsync(self.fd)
        os.close(self.fd)
        os.rename(self.tmp, self.path)
        return toc_crc


def file_sha256(path):
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        off = 0
        while True:
            b = os.pread(fd, 64 << 20, off)
            if not b:
                break
            h.update(b)
            os.posix_fadvise(fd, off, len(b), os.POSIX_FADV_DONTNEED)
            off += len(b)
    finally:
        os.close(fd)
    return h.hexdigest()


def print_plan(units, r, verbose):
    outs, toc_off, data_off, fb = glw.layout_rank(units, r)
    comp = collections.OrderedDict()
    lay = collections.Counter()
    res = 0
    for o in outs:
        c = comp.setdefault(o.component, [0, 0, 0])
        c[0] += o.bytes if o.kind != glw.RESERVED else 0
        c[1] += 1
        c[2] += o.reserve_bytes
        lay[glw.LAYOUT_NAME[o.layout]] += o.bytes
        res += o.reserve_bytes
    print(f"rank {r}/{glw.TP}: {len(outs)} TOC entries, file {fb / GIB:.3f} GiB ({fb} B; TOC {data_off - toc_off} B) "
          f"before the fill, routed experts reserved: {res / GIB:.3f} GiB, {(fb + res) / GIB:.3f} GiB filled")
    for c, (b, n, rb) in comp.items():
        print(f"  {c:40s} {b / GIB:8.3f} GiB  {n:5d} entries" + (f"  ({rb / GIB:.3f} GiB to fill)" if rb else ""))
    print("  by layout: " + ", ".join(f"{k} {v / GIB:.3f} GiB" for k, v in lay.items()))
    if verbose:
        for o in outs:
            print(f"    {o.name:42s} {glw.DTYPE_NAME[o.dtype]:5s} {'x'.join(map(str, o.shape)):>14s} "
                  f"{glw.LAYOUT_NAME[o.layout]:11s} {glw.KIND_NAME[o.kind]:10s} {o.bytes or o.reserve_bytes:>11d} "
                  f"{o.slice}")
    return fb


def open_checkpoint(a):
    if not a.plan and not (a.manifest and a.hashes) and not a.unverified_ok:
        raise SystemExit("--manifest and --hashes (every shard checked against the Hub's SHA-256), or --unverified-ok "
                         "for a test")
    ck = sources.Checkpoint(a.ckpt, a.index, a.manifest, a.hashes)
    st = collections.Counter(s["state"] for s in ck.shards.values())
    print(f"checkpoint {ck.dir}: {len(ck.meta)} tensors in {len(ck.shards)} shards ({dict(st)})", flush=True)
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="altar-1's snapshot directory (the shards)")
    ap.add_argument("--index", help="model.safetensors.index.json (default: in --ckpt)")
    ap.add_argument("--manifest", help="manifest.tsv: the Hub's LFS SHA-256 and size of every file")
    ap.add_argument("--hashes", help="sha256sum output over the shards, measured on this box")
    ap.add_argument("--unverified-ok", action="store_true", help="a test run on shards not checked by SHA-256")
    ap.add_argument("--out-dir")
    ap.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--list", action="store_true", help="with --plan: every tensor")
    ap.add_argument("--only", nargs="+", help="a test file: only tensors whose names contain one of these")
    ap.add_argument("--bf16-only", action="store_true", help="leave out the INT4 attention tiles (packer pending)")
    ap.add_argument("--test-packers", action="store_true", help="packers.py's test stand-ins (never for rank files)")
    ap.add_argument("--expert-packer", help="the expert kernels' int4_pack.py (default: engine/weights/int4_pack.py)")
    ap.add_argument("--attn-packer", help="the attention side's pack.py with int4_tiles (default: attn/tools/pack.py)")
    ap.add_argument("--commit", default="", help="the repo commit the tools were taken from, for the manifest")
    ap.add_argument("--suffix", default="", help="file name suffix (test files)")
    a = ap.parse_args()
    glw._check_crc()
    t0 = time.time()
    reg = packers.register(test=a.test_packers, experts=a.expert_packer, attn=a.attn_packer)
    print("packers: " + ", ".join(f"{k} {v or 'pending'}" for k, v in reg.items()), flush=True)
    ck = open_checkpoint(a)
    units = glw.make_plan(ck, a.bf16_only)
    used, reserved, pending = glw.coverage(ck, units, a.bf16_only)
    print(f"coverage: {len(used)} sources of {len(units)} units, {len(reserved)} routed-expert tensors for the reserved "
          f"entries, {len(pending)} INT4 attention tensors pending (--bf16-only) ({time.time() - t0:.1f} s)")
    if a.plan:
        for r in a.ranks:
            print_plan(units, r, a.list)
        return 0
    if not a.out_dir:
        raise SystemExit("--out-dir or --plan")
    os.makedirs(a.out_dir, exist_ok=True)
    tool = glw.tool_version()
    tool_s = tool[:16] + (f" {a.commit[:12]}" if a.commit else "")
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    files = {}
    for r in a.ranks:
        outs, toc_off, data_off, fb = glw.layout_rank(units, r)
        if a.only:
            outs = [o for o in outs if any(x in o.name for x in a.only)]
            data_off = glw.align_up(toc_off + len(outs) * glw.ENT_BYTES)
            off = data_off
            for o in outs:
                if o.kind != glw.RESERVED:
                    o.offset, off = off, off + glw.align_up(o.bytes)
            fb = off
        path = os.path.join(a.out_dir, f"rank{r}{a.suffix}.glw")
        files[r] = RankFile(path, r, outs, toc_off, data_off, fb)
        print(f"rank {r}: {path}, {len(outs)} entries, {fb / GIB:.3f} GiB", flush=True)
    srcinfo = {}
    t1 = time.time()
    written = 0
    for u in units:
        if all(o.kind == glw.RESERVED for o in u.outs(a.ranks[0])):
            continue
        if a.only and not any(o.name in files[r].by_name for r in a.ranks for o in u.outs(r)):
            continue
        tu = time.time()
        R = glw.Reader(ck, srcinfo)
        for r in a.ranks:
            outs = {o.name: o for o in u.outs(r)}
            built = u.build(R, r)
            if set(built) != set(outs):
                raise SystemExit(f"{u.name}: built {sorted(set(built) ^ set(outs))[:4]} not as planned")
            for n, arr in built.items():
                if n in files[r].by_name:
                    files[r].write(n, arr)
                    written += arr.nbytes
            files[r].flush()
        del R
        print(f"  {u.name}: {time.time() - tu:.1f} s; {written / GIB:.2f} GiB written, {time.time() - t1:.0f} s",
              flush=True)
    for r, f in files.items():
        toc_crc = f.finish(REVISION, tool_s, created)
        sha = file_sha256(f.path)
        man = {
            "format": "glw (altar-1)", "magic": glw.MAGIC.decode(), "version": glw.VERSION, "rank": r, "world": glw.TP,
            "file": os.path.basename(f.path), "file_bytes": f.file_bytes, "file_sha256": sha,
            "toc_crc32c": f"{toc_crc:08x}", "created": created, "seconds": round(time.time() - t1, 1),
            "tool": {"version": tool, "commit": a.commit, "argv": [os.path.basename(x) for x in sys.argv[:1]] + sys.argv[1:],
                     "files": {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in glw.tool_files()},
                     "numpy": np.__version__},
            "checkpoint": dict(ck.record(), revision=REVISION, verified=bool(a.manifest and a.hashes)),
            "packers": reg, "bf16_only": a.bf16_only, "only": a.only or [],
            "rule": "no requantisation: BF16/F32 copied; INT4 packed values, scales, zero points re-tiled; an INT4 "
                    "layer's W_UK^T in BF16 is bf16_rne((q - z) * s) (docs/model.md § 4.2)",
            "sources": srcinfo,
            "tensors": [{"name": o.name, "offset": o.offset, "bytes": o.bytes, "crc32c": f"{f.crc.get(o.name, 0):08x}",
                         "sha256": f.sha.get(o.name, ""), "kind": glw.KIND_NAME[o.kind], "flags": o.flags,
                         "layout": glw.LAYOUT_NAME[o.layout], "sources": o.sources} for o in f.outs],
            "routed_expert_tensors_reserved": len(reserved), "int4_attention_pending": len(pending),
        }
        json.dump(man, open(f.path + ".manifest.json", "w"), indent=1)
        print(f"rank {r}: {f.path} {f.file_bytes} B ({f.file_bytes / GIB:.3f} GiB), sha256 {sha}, TOC crc {toc_crc:08x}")
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
