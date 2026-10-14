#!/usr/bin/env python3
"""glw convert: cut the per-rank .glw files (every tensor but the routed experts, whose TOC entries are reserved)
from NVIDIA's checkpoint (the BF16 master for everything it holds in BF16) and zai's BF16 mirror (the dense MLPs
of layers 0-2, fetched by range request into --cache), on the CPU.

    python -I convert.py --ckpt NV --cache DIR --plan [--list R]       the plan: per-rank bytes by class, coverage
    python -I convert.py --ckpt NV --cache DIR --out-dir DIR [--ranks 0 1 2 3] [--only SUBSTR] [--commit C]

One pass over the sources writes every requested rank: each unit's sources are read once (SHA-256 while
reading), FP8 ones quantised once (quant.fp8_rtn, zai's rule) with their relative RMS error against the master,
and each rank's outputs re-cut by engine/attn/tools/pack.py. Files are written as OUT.tmp and renamed once the TOC
and header are in place; beside each, OUT.manifest.json (docs/layout.md § 5). Peak memory is one unit's sources
(the embedding and head, 3.8 GB) plus one rank's outputs. Run at nice 19 with CUDA_VISIBLE_DEVICES= (no GPU).
"""
import argparse, collections, hashlib, json, os, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import glw  # noqa: E402
import quant  # noqa: E402
import sources  # noqa: E402

GIB = 1 << 30


def load_sources(unit, ck, mirror, srcinfo):
    S = {}
    for s in unit.srcs:
        t, sha = (ck if s.origin == "nv" else mirror).get(s.name)
        info = {"origin": "nvidia" if s.origin == "nv" else "zai-org/GLM-5.3-BF16", "sha256": sha,
                "dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape), "fp8": s.fp8}
        if s.fp8:
            codes, scales = quant.fp8_rtn(t)
            info["rel_err"] = quant.dequant_rel_err(t, codes, scales)
            S[s.name] = (codes, scales)
        else:
            S[s.name] = t
        srcinfo[s.name] = info
    return S


def rel_err_of(o, srcinfo):
    errs = [srcinfo[n]["rel_err"] for n in srcinfo if srcinfo[n].get("rel_err") is not None and
            glw.short(n) in [x.replace("zai-bf16:", "") for x in o.sources]]
    return max(errs) if errs else 0.0


class RankFile:
    def __init__(self, path, rank, outs, toc_offset, data_offset, file_bytes):
        self.path, self.tmp, self.rank = path, path + ".tmp", rank
        self.outs, self.toc_offset, self.data_offset, self.file_bytes = outs, toc_offset, data_offset, file_bytes
        self.by_name = {o.name: o for o in outs}
        self.crc, self.sha, self.done = {}, {}, set()
        self.fd = os.open(self.tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.posix_fallocate(self.fd, 0, file_bytes)

    def write(self, name, arr, rel_err):
        o = self.by_name[name]
        want = {glw.F32: np.float32, glw.BF16: np.uint16, glw.F8: np.uint8, glw.U8: np.uint8, glw.U32: np.uint32}[o.dtype]
        if arr.dtype != want or arr.nbytes != o.bytes:
            raise SystemExit(f"{name}: built {arr.dtype} {arr.nbytes} B, planned {np.dtype(want)} {o.bytes} B")
        if o.dtype == glw.F8 and arr.size != int(np.prod(o.shape)):
            raise SystemExit(f"{name}: element count")
        b = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        os.pwrite(self.fd, memoryview(b), o.offset)
        pad = glw.align_up(o.bytes) - o.bytes
        if pad:
            os.pwrite(self.fd, bytes(pad), o.offset + o.bytes)
        self.crc[name] = glw.crc32c(b)
        self.sha[name] = hashlib.sha256(memoryview(b)).hexdigest()
        o.rel_err = rel_err
        self.done.add(name)

    def flush(self):
        os.fdatasync(self.fd)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def finish(self, ck_dir, tool, created):
        missing = [o.name for o in self.outs if o.kind != glw.RESERVED and o.name not in self.done]
        if missing:
            raise SystemExit(f"rank {self.rank}: {len(missing)} tensors not written, e.g. {missing[:3]}")
        toc = b"".join(o.entry(self.crc.get(o.name, 0)) for o in self.outs)
        toc_crc = glw.crc32c(toc)
        os.pwrite(self.fd, toc + bytes(self.data_offset - self.toc_offset - len(toc)), self.toc_offset)
        enc = lambda s, n: s.encode()[:n]  # noqa: E731
        hdr = glw.HDR.pack(glw.MAGIC, glw.VERSION, glw.HDR_BYTES, self.rank, glw.TP, len(self.outs), glw.ENT_BYTES,
                           self.toc_offset, self.data_offset, self.file_bytes - self.data_offset, self.file_bytes,
                           glw.ALIGN, toc_crc, enc("GLM-5.3", 64), enc(ck_dir, 256), enc(created, 32), enc(tool, 64))
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


def print_plan(units, r, ck, verbose):
    outs, toc_off, data_off, fb = glw.layout_rank(units, r)
    comp = collections.OrderedDict()
    lay = collections.Counter()
    res = 0
    for o in outs:
        c = comp.setdefault(o.component, [0, 0])
        c[0] += o.bytes if o.kind != glw.RESERVED else 0
        c[1] += 1
        lay[glw.LAYOUT_NAME[o.layout]] += o.bytes
        res += o.reserve_bytes
    print(f"rank {r}/{glw.TP}: {len(outs)} TOC entries, file {fb / GIB:.3f} GiB ({fb} B; TOC {data_off - toc_off} B), "
          f"routed experts reserved: {res / GIB:.2f} GiB at NVFP4 when filled")
    for c, (b, n) in comp.items():
        print(f"  {c:34s} {b / GIB:8.3f} GiB  {n:5d} entries")
    print("  by layout: " + ", ".join(f"{k} {v / GIB:.3f} GiB" for k, v in lay.items()))
    if verbose:
        for o in outs:
            print(f"    {o.name:42s} {glw.DTYPE_NAME[o.dtype]:7s} {'x'.join(map(str, o.shape)):>14s} "
                  f"{glw.LAYOUT_NAME[o.layout]:11s} {glw.KIND_NAME[o.kind]:10s} {o.bytes:>11d} "
                  f"{('blk ' + str(o.blk[0]) + 'x' + str(o.blk[1])) if o.blk[0] else '':12s} {o.slice}")
    return fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="NVIDIA's GLM-5.3-NVFP4 snapshot directory")
    ap.add_argument("--cache", required=True, help="fetch cache for zai's repos (fetch.py)")
    ap.add_argument("--out-dir")
    ap.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--list", action="store_true", help="with --plan: every tensor")
    ap.add_argument("--only", help="a test file: only tensors whose names contain this")
    ap.add_argument("--shared-layout", choices=["t64", "t128"], default="t128")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--commit", default="", help="the repo commit the tools were taken from, for the manifest")
    ap.add_argument("--suffix", default="", help="file name suffix (test files)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    glw._check_crc()
    t0 = time.time()
    ck = sources.Checkpoint(a.ckpt)
    units = glw.make_plan(ck, a.shared_layout)
    used, reserved, replaced = glw.coverage(ck, units)
    print(f"checkpoint: {len(ck.meta)} tensors; {len(used)} sources of {len(units)} units, {len(reserved)} routed-expert "
          f"tensors for the reserved entries, {len(replaced)} dense-MLP tensors replaced by the BF16 mirror's master "
          f"({time.time() - t0:.1f} s)")
    if a.plan:
        for r in a.ranks:
            print_plan(units, r, ck, a.list)
        return 0
    if not a.out_dir:
        raise SystemExit("--out-dir or --plan")
    os.makedirs(a.out_dir, exist_ok=True)
    mirror = sources.Fetched(a.cache, "bf16")
    tool = glw.tool_version()
    tool_s = tool[:16] + (f" {a.commit[:12]}" if a.commit else "")
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ck_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.normpath(ck.dir)))) + "@" + \
        os.path.basename(os.path.normpath(ck.dir))  # models--nvidia--GLM-5.3-NVFP4@<revision>
    files = {}
    for r in a.ranks:
        outs, toc_off, data_off, fb = glw.layout_rank(units, r)
        if a.only:
            outs = [o for o in outs if a.only in o.name]
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
        if not u.srcs:
            continue
        if a.only and not any(o.name in files[r].by_name for r in a.ranks for o in u.outs(r)):
            continue
        tu = time.time()
        S = load_sources(u, ck, mirror, srcinfo)
        for r in a.ranks:
            outs = {o.name: o for o in u.outs(r)}
            built = u.build(S, r)
            if set(built) != set(outs):
                raise SystemExit(f"{u.name}: built {sorted(set(built) ^ set(outs))[:4]} not as planned")
            for n, arr in built.items():
                if n in files[r].by_name:
                    files[r].write(n, arr, rel_err_of(outs[n], srcinfo) if outs[n].flags & glw.LOSSY else 0.0)
                    written += arr.nbytes
            files[r].flush()
        del S
        print(f"  {u.name}: {time.time() - tu:.1f} s; {written / GIB:.2f} GiB written, {time.time() - t1:.0f} s",
              flush=True)
    for r, f in files.items():
        toc_crc = f.finish(ck_id, tool_s, created)
        sha = file_sha256(f.path)
        man = {
            "format": "glw", "version": glw.VERSION, "rank": r, "world": glw.TP, "file": os.path.basename(f.path),
            "file_bytes": f.file_bytes, "file_sha256": sha, "toc_crc32c": f"{toc_crc:08x}", "created": created,
            "seconds": round(time.time() - t1, 1),
            "tool": {"version": tool, "commit": a.commit, "argv": [os.path.basename(x) for x in sys.argv[:1]] + sys.argv[1:],
                     "files": {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in glw.tool_files()},
                     "torch": torch.__version__, "numpy": np.__version__},
            "master": {"nvidia": ck_id, "bf16_mirror": "zai-org/GLM-5.3-BF16@9d2398f478cab2de883137db3a36ad2c96205e24",
                       "fp8_rule": "per 128x128 block: scale = f32(amax) * f32(1/448); codes = e4m3_rne(f32(w) / scale)"},
            "only": a.only or "", "shared_layout": a.shared_layout,
            "sources": {n: v for n, v in srcinfo.items()},
            "tensors": [{"name": o.name, "offset": o.offset, "bytes": o.bytes, "crc32c": f"{f.crc.get(o.name, 0):08x}",
                         "sha256": f.sha.get(o.name, ""), "rel_err": o.rel_err, "kind": glw.KIND_NAME[o.kind],
                         "layout": glw.LAYOUT_NAME[o.layout], "sources": o.sources} for o in f.outs],
            "routed_expert_tensors_reserved": len(reserved), "dense_tensors_replaced_by_master": sorted(replaced),
        }
        json.dump(man, open(f.path + ".manifest.json", "w"), indent=1)
        print(f"rank {r}: {f.path} {f.file_bytes} B ({f.file_bytes / GIB:.3f} GiB), sha256 {sha}, TOC crc {toc_crc:08x}")
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
