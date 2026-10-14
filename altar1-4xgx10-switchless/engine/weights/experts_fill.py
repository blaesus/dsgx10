#!/usr/bin/env python3
"""Fill a rank file's reserved routed-expert entries with altar-1's experts, as shipped: INT4-g32 on layers 4-76
(packed values, BF16 scales, zero points re-tiled), BF16 on layers 3, 77 and the MTP layer 78. Nothing is
requantised.

    python -I experts_fill.py --ckpt DIR --manifest M --hashes H --file rankR.glw [...] --layers 3 [4 ...]
                              [--reset-from L] [--test-packers]

A layer's TOC entries are its desc table (gx::MatDesc [168][3]) and one entry an expert (gate, up, down matrices
back to back, format tag gate | up << 4 | down << 8, glw.h ExpertFmt); the 168 expert entries are contiguous and
form the layer's pool, MatDesc::off relative to expert 0's entry. Rank r's slice of each matrix is cut on group
boundaries (gate/up: rows [512r, +512), zero-point rows [64r, +64); down: input columns [512r, +512), packed words
[64r, +64), scale and zero-point groups [16r, +16); BF16 the same rows / columns) and packed by the expert kernels'
packer for the layer's format (glw.EXPERT_FORMATS, registered by packers.py from engine/experts/layout.h's port);
every matrix written is unpacked again by the format's own inverse and compared with its source slice before the
layer is committed.

Filling follows the TOC order (layer 3 before 4 ...), which keeps the data region in TOC order; --reset-from L
returns layers >= L to reserved and cuts the files there. Each layer is committed (data synced, then TOC, header,
manifest) before the next, so a rerun with the same arguments resumes after the last whole layer. Several --file
are filled in one pass over the sources (the disk budget on the converting box allows two rank files at a time).
"""
import argparse, hashlib, json, os, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

import glw  # noqa: E402
import int4  # noqa: E402
import packers  # noqa: E402
import sources  # noqa: E402

E = glw.N_EXPERTS
KI = glw.EXPERT_INTER // glw.TP  # 512
PROJ = ("gate", "up", "down")


def expert_sources(ck, L, e, fmt, srcs):
    """The three full matrices of expert e (int4 dicts, or BF16 uint16 arrays), their reads' SHA-256 in `srcs`."""
    out = []
    for proj in PROJ:
        base = f"model.layers.{L}.mlp.experts.{e}.{proj}_proj"
        if fmt == glw.FMT_INT4:
            t, shas = int4.parts(ck, base)
            want = (glw.EXPERT_INTER, glw.D) if proj != "down" else (glw.D, glw.EXPERT_INTER)
            if int4.shape(t) != want:
                raise SystemExit(f"{base}: shape {int4.shape(t)}")
            srcs.update(shas)
        else:
            t, sha = ck.get(base + ".weight")
            want = (glw.EXPERT_INTER, glw.D) if proj != "down" else (glw.D, glw.EXPERT_INTER)
            if ck.dtype(base + ".weight") != "BF16" or t.shape != want:
                raise SystemExit(f"{base}.weight: {ck.dtype(base + '.weight')} {t.shape}")
            srcs[base + ".weight"] = sha
        out.append(t)
    return out


def rank_slice(fmt, t, m, r):
    a, b = KI * r, KI * (r + 1)
    if fmt == glw.FMT_INT4:
        return int4.cols(t, a, b) if m == 2 else int4.rows(t, a, b)
    return np.ascontiguousarray(t[:, a:b] if m == 2 else t[a:b])


def same(fmt, x, y):
    return int4.equal(x, y) if fmt == glw.FMT_INT4 else (x.shape == y.shape and np.array_equal(x, y))


class Target:
    def __init__(self, path):
        self.path = path
        self.hd, self.ents, _ = glw.read_toc(path)
        self.r = self.hd["rank"]
        self.idx = {e["name"]: i for i, e in enumerate(self.ents)}
        self.man_p = path + ".manifest.json"
        self.man = json.load(open(self.man_p)) if os.path.exists(self.man_p) else None
        self.mt = {t["name"]: t for t in self.man["tensors"]} if self.man else {}
        self.fd = os.open(path, os.O_RDWR)
        self.end = self.hd["file_bytes"]
        self.relast()

    def relast(self):
        self.last = max([i for i, e in enumerate(self.ents) if e["kind"] != glw.RESERVED], default=-1)

    def filled(self, L):
        ids = [self.idx[f"layers.{L}.experts.desc"]] + [self.idx[f"layers.{L}.experts.{e}"] for e in range(E)]
        k = [self.ents[i]["kind"] != glw.RESERVED for i in ids]
        if any(k) and not all(k):
            raise SystemExit(f"{self.path} layer {L}: partly filled")
        return all(k)

    def reset_from(self, L0):
        """Return every expert entry of layers >= L0 to reserved and cut the file at the first one's offset."""
        cut = None
        for e in self.ents:
            if e["layer"] != glw.GLOBAL and e["layer"] >= L0 and e["name"].startswith(f"layers.{e['layer']}.experts.") \
                    and e["kind"] != glw.RESERVED:
                cut = e["offset"] if cut is None else min(cut, e["offset"])
                e.update(offset=0, bytes=0, kind=glw.RESERVED, flags=glw.RESERVED_FLAG, crc=0)
                if e["name"] in self.mt:
                    self.mt[e["name"]].update(offset=0, bytes=0, crc32c="00000000", kind="reserved", sha256="")
        if cut is None:
            return
        later = [e for e in self.ents if e["kind"] != glw.RESERVED and e["offset"] + e["bytes"] > cut]
        if later:
            raise SystemExit(f"{self.path}: {later[0]['name']} lies past the cut at {cut}")
        self.end = cut
        if self.man:
            d = self.man.get("expert_sources", {})
            for n in [n for n in d if int(n.split(".")[2]) >= L0]:
                del d[n]
            for n in [n for n in self.man.get("expert_layers", {}) if int(n) >= L0]:
                del self.man["expert_layers"][n]
        self.relast()
        self.commit({"reset_from_layer": L0, "cut_at": cut,
                     "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(f"{self.path}: expert entries of layers >= {L0} reserved again, file cut at {cut} B", flush=True)

    def declare(self, L, tag):
        """(Re-)declare a reserved layer's entries for the format tag (shape, reserve_bytes, fmt)."""
        eb = glw.expert_entry_bytes(tag)
        for n in [f"layers.{L}.experts.desc"] + [f"layers.{L}.experts.{e}" for e in range(E)]:
            x = self.ents[self.idx[n]]
            if x["kind"] != glw.RESERVED:
                raise SystemExit(f"{self.path} {n}: not reserved")
            x["fmt"] = tag
            if not n.endswith(".desc"):
                x.update(shape=(eb,), reserve_bytes=eb)
        return eb

    def put(self, i, arr, off):
        e = self.ents[i]
        if arr.nbytes != e["reserve_bytes"] or arr.nbytes % glw.ALIGN and not e["name"].endswith(".desc"):
            raise SystemExit(f"{e['name']}: {arr.nbytes} B, reserved {e['reserve_bytes']}")
        os.pwrite(self.fd, memoryview(arr), off)
        pad = glw.align_up(arr.nbytes) - arr.nbytes
        if pad:
            os.pwrite(self.fd, bytes(pad), off + arr.nbytes)
        e.update(offset=off, bytes=arr.nbytes, kind=glw.RECUT, flags=0, crc=glw.crc32c(arr))
        if e["name"] in self.mt:
            self.mt[e["name"]].update(offset=off, bytes=arr.nbytes, crc32c=f"{e['crc']:08x}", kind="recut",
                                      sha256=hashlib.sha256(memoryview(arr)).hexdigest())

    def commit(self, record=None, final=False):
        """Data synced first, then the TOC and header (the file's state after the last whole layer), then the
        manifest: an interruption leaves the last committed layer's file, and a rerun goes on from there."""
        os.fdatasync(self.fd)
        toc = b""
        for e in self.ents:
            o = glw.Out(e["name"], e["dtype"], e["shape"], e["layer"], e["layout"], e["kind"], e["flags"], e["blk"],
                        e["source"].split(",") if e["source"] else [], e["slice"], e["fmt"], e["reserve_bytes"])
            o.bytes, o.offset, o.rel_err = e["bytes"], e["offset"], e["rel_err"]
            toc += o.entry(e["crc"])
        toc_crc = glw.crc32c(toc)
        os.pwrite(self.fd, toc, self.hd["toc_offset"])
        raw = bytearray(os.pread(self.fd, glw.HDR_BYTES, 0))
        h = list(glw.HDR.unpack(bytes(raw[:glw.HDR.size])))
        h[9], h[10], h[12] = self.end - self.hd["data_offset"], self.end, toc_crc
        raw[:glw.HDR.size] = glw.HDR.pack(*h)
        os.pwrite(self.fd, bytes(raw), 0)
        os.ftruncate(self.fd, self.end)
        os.fsync(self.fd)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        self.toc_crc = toc_crc
        if self.man:
            self.man["file_bytes"], self.man["toc_crc32c"] = self.end, f"{toc_crc:08x}"
            self.man["file_sha256"] = None  # set by the final commit
            if record:
                self.man.setdefault("expert_fills", []).append(record)
            tmp = self.man_p + ".tmp"
            json.dump(self.man, open(tmp, "w"), indent=1)
            os.replace(tmp, self.man_p)

    def finish(self, record):
        self.commit(record, final=True)
        os.close(self.fd)
        sha = None
        if self.man:
            import convert
            sha = convert.file_sha256(self.path)
            self.man["file_sha256"] = sha
            tmp = self.man_p + ".tmp"
            json.dump(self.man, open(tmp, "w"), indent=1)
            os.replace(tmp, self.man_p)
        print(f"{self.path}: rank {self.r}, {self.end} B ({self.end / 2**30:.3f} GiB), TOC crc {self.toc_crc:08x}"
              + (f", sha256 {sha}" if sha else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index")
    ap.add_argument("--manifest")
    ap.add_argument("--hashes")
    ap.add_argument("--unverified-ok", action="store_true")
    ap.add_argument("--file", nargs="+", default=[], help="rank files to fill (one pass over the sources)")
    ap.add_argument("--layers", type=int, nargs="+", default=[])
    ap.add_argument("--reset-from", type=int)
    ap.add_argument("--test-packers", action="store_true", help="packers.py's test stand-ins (never for rank files)")
    ap.add_argument("--expert-packer", help="the expert kernels' int4_pack.py (default: engine/weights/int4_pack.py)")
    ap.add_argument("--attn-packer", help="the attention side's pack.py with int4_tiles (default: attn/tools/pack.py)")
    a = ap.parse_args()
    glw._check_crc()
    reg = packers.register(test=a.test_packers, experts=a.expert_packer, attn=a.attn_packer)
    if not (a.manifest and a.hashes) and not a.unverified_ok:
        raise SystemExit("--manifest and --hashes, or --unverified-ok for a test")
    ck = sources.Checkpoint(a.ckpt, a.index, a.manifest, a.hashes)
    T = [Target(f) for f in a.file]
    if a.reset_from is not None:
        for t in T:
            t.reset_from(a.reset_from)
    t0 = time.time()
    done = []
    for L in a.layers:
        st = [t.filled(L) for t in T]
        if all(st):
            print(f"layer {L}: already filled in every file, skipped", flush=True)
            continue
        if any(st):
            raise SystemExit(f"layer {L}: filled in some files only; reset them to a common layer first")
        fmt = glw.layer_expert_fmt(L)
        F = glw.EXPERT_FORMATS.get(fmt)
        if F is None:
            raise SystemExit(f"layer {L}: no packer for {glw.FMT_NAME[fmt]} experts registered")
        tag = glw.expert_fmt(fmt)
        tl = time.time()
        names = [f"layers.{L}.experts.desc"] + [f"layers.{L}.experts.{e}" for e in range(E)]
        for t in T:
            ids = [t.idx[n] for n in names]
            if ids[0] <= t.last or ids != list(range(ids[0], ids[0] + E + 1)):
                raise SystemExit(f"{t.path} layer {L}: out of TOC order (fill earlier layers first, or --reset-from)")
            eb = t.declare(L, tag)
        shapes = [(KI, glw.D), (KI, glw.D), (glw.D, KI)]
        moff = np.cumsum([0] + [F.mat_bytes(*s) for s in shapes])
        if moff[3] != eb:
            raise SystemExit(f"layer {L}: matrices {moff[3]} B, entry {eb} B")
        desc_bytes = E * 3 * glw.MATDESC_BYTES
        srcs = {}
        desc = {t.path: bytearray() for t in T}
        base = {t.path: t.end + glw.align_up(desc_bytes) for t in T}  # desc first, then the 168 experts
        tr = 0.0
        for e in range(E):
            tr0 = time.time()
            full = expert_sources(ck, L, e, fmt, srcs)
            tr += time.time() - tr0
            for t in T:
                parts = []
                for m in range(3):
                    sl = rank_slice(fmt, full[m], m, t.r)
                    b = np.ascontiguousarray(F.pack(sl)).view(np.uint8).reshape(-1)
                    if b.size != moff[m + 1] - moff[m]:
                        raise SystemExit(f"layer {L} expert {e} {PROJ[m]}: packed {b.size} B, format says "
                                         f"{moff[m + 1] - moff[m]}")
                    if not same(fmt, F.unpack(b, *shapes[m]), sl):  # the fill's own read-back
                        raise SystemExit(f"layer {L} expert {e} {PROJ[m]} rank {t.r}: the packed matrix does not "
                                         "unpack to its source slice")
                    parts.append(b)
                    desc[t.path] += F.desc(int(e * eb + moff[m]))
                t.put(t.idx[names[e + 1]], np.concatenate(parts), base[t.path] + e * eb)
            if e % 56 == 55:
                print(f"  layer {L}: expert {e} ({time.time() - tl:.0f} s, {tr:.0f} s reading)", flush=True)
        lrec = {"fmt": f"{tag:03x}", "format": F.name, "entry_bytes": eb, "packer": packers.source_of(F)}
        for t in T:
            d = np.frombuffer(bytes(desc[t.path]), np.uint8)
            if d.size != desc_bytes:
                raise SystemExit(f"layer {L}: desc {d.size} B")
            t.put(t.idx[names[0]], d, t.end)
            t.end = base[t.path] + E * eb
            t.last = t.idx[names[-1]]
            if t.man:
                t.man.setdefault("expert_sources", {}).update(srcs)
                t.man.setdefault("expert_layers", {})[str(L)] = lrec
            t.commit()
        done.append(L)
        print(f"layer {L}: {F.name}, {E * eb / 2**30:.3f} GiB a rank, committed ({time.time() - tl:.0f} s, "
              f"{tr:.0f} s of it reading; {time.time() - t0:.0f} s in all)", flush=True)
    rec = {"layers": done, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "packers": reg,
           "rule": "no requantisation: INT4 packed values, BF16 scales and zero points (layers 4-76) and BF16 "
                   "weights (3, 77, 78) re-tiled; every matrix read back against its source slice",
           "tool": {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()
                    for p in [os.path.join(ROOT, f) for f in ("experts_fill.py", "int4.py", "packers.py", "glw.py")]
                    + packers.sources()}}
    for t in T:
        t.finish(rec)


if __name__ == "__main__":
    sys.exit(main())
