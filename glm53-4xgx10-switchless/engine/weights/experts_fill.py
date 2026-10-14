#!/usr/bin/env python3
"""Fill a rank file's reserved routed-expert entries with NVIDIA's NVFP4 experts (the first fill; the E2M1-subset
"S3" experts come from the quantiser later and use the same pool and descriptor table).

    python -I experts_fill.py --ckpt NV --file rankR.glw --layers 3 [4 ...] [--dump-canon DIR --dump-experts 0 17 255]

A layer's TOC entries are its desc table (gx::MatDesc [256][3]) and one entry an expert (gate, up, down blocks back
to back, format tag gate | up << 4 | down << 8, glw.h ExpertFmt); the 256 expert entries are contiguous and form
the layer's pool, MatDesc::off relative to expert 0's entry.

For each layer, in TOC order: rank r's slice of every expert's gate and up (rows [512r, +512)) and down (input
columns [512r, +512): value bytes [256r, +256), scale columns [32r, +32)) of NVIDIA's canonical NVFP4 tensors
(docs/model.md § 5.2) is re-tiled into the expert kernels' 8-row x 512-k blocks (docs/experts.md § 2,
engine/experts/layout.h pack_nvfp4, ported here to numpy), one entry an expert; the layer's
descriptor table is gx::MatDesc [256][3] with off = (3e + m) * 1,769,472, s2 = weight_scale_2, fmt = kNVFP4
(tab and nib zero: NVFP4 needs neither). The data is appended at the file's end, the two TOC entries are filled in
(kind recut, CRC), and the TOC and header rewritten; no other byte moves, so every
non-expert tensor keeps its offset, CRC and SHA-256. The manifest beside the file gains the entries and the new
file SHA-256. --dump-canon writes the canonical rank slices of a few experts as raw files for
tests/expert_fill_test (layout.h's own reader against its own canonical reader).

Filling must follow the TOC order (layer 3 before 4 ...), which keeps the data region in TOC order; --reset-from L
returns layers >= L to reserved and cuts the files there. Each layer is committed (data synced, then TOC, header,
manifest) before the next, so a rerun with the same arguments resumes after the last whole layer.

--recipe mix|s3x2 (with --master, the BF16 experts) quantises each expert once with s3x2_pack.py and cuts every
rank's slice from that result; the layer's entries are re-declared for the recipe's format tag and size.
--recipe nvfp4 copies NVIDIA's bytes, or with --master quantises by nvfp4_rtn and checks them against NVIDIA's.
"""
import argparse, hashlib, json, os, struct, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

import glw  # noqa: E402
import sources  # noqa: E402

KH, KI, E = 6144, 512, 256
MAT = (KI // 8) * (KH // 512) * glw.Q4_BLK  # 1,769,472: gate/up [512][6144] = down [6144][512]
assert MAT == (KH // 8) * (KI // 512) * glw.Q4_BLK


def pack_nvfp4(w, s):
    """layout.h pack_nvfp4: canonical w [N][K/2] (low nibble = even k), s [N][K/16] e4m3 -> [N/8][K/512][2304 B].
    Values: block byte 512 jj + 16 l + 8 q + c = w[n][256 b + 64 jj + 32 q + 8 t + c] (l = 4 g + t, n = 8 n8 + g);
    scales: block byte 2048 + 8 l + 2 jj + q = s[n][32 b + 8 jj + 4 q + t]."""
    N, Kb = w.shape
    K = 2 * Kb
    B = K // 512
    v = w.reshape(N // 8, 8, B, 4, 2, 4, 8).transpose(0, 2, 3, 1, 5, 4, 6).reshape(N // 8, B, 2048)
    sc = s.reshape(N // 8, 8, B, 4, 2, 4).transpose(0, 2, 1, 5, 3, 4).reshape(N // 8, B, 256)
    return np.ascontiguousarray(np.concatenate([v, sc], axis=2)).reshape(-1)


def matdesc(off, s2, fmt=0):
    return struct.pack("<QfI8I2I2I", off, s2, fmt, *([0] * 12))


QUANT = os.path.normpath(os.path.join(ROOT, "..", "..", "quant"))


def nvfp4_rtn_module():
    """quant/formats.py's nvfp4_rtn: NVIDIA's recipe (s2 = amax * f32(1/2688), e4m3 per 16 along K, E2M1
    round-to-nearest-even, low nibble first; gate and up share amax)."""
    if QUANT not in sys.path:
        sys.path.insert(0, QUANT)
    import formats
    return formats


def expert_canon(ck, L, e, srcs, rel):
    """Expert e's canonical NVFP4 (w [N][K/2], s [N][K/16] uint8, s2 float) for gate, up, down: NVIDIA's tensors
    for layers 3-77; for the MTP layer (BF16 in NVIDIA's file) our round-to-nearest-even by NVIDIA's recipe,
    with the relative RMS error against the BF16 recorded in `rel`."""
    import torch
    p = f"model.layers.{L}.mlp.experts.{e}."
    out = []
    if L != glw.MTP:
        for proj in ("gate", "up", "down"):
            n = f"{p}{proj}_proj."
            w, ws = ck.get(n + "weight")
            s, ss = ck.get(n + "weight_scale")
            s2, s2s = ck.get(n + "weight_scale_2")
            srcs[n + "weight"], srcs[n + "weight_scale"], srcs[n + "weight_scale_2"] = ws, ss, s2s
            out.append((w.numpy(), s.view(torch.uint8).numpy(), float(s2.reshape(-1)[0])))
        return out
    F = nvfp4_rtn_module()
    W = {}
    for proj in ("gate", "up", "down"):
        n = f"{p}{proj}_proj.weight"
        t, sha = ck.get(n)
        srcs[n] = sha
        W[proj] = t.float()
    gu = torch.maximum(W["gate"].abs().max(), W["up"].abs().max())
    for proj in ("gate", "up", "down"):
        q = F.nvfp4_rtn(W[proj], gu if proj != "down" else None)
        d = F.nvfp4_dequant(q)
        rel[f"{p}{proj}_proj.weight"] = float((d - W[proj]).norm() / W[proj].norm())
        out.append((q["weight"], q["weight_scale"], float(q["weight_scale_2"])))
    return out


def rank_slice(m, w, s, r):
    if m == 2:  # down [6144][1024 B], [6144][128]: the rank's input columns
        return np.ascontiguousarray(w[:, 256 * r:256 * r + 256]), np.ascontiguousarray(s[:, 32 * r:32 * r + 32])
    return np.ascontiguousarray(w[KI * r:KI * r + KI]), np.ascontiguousarray(s[KI * r:KI * r + KI])


def rtn_check(ck, master, layer, experts):
    """nvfp4_rtn on the BF16 master's experts of a layer NVIDIA quantised must give NVIDIA's bytes."""
    import torch
    F = nvfp4_rtn_module()
    man = {}
    for line in open(os.path.join(master, "manifest.jsonl")):
        x = json.loads(line)
        man[x["name"]] = x
    eq = tot = 0
    for e in experts:
        p = f"model.layers.{layer}.mlp.experts.{e}."
        W = {}
        for proj in ("gate", "up", "down"):
            n = f"{p}{proj}_proj.weight"
            buf = open(os.path.join(master, "tensors", n + ".bin"), "rb").read()
            if hashlib.sha256(buf).hexdigest() != man[n]["sha256"]:
                raise SystemExit(f"{n}: the master file's SHA-256 differs from its fetch record")
            W[proj] = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(man[n]["shape"]).float()
        gu = torch.maximum(W["gate"].abs().max(), W["up"].abs().max())
        for proj in ("gate", "up", "down"):
            q = F.nvfp4_rtn(W[proj], gu if proj != "down" else None)
            n = f"{p}{proj}_proj."
            nw = ck.get(n + "weight")[0].numpy()
            ns = ck.get(n + "weight_scale")[0].view(torch.uint8).numpy()
            n2 = ck.get(n + "weight_scale_2")[0].reshape(-1).numpy()
            ok = (np.array_equal(q["weight"], nw), np.array_equal(q["weight_scale"], ns),
                  q["weight_scale_2"].reshape(-1).view(np.uint32)[0] == n2.view(np.uint32)[0])
            eq += all(ok)
            tot += 1
            print(f"  {n}: codes {'=' if ok[0] else int((q['weight'] != nw).sum())}, scales "
                  f"{'=' if ok[1] else int((q['weight_scale'] != ns).sum())}, scale_2 {'=' if ok[2] else 'differs'}")
    print(f"nvfp4_rtn against NVIDIA's bytes, layer {layer}: {eq} of {tot} tensors identical")
    return eq == tot


def unpack_nvfp4(b, N, K):
    """pack_nvfp4's inverse (the fill's own read-back of every matrix it writes)."""
    B = K // 512
    blk = b.reshape(N // 8, B, glw.Q4_BLK)
    w = blk[:, :, :2048].reshape(N // 8, B, 4, 8, 4, 2, 8).transpose(0, 3, 1, 2, 5, 4, 6).reshape(N, K // 2)
    s = blk[:, :, 2048:].reshape(N // 8, B, 8, 4, 4, 2).transpose(0, 2, 1, 4, 5, 3).reshape(N, K // 16)
    return w, s


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
                e.update(offset=0, bytes=0, kind=glw.RESERVED, flags=glw.RESERVED_FLAG, crc=0, rel_err=0.0)
                if e["name"] in self.mt:
                    self.mt[e["name"]].update(offset=0, bytes=0, crc32c="00000000", kind="reserved", rel_err=0.0,
                                              sha256="")
        if cut is None:
            return
        later = [e for e in self.ents if e["kind"] != glw.RESERVED and e["offset"] + e["bytes"] > cut]
        if later:
            raise SystemExit(f"{self.path}: {later[0]['name']} lies past the cut at {cut}")
        self.end = cut
        if self.man:
            for k in ("expert_rel_err", "expert_sources"):
                d = self.man.get(k, {})
                for n in [n for n in d if int(n.split(".")[2]) >= L0]:
                    del d[n]
        self.relast()
        self.commit({"reset_from_layer": L0, "cut_at": cut,
                     "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(f"{self.path}: expert entries of layers >= {L0} reserved again, file cut at {cut} B", flush=True)

    def declare(self, L, tag):
        """Re-declare a reserved layer's entries for the format tag (shape, reserve_bytes, fmt)."""
        eb = glw.expert_entry_bytes(tag)
        for n in [f"layers.{L}.experts.desc"] + [f"layers.{L}.experts.{e}" for e in range(E)]:
            x = self.ents[self.idx[n]]
            if x["kind"] != glw.RESERVED:
                raise SystemExit(f"{self.path} {n}: not reserved")
            x["fmt"] = tag
            if not n.endswith(".desc"):
                x.update(shape=(eb,), reserve_bytes=eb)
        return eb

    def put(self, i, arr, off, rel_err=0.0, lossy=False):
        e = self.ents[i]
        if arr.nbytes != e["reserve_bytes"] or arr.nbytes % glw.ALIGN:
            raise SystemExit(f"{e['name']}: {arr.nbytes} B, reserved {e['reserve_bytes']}")
        os.pwrite(self.fd, memoryview(arr), off)
        e.update(offset=off, bytes=arr.nbytes, kind=glw.RECUT, flags=glw.LOSSY if lossy else 0,
                 crc=glw.crc32c(arr), rel_err=rel_err)
        if e["name"] in self.mt:
            self.mt[e["name"]].update(offset=off, bytes=arr.nbytes, crc32c=f"{e['crc']:08x}", kind="recut",
                                      rel_err=rel_err, sha256=hashlib.sha256(memoryview(arr)).hexdigest())

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
        if final or record:
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


class Master:
    """BF16 master experts fetched by quant/fetch_master.py: DIR/manifest.jsonl (name, shard, dtype, shape,
    sha256) and DIR/tensors/NAME.bin; every tensor checked against its fetch record as it is read."""

    def __init__(self, d):
        self.dir = os.path.expanduser(d)
        self.man = {}
        for line in open(os.path.join(self.dir, "manifest.jsonl")):
            x = json.loads(line)
            self.man[x["name"]] = x
        self.man_sha = hashlib.sha256(open(os.path.join(self.dir, "manifest.jsonl"), "rb").read()).hexdigest()

    def __contains__(self, n):
        return n in self.man

    def get(self, n):
        import torch
        r = self.man[n]
        buf = open(os.path.join(self.dir, "tensors", n + ".bin"), "rb").read()
        sha = hashlib.sha256(buf).hexdigest()
        if sha != r["sha256"] or r["dtype"] != "BF16":
            raise SystemExit(f"{n}: SHA-256 or dtype differs from the fetch record")
        return torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(r["shape"]), sha


def provenance_index(path):
    """quant/run.py's records (all.jsonl): (layer, expert, quantiser format, proj) -> SHA-256 of the packed dict."""
    out = {}
    for line in open(path):
        x = json.loads(line)
        for proj, r in x.get("proj", {}).items():
            if "sha256" in r:
                out.setdefault((x["layer"], x["expert"], x["format"], proj), set()).add(r["sha256"])
    return out


def phash(packed):
    """quant/run.py phash: SHA-256 over the packed dict's sorted keys, dtypes, shapes and bytes."""
    h = hashlib.sha256()
    for k in sorted(packed):
        v = np.ascontiguousarray(packed[k])
        h.update(k.encode())
        h.update(str(v.dtype).encode() + str(v.shape).encode())
        h.update(v.tobytes())
    return h.hexdigest()


QFMT = {0: "nvfp4", 1: "e2m1sub3x2_g16"}  # s3x2_pack format -> quant/run.py's format name


def expert_quantised(S, src, L, e, recipe, srcs, rel, shards, prov, pstat, nv=None):
    """Quantise expert e once from its BF16 (s3x2_pack.quantise_expert): [(fmt, canon)] x 3 full tensors."""
    W = []
    for proj in S.PROJ:
        n = f"model.layers.{L}.mlp.experts.{e}.{proj}_proj.weight"
        t, sha = src.get(n)
        srcs[n] = sha
        if isinstance(src, Master):
            shards[src.man[n]["shard"]] = None
        W.append(t.float())
    q = S.quantise_expert(*W, recipe)
    for m, (f, c) in enumerate(q):
        sl = {"fmt": f, "w": S.canon_arrays(f, c)[0], "s": S.canon_arrays(f, c)[1]}
        d = S.value_nos2(sl) * float(S.canon_arrays(f, c)[2])
        rel[f"model.layers.{L}.mlp.experts.{e}.{S.PROJ[m]}_proj.weight"] = float((d - W[m]).norm() / W[m].norm())
        if f == S.FMT_NVFP4 and nv is not None and L != glw.MTP:
            # NVFP4 has a stronger reference than the records: NVIDIA's own bytes
            import torch
            n = f"model.layers.{L}.mlp.experts.{e}.{S.PROJ[m]}_proj."
            ok = (np.array_equal(c["weight"], nv.get(n + "weight")[0].numpy()) and
                  np.array_equal(c["weight_scale"], nv.get(n + "weight_scale")[0].view(torch.uint8).numpy()) and
                  np.float32(c["weight_scale_2"]).view(np.uint32) ==
                  np.float32(nv.get(n + "weight_scale_2")[0].reshape(-1)[0].item()).view(np.uint32))
            if not ok:
                raise SystemExit(f"layer {L} expert {e} {S.PROJ[m]}: nvfp4_rtn of the master differs from NVIDIA's bytes")
            pstat["nv_equal"] += 1
        elif prov is not None:
            want = prov.get((L, e, QFMT[f], S.PROJ[m]))
            if want is None:
                pstat["absent"] += 1
            elif len(want) > 1:
                raise SystemExit(f"layer {L} expert {e} {S.PROJ[m]}: the quantiser's records disagree")
            elif phash(c) in want:
                pstat["equal"] += 1
            else:
                raise SystemExit(f"layer {L} expert {e} {S.PROJ[m]}: the packed {QFMT[f]} differs from the "
                                 "quantiser's record (provenance)")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="NVIDIA's NVFP4 checkpoint (layers 3-77 NVFP4, the MTP layer BF16)")
    ap.add_argument("--file", nargs="+", default=[], help="rank files to fill (one pass over the sources)")
    ap.add_argument("--layers", type=int, nargs="+", default=[])
    ap.add_argument("--recipe", choices=["nvfp4", "mix", "s3x2"], default="nvfp4",
                    help="nvfp4: NVIDIA's bytes (layers 3-77, from --ckpt) or nvfp4_rtn of the BF16; mix: S3x2 gate/up "
                    "+ NVFP4 down; s3x2: S3x2 on all three (engine/weights/s3x2_pack.py, from the BF16)")
    ap.add_argument("--master", help="BF16 master directory (quant/fetch_master.py) for layers 3-77; needed for "
                    "mix and s3x2; with nvfp4 the experts are then quantised by nvfp4_rtn instead of copied")
    ap.add_argument("--provenance", help="quant/run.py's all.jsonl: every S3x2 tensor it has a record of must have "
                    "its SHA-256 (NVFP4 tensors quantised from the master are compared with NVIDIA's bytes instead)")
    ap.add_argument("--reset-from", type=int, help="first return the expert entries of layers >= this to reserved "
                    "and cut the files there (to fill earlier layers after later ones)")
    ap.add_argument("--dump-canon")
    ap.add_argument("--dump-experts", type=int, nargs="*", default=[0, 17, 255])
    ap.add_argument("--rtn-check", help="BF16 master fetch directory (quant/fetch_master.py): check nvfp4_rtn "
                    "against NVIDIA's bytes on --check-layer's experts first")
    ap.add_argument("--check-layer", type=int, default=3)
    ap.add_argument("--check-experts", type=int, nargs="*", default=[0, 1, 2, 3, 100, 255])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    glw._check_crc()
    import s3x2_pack as S
    ck = sources.Checkpoint(a.ckpt)
    master = Master(a.master) if a.master else None
    prov = provenance_index(a.provenance) if a.provenance else None
    pstat = {"equal": 0, "absent": 0, "nv_equal": 0}
    if a.rtn_check and not rtn_check(ck, a.rtn_check, a.check_layer, a.check_experts):
        raise SystemExit("nvfp4_rtn does not reproduce NVIDIA's NVFP4: not filling")
    if a.dump_canon:
        os.makedirs(a.dump_canon, exist_ok=True)
    T = [Target(f) for f in a.file]
    if a.reset_from is not None:
        for t in T:
            t.reset_from(a.reset_from)
    tag = glw.expert_fmt(*S.RECIPES[a.recipe])
    uses_s3 = S.FMT_S3 in S.RECIPES[a.recipe]
    t0 = time.time()
    done = []
    for L in a.layers:
        st = [t.filled(L) for t in T]
        if all(st):
            print(f"layer {L}: already filled in every file, skipped", flush=True)
            continue
        if any(st):
            raise SystemExit(f"layer {L}: filled in some files only; reset them to a common layer first")
        # the source: NVIDIA's NVFP4 bytes (nvfp4, no master), else the BF16 (the master; the MTP layer's in --ckpt)
        copy_nv = a.recipe == "nvfp4" and L != glw.MTP and master is None
        src = ck if L == glw.MTP else master
        if not copy_nv and src is None:
            raise SystemExit(f"layer {L}: recipe {a.recipe} needs the BF16 master (--master)")
        tl = time.time()
        names = [f"layers.{L}.experts.desc"] + [f"layers.{L}.experts.{e}" for e in range(E)]
        for t in T:
            ids = [t.idx[n] for n in names]
            if ids[0] <= t.last or ids != list(range(ids[0], ids[0] + E + 1)):
                raise SystemExit(f"{t.path} layer {L}: out of TOC order (fill earlier layers first, or --reset-from)")
            eb = t.declare(L, tag)
        srcs, rel, shards = {}, {}, {}
        desc = {t.path: bytearray() for t in T}
        base = {t.path: t.end + E * 3 * glw.MATDESC_BYTES for t in T}  # desc first, then the 256 experts
        tq = 0.0
        for e in range(E):
            tq0 = time.time()
            if copy_nv:
                q = [(S.FMT_NVFP4, {"weight": w, "weight_scale": s, "weight_scale_2": np.float32(s2)})
                     for w, s, s2 in expert_canon(ck, L, e, srcs, rel)]
            else:
                q = expert_quantised(S, src, L, e, a.recipe, srcs, rel, shards, prov, pstat, ck)
            tq += time.time() - tq0
            er = max([rel.get(f"model.layers.{L}.mlp.experts.{e}.{p}_proj.weight", 0.0) for p in ("gate", "up", "down")])
            for t in T:
                sl = [S.rank_slice(f, c, m, t.r) for m, (f, c) in enumerate(q)]
                ent, etag, _ = S.expert_entry(sl)
                if etag != tag or ent.size != eb:
                    raise SystemExit(f"layer {L} expert {e}: tag {etag:x} / {ent.size} B, declared {tag:x} / {eb} B")
                off = 0
                for m, x in enumerate(sl):  # the fill's own read-back of each NVFP4 matrix
                    mb = S.mat_bytes(x["fmt"], x["w"].shape[0], x["s"].shape[1] * 16)
                    if x["fmt"] == S.FMT_NVFP4:
                        bw, bs = unpack_nvfp4(ent[off:off + mb], x["w"].shape[0], x["s"].shape[1] * 16)
                        if not (np.array_equal(bw, x["w"]) and np.array_equal(bs, x["s"])):
                            raise SystemExit(f"layer {L} expert {e} matrix {m} rank {t.r}: the packed blocks do not "
                                             "unpack to the source slice")
                    off += mb
                desc[t.path] += S.expert_descs(sl, e * eb)
                if a.dump_canon and e in a.dump_experts:
                    for m, x in enumerate(sl):
                        x["w"].tofile(os.path.join(a.dump_canon, f"L{L}.r{t.r}.e{e}.{S.PROJ[m]}.w"))
                        x["s"].tofile(os.path.join(a.dump_canon, f"L{L}.r{t.r}.e{e}.{S.PROJ[m]}.s"))
                t.put(t.idx[names[e + 1]], ent, base[t.path] + e * eb, er, bool(rel))
            if e % 64 == 63:
                print(f"  layer {L}: expert {e} ({time.time() - tl:.0f} s, {tq:.0f} s quantising)", flush=True)
        lrec = {"recipe": a.recipe, "fmt": f"{tag:03x}", "entry_bytes": eb,
                "source": "nvidia-nvfp4" if copy_nv else ("nvidia-bf16" if L == glw.MTP else "master-bf16")}
        if uses_s3:
            lrec["subsets"] = [list(x) for x in S.SUBSETS]
        if shards:
            lrec["master_shards"] = sorted(shards)
            lrec["master_manifest_sha256"] = master.man_sha
        for t in T:
            d = np.frombuffer(bytes(desc[t.path]), np.uint8)
            t.put(t.idx[names[0]], d, t.end)
            t.end = base[t.path] + E * eb
            t.last = t.idx[names[-1]]
            if t.man:
                t.man.setdefault("expert_sources", {}).update(srcs)
                t.man.setdefault("expert_layers", {})[str(L)] = lrec
                if rel:
                    t.man.setdefault("expert_rel_err", {}).update(rel)
            t.commit()
        done.append(L)
        print(f"layer {L}: {a.recipe}, {E * eb / 2**30:.3f} GiB a rank, committed" +
              (f", relative RMS error vs the BF16 {min(rel.values()):.4f}-{max(rel.values()):.4f}" if rel else "") +
              f" ({time.time() - tl:.0f} s, {tq:.0f} s of it reading and quantising; {time.time() - t0:.0f} s in all)",
              flush=True)
    if prov is not None:
        print(f"provenance: {pstat['equal']} S3x2 tensors equal to the quantiser's records, "
              f"{pstat['absent']} without a record", flush=True)
    if pstat["nv_equal"]:
        print(f"{pstat['nv_equal']} NVFP4 tensors from the master equal to NVIDIA's bytes", flush=True)
    rec = {"layers": done, "recipe": a.recipe, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "format": {"nvfp4": "NVFP4: NVIDIA's codes and scales for layers 3-77 (or nvfp4_rtn of the BF16 master, "
                               "the same bytes); the MTP layer (78) nvfp4_rtn of its BF16",
                      "mix": "S3x2 gate/up (quant/formats.py levelset3 with the subsets) + NVFP4 down (nvfp4_rtn)",
                      "s3x2": "S3x2 on gate, up and down (levelset3 with the subsets)"}[a.recipe]
                     + "; the MTP layer is outside the exactness bar (DESIGN decision 6: it only drafts)",
           "tool": {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
                    for d, f in ((ROOT, "experts_fill.py"), (ROOT, "s3x2_pack.py"), (QUANT, "formats.py"))}}
    if uses_s3:
        rec["subsets"] = [list(x) for x in S.SUBSETS]
    if prov is not None or pstat["nv_equal"]:
        rec["provenance"] = dict(pstat, records=None if prov is None else hashlib.sha256(open(a.provenance, "rb").read()).hexdigest())
    for t in T:
        t.finish(rec)


if __name__ == "__main__":
    sys.exit(main())
