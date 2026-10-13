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

Filling must follow the TOC order (layer 3 before 4 ...), which keeps the data region in TOC order.
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


class Target:
    def __init__(self, path):
        self.path = path
        self.hd, self.ents, _ = glw.read_toc(path)
        self.r = self.hd["rank"]
        self.idx = {e["name"]: i for i, e in enumerate(self.ents)}
        self.last = max([i for i, e in enumerate(self.ents) if e["kind"] != glw.RESERVED], default=-1)
        self.man_p = path + ".manifest.json"
        self.man = json.load(open(self.man_p)) if os.path.exists(self.man_p) else None
        self.mt = {t["name"]: t for t in self.man["tensors"]} if self.man else {}
        self.fd = os.open(path, os.O_RDWR)
        self.end = self.hd["file_bytes"]

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

    def finish(self, record):
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
        os.close(self.fd)
        sha = None
        if self.man:
            import convert
            sha = convert.file_sha256(self.path)
            self.man["file_bytes"], self.man["toc_crc32c"], self.man["file_sha256"] = self.end, f"{toc_crc:08x}", sha
            self.man.setdefault("expert_fills", []).append(record)
            json.dump(self.man, open(self.man_p, "w"), indent=1)
        print(f"{self.path}: rank {self.r}, {self.end} B ({self.end / 2**30:.3f} GiB), TOC crc {toc_crc:08x}"
              + (f", sha256 {sha}" if sha else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--file", nargs="+", default=[], help="rank files to fill (one pass over the sources)")
    ap.add_argument("--layers", type=int, nargs="+", default=[])
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
    ck = sources.Checkpoint(a.ckpt)
    if a.rtn_check and not rtn_check(ck, a.rtn_check, a.check_layer, a.check_experts):
        raise SystemExit("nvfp4_rtn does not reproduce NVIDIA's NVFP4: not filling")
    if a.dump_canon:
        os.makedirs(a.dump_canon, exist_ok=True)
    T = [Target(f) for f in a.file]
    t0 = time.time()
    for L in a.layers:
        names = [f"layers.{L}.experts.desc"] + [f"layers.{L}.experts.{e}" for e in range(E)]
        for t in T:
            ids = [t.idx[n] for n in names]
            if any(t.ents[i]["kind"] != glw.RESERVED for i in ids) or ids[0] <= t.last or \
                    ids != list(range(ids[0], ids[0] + E + 1)):
                raise SystemExit(f"{t.path} layer {L}: already filled, or out of TOC order")
            if any(t.ents[i]["fmt"] != glw.expert_fmt() for i in ids[1:]):
                raise SystemExit(f"{t.path} layer {L}: an expert's format tag is not NVFP4")
        srcs, rel = {}, {}
        desc = {t.path: bytearray() for t in T}
        base = {t.path: t.end + E * 3 * glw.MATDESC_BYTES for t in T}  # desc first, then the 256 experts
        for e in range(E):
            canon = expert_canon(ck, L, e, srcs, rel)
            er = max([rel.get(f"model.layers.{L}.mlp.experts.{e}.{p}_proj.weight", 0.0) for p in ("gate", "up", "down")])
            for t in T:
                ent = np.empty(3 * MAT, np.uint8)
                for m, (w, s, s2) in enumerate(canon):
                    ws, ss = rank_slice(m, w, s, t.r)
                    ent[m * MAT:(m + 1) * MAT] = pack_nvfp4(ws, ss)
                    desc[t.path] += matdesc((3 * e + m) * MAT, s2)
                    if a.dump_canon and e in a.dump_experts:
                        proj = ("gate", "up", "down")[m]
                        ws.tofile(os.path.join(a.dump_canon, f"L{L}.r{t.r}.e{e}.{proj}.w"))
                        ss.tofile(os.path.join(a.dump_canon, f"L{L}.r{t.r}.e{e}.{proj}.s"))
                t.put(t.idx[names[e + 1]], ent, base[t.path] + e * 3 * MAT, er, L == glw.MTP)
            if e % 32 == 31:
                print(f"  layer {L}: expert {e} ({time.time() - t0:.0f} s)", flush=True)
        for t in T:
            d = np.frombuffer(bytes(desc[t.path]), np.uint8)
            t.put(t.idx[names[0]], d, t.end)
            t.end = base[t.path] + E * 3 * MAT
            t.last = t.idx[names[-1]]
            if t.man:
                t.man.setdefault("expert_sources", {}).update(srcs)
                if rel:
                    t.man.setdefault("expert_rel_err", {}).update(rel)
            os.fdatasync(t.fd)
            os.posix_fadvise(t.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        print(f"layer {L}: {E * 3 * MAT / 2**30:.3f} GiB of NVFP4 blocks a rank" +
              (f", relative RMS error vs the BF16 {min(rel.values()):.4f}-{max(rel.values()):.4f}" if rel else "") +
              f" ({time.time() - t0:.0f} s)", flush=True)
    rec = {"layers": a.layers, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "format": "NVFP4: NVIDIA's codes and scales for layers 3-77; for the MTP layer (78) our round-to-nearest-even "
                     "of its BF16 by NVIDIA's recipe (quant/formats.py nvfp4_rtn), outside the exactness bar "
                     "(DESIGN decision 6: it only drafts)",
           "tool": {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
                    for d, f in ((ROOT, "experts_fill.py"), (QUANT, "formats.py"))}}
    for t in T:
        t.finish(rec)


if __name__ == "__main__":
    sys.exit(main())
