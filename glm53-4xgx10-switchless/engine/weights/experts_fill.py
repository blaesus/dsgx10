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


def layer_experts(ck, L, r, dump=None, dump_experts=()):
    """Rank r's 256 expert entries (gate, up, down NVFP4 blocks back to back) and the desc table, offsets relative to
    expert 0's entry; plus the source tensors' SHA-256s."""
    import torch
    p = f"model.layers.{L}.mlp.experts."
    experts, desc, srcs = [], bytearray(), {}
    for e in range(E):
        ent = np.empty(3 * MAT, np.uint8)
        for m, proj in enumerate(("gate", "up", "down")):
            n = f"{p}{e}.{proj}_proj."
            w, ws = ck.get(n + "weight")
            s, ss = ck.get(n + "weight_scale")
            s2, s2s = ck.get(n + "weight_scale_2")
            srcs[n + "weight"], srcs[n + "weight_scale"], srcs[n + "weight_scale_2"] = ws, ss, s2s
            w, s = w.numpy(), s.view(torch.uint8).numpy()
            if proj == "down":  # [6144][1024 B], [6144][128]: the rank's input columns
                w, s = np.ascontiguousarray(w[:, 256 * r:256 * r + 256]), np.ascontiguousarray(s[:, 32 * r:32 * r + 32])
            else:  # [2048][3072 B], [2048][384]: the rank's rows
                w, s = np.ascontiguousarray(w[KI * r:KI * r + KI]), np.ascontiguousarray(s[KI * r:KI * r + KI])
            ent[m * MAT:(m + 1) * MAT] = pack_nvfp4(w, s)
            desc += matdesc((3 * e + m) * MAT, float(s2.reshape(-1)[0]))
            if dump and e in dump_experts:
                w.tofile(os.path.join(dump, f"L{L}.r{r}.e{e}.{proj}.w"))
                s.tofile(os.path.join(dump, f"L{L}.r{r}.e{e}.{proj}.s"))
        experts.append(ent)
    return experts, np.frombuffer(bytes(desc), np.uint8), srcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--dump-canon")
    ap.add_argument("--dump-experts", type=int, nargs="*", default=[0, 17, 255])
    a = ap.parse_args()
    glw._check_crc()
    ck = sources.Checkpoint(a.ckpt)
    hd, ents, _ = glw.read_toc(a.file)
    r = hd["rank"]
    idx = {e["name"]: i for i, e in enumerate(ents)}
    last_filled = max([i for i, e in enumerate(ents) if e["kind"] != glw.RESERVED], default=-1)
    man_p = a.file + ".manifest.json"
    man = json.load(open(man_p)) if os.path.exists(man_p) else None
    if a.dump_canon:
        os.makedirs(a.dump_canon, exist_ok=True)
    fd = os.open(a.file, os.O_RDWR)
    t0 = time.time()
    end = hd["file_bytes"]
    for L in a.layers:
        names = [f"layers.{L}.experts.desc"] + [f"layers.{L}.experts.{e}" for e in range(E)]
        ids = [idx[n] for n in names]
        if any(ents[i]["kind"] != glw.RESERVED for i in ids) or ids[0] <= last_filled or ids != list(range(ids[0], ids[0] + E + 1)):
            raise SystemExit(f"layer {L}: already filled, or out of TOC order")
        if any(ents[i]["fmt"] != glw.expert_fmt() for i in ids[1:]):
            raise SystemExit(f"layer {L}: an expert's format tag is not NVFP4; this fill writes NVIDIA's NVFP4 only")
        if L == glw.MTP:
            raise SystemExit("the MTP layer's experts are BF16 in NVIDIA's file: NVFP4 by our RTN is a later item")
        experts, desc, srcs = layer_experts(ck, L, r, a.dump_canon, a.dump_experts)
        for i, arr in zip(ids, [desc] + experts):
            e = ents[i]
            if arr.nbytes != e["reserve_bytes"] or arr.nbytes % glw.ALIGN:
                raise SystemExit(f"{e['name']}: {arr.nbytes} B, reserved {e['reserve_bytes']}")
            os.pwrite(fd, memoryview(arr), end)
            e.update(offset=end, bytes=arr.nbytes, kind=glw.RECUT, flags=0, crc=glw.crc32c(arr))
            if man:
                for t in man["tensors"]:
                    if t["name"] == e["name"]:
                        t.update(offset=end, bytes=arr.nbytes, crc32c=f"{e['crc']:08x}", kind="recut",
                                 sha256=hashlib.sha256(memoryview(arr)).hexdigest())
            end += arr.nbytes
        if man:
            man.setdefault("expert_sources", {}).update(srcs)
        last_filled = ids[-1]
        os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        print(f"layer {L}: {sum(x.nbytes for x in experts) / 2**30:.3f} GiB of NVFP4 blocks + {desc.nbytes} B of "
              f"descriptors ({time.time() - t0:.0f} s)", flush=True)
    # TOC and header
    toc = b""
    for e in ents:
        o = glw.Out(e["name"], e["dtype"], e["shape"], e["layer"], e["layout"], e["kind"], e["flags"], e["blk"],
                    e["source"].split(",") if e["source"] else [], e["slice"], e["fmt"], e["reserve_bytes"])
        o.bytes, o.offset, o.rel_err = e["bytes"], e["offset"], e["rel_err"]
        toc += o.entry(e["crc"])
    toc_crc = glw.crc32c(toc)
    os.pwrite(fd, toc, hd["toc_offset"])
    raw = bytearray(os.pread(fd, glw.HDR_BYTES, 0))
    h = list(glw.HDR.unpack(bytes(raw[:glw.HDR.size])))
    h[9], h[10], h[12] = end - hd["data_offset"], end, toc_crc  # data_bytes, file_bytes, toc_crc32c
    raw[:glw.HDR.size] = glw.HDR.pack(*h)
    os.pwrite(fd, bytes(raw), 0)
    os.ftruncate(fd, end)
    os.fsync(fd)
    os.close(fd)
    if man:
        import convert
        man["file_bytes"], man["toc_crc32c"] = end, f"{toc_crc:08x}"
        man["file_sha256"] = convert.file_sha256(a.file)
        man.setdefault("expert_fills", []).append({"layers": a.layers, "format": "NVFP4 (NVIDIA's codes and scales)",
                                                   "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        json.dump(man, open(man_p, "w"), indent=1)
    print(f"{a.file}: {end} B, TOC crc {toc_crc:08x} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
