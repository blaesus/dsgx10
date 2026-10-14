#!/usr/bin/env python3
"""altar-1's routed experts for the converter: the checkpoint's INT4-g32 (compressed-tensors pack-quantized) and
BF16 expert tensors cut to a rank's slice and re-tiled, losslessly, into the expert kernels' blocks, with their
descriptors (docs/experts.md section 2; engine/experts/layout.h is the specification this ports, tested byte for
byte against it by engine/experts/tests/test_experts.cu --real on this module's dumps).

Nothing is quantised: every stored nibble, scale, zero point and BF16 value is the checkpoint's. numpy only.

Formats (gx::Fmt, glw.h ExpertFmt): NVFP4 0 (kept for the MTP layer's option), INT4 2 (layers 4-76), BF16 3 (layers
3, 77 and the MTP layer 78). Tag 1 (GLM-5.3's S3) is retired and refused, as every unknown tag.

What the converter calls (experts_fill.py):

    import int4_pack as P
    c = [P.canon_int4(wp, s, zp) for (wp, s, zp) in gate_up_down]   # checkpoint arrays (int32, uint16, int32)
      # or P.canon_bf16(w) for a BF16 layer (uint16 bf16 bits [N][K])
    sl = [P.rank_slice(c[m], m, r) for m in range(3)]                # rank r's gate/up rows, down columns
    entry, tag, moff = P.expert_entry(sl)                             # the TOC entry's bytes, fmt tag, matrix offsets
    descs = P.expert_descs(sl, entry_off_in_pool)                     # 3 x 64 B MatDesc (gate, up, down)

An INT4 matrix is 1,818,624 B a rank (444 x 4 KiB), an expert 5,455,872 B; a BF16 matrix 6,291,456 B, an expert
18,874,368 B. The weight of an INT4 element is bf16((u - u_z) * s) (the stored nibbles u = q + 8, u_z = z + 8):
compressed-tensors' dequantize, the product rounded once to nearest even (value_bits; docs/findings.md A0).

Checks:  python3 -I int4_pack.py selftest      (random tensors: vectorised packers vs a per-element port of
                                                 layout.h's readers; slices; descriptors)
         python3 -I int4_pack.py dump --ckpt DIR --layer L --experts 0,57 --rank 1 --out OUT
         python3 -I int4_pack.py dump-random --out OUT
dump writes, per matrix, the canonical slice (INT4: .wp .s .zp; BF16: .w), the packed blocks (.packed), the MatDesc
at off 0 (.desc) and the contract's values (.val, bf16 [N][K]) with index.tsv, for test_experts --real.
"""
import argparse, json, os, struct, sys

import numpy as np

FMT_NVFP4, FMT_S3_RETIRED, FMT_INT4, FMT_BF16 = 0, 1, 2, 3
KNOWN = (FMT_NVFP4, FMT_INT4, FMT_BF16)
BLK = {FMT_NVFP4: 2304, FMT_INT4: 2368, FMT_BF16: 2048}  # experts.h kQ4Blk, kI4Blk, kB16Blk
BLK_K = {FMT_NVFP4: 512, FMT_INT4: 512, FMT_BF16: 128}
NAME = {FMT_NVFP4: "NVFP4", FMT_INT4: "INT4", FMT_BF16: "BF16"}
KH, KI, TP, INTER, GROUP = 6144, 512, 4, 2048, 32
PROJ = ("gate", "up", "down")
MATDESC = struct.Struct("<QfI12I")  # experts.h MatDesc: off, s2, fmt, rsv[12]
assert MATDESC.size == 64


def check_fmt(fmt):
    if fmt not in KNOWN:
        raise ValueError(f"expert format {fmt}: not NVFP4 (0), INT4 (2) or BF16 (3)")
    return fmt


def mat_bytes(fmt, N, K):
    check_fmt(fmt)
    return (N // 8) * (K // BLK_K[fmt]) * BLK[fmt]


# ---- canonical tensors (the checkpoint's) ------------------------------------------------------------------------

def canon_int4(weight_packed, weight_scale, weight_zero_point):
    """weight_packed int32 [N][K/8], weight_scale bf16 bits uint16 [N][K/32], weight_zero_point int32 [N/8][K/32]
    -> {fmt, N, K, wp: uint8 [N][K/2] (the words' little-endian bytes: byte k/2, low nibble = even k), s, zp}."""
    wp = np.ascontiguousarray(weight_packed)
    assert wp.dtype == np.int32 and wp.ndim == 2
    N, K = wp.shape[0], wp.shape[1] * 8
    s = np.ascontiguousarray(weight_scale).view(np.uint16)
    zp = np.ascontiguousarray(weight_zero_point)
    assert zp.dtype == np.int32 and s.shape == (N, K // GROUP) and zp.shape == (N // 8, K // GROUP), (s.shape, zp.shape)
    return {"fmt": FMT_INT4, "N": N, "K": K, "wp": wp.astype("<i4").view(np.uint8).reshape(N, K // 2), "s": s,
            "zp": zp.astype("<i4").view(np.uint32)}


def canon_bf16(w):
    """BF16 bits uint16 [N][K]."""
    w = np.ascontiguousarray(w).view(np.uint16)
    return {"fmt": FMT_BF16, "N": w.shape[0], "K": w.shape[1], "w": w}


def nibbles(wp):
    """uint8 [N][K/2] -> the stored value nibbles u [N][K] (uint8)."""
    return np.stack([wp & 15, wp >> 4], axis=-1).reshape(wp.shape[0], -1)


def zp_nibbles(zp, N):
    """uint32 [N/8][G] -> the stored zero-point nibbles u_z [N][G] (row 8 r + i at bits 4 i of word r)."""
    sh = (4 * np.arange(8, dtype=np.uint32))[None, :, None]
    return ((zp[:, None, :] >> sh) & 15).astype(np.uint8).reshape(N, zp.shape[1])


def bf16_round_bits(f):
    """f32 -> bf16 bits, round to nearest even (finite values)."""
    u = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bits_to_f32(b):
    return (np.ascontiguousarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def value_bits(c):
    """The contract's weight as bf16 bits [N][K]: INT4 bf16((u - u_z) * s) (the f32 product of the exact integer and
    the bf16 scale is exact, 12 significant bits at most, then one rounding); BF16 the values themselves."""
    if c["fmt"] == FMT_BF16:
        return c["w"].copy()
    check_fmt(c["fmt"])
    if c["fmt"] != FMT_INT4:
        raise ValueError("value_bits: INT4 or BF16 only")
    N, K = c["N"], c["K"]
    d = nibbles(c["wp"]).astype(np.int16) - np.repeat(zp_nibbles(c["zp"], N), GROUP, axis=1).astype(np.int16)
    return bf16_round_bits(d.astype(np.float32) * np.repeat(bits_to_f32(c["s"]), GROUP, axis=1))


def rank_slice(c, m, r):
    """Rank r of 4: gate/up (m 0, 1) rows [512r, +512); down (m 2) input columns [512r, +512). INT4: the zero points'
    packed rows [64r, +64) (gate/up) or groups [16r, +16) (down); every cut is group- and word-aligned."""
    i = INTER // TP
    out = {"fmt": c["fmt"]}
    if m < 2:
        assert (c["N"], c["K"]) == (INTER, KH), (c["N"], c["K"])
        rs = slice(i * r, i * (r + 1))
        out.update(N=i, K=KH)
        if c["fmt"] == FMT_INT4:
            out.update(wp=c["wp"][rs], s=c["s"][rs], zp=c["zp"][i // 8 * r:i // 8 * (r + 1)])
        else:
            out.update(w=c["w"][rs])
    else:
        assert (c["N"], c["K"]) == (KH, INTER), (c["N"], c["K"])
        out.update(N=KH, K=i)
        g = slice(i // GROUP * r, i // GROUP * (r + 1))
        if c["fmt"] == FMT_INT4:
            out.update(wp=c["wp"][:, i // 2 * r:i // 2 * (r + 1)], s=c["s"][:, g], zp=c["zp"][:, g])
        else:
            out.update(w=c["w"][:, i * r:i * (r + 1)])
    for k, v in list(out.items()):
        if isinstance(v, np.ndarray):
            out[k] = np.ascontiguousarray(v)
    return out


# ---- the engine layout (layout.h pack_int4, pack_bf16) ---------------------------------------------------------------

def pack_int4(wp, s, zp, N, K):
    """layout.h pack_int4, vectorised: [N/8][K/512] blocks of 2368 B."""
    B, N8 = K // 512, N // 8
    # values: wp bytes of (n = 8 n8 + g, k = 512 b + 128 jj + 64 q + 16 t + 2 x) -> block byte 512 jj + 16 (4 g + t) + 8 q + x
    v = wp.reshape(N8, 8, B, 4, 2, 4, 8).transpose(0, 2, 3, 1, 5, 4, 6).reshape(N8, B, 2048)
    # scales: group c' = 4 jj + 2 q + h = 2 j + h of the block -> 2048 + 32 g + 16 h + 2 j
    sc = s.reshape(N8, 8, B, 8, 2).transpose(0, 2, 1, 4, 3).reshape(N8, B, 128).astype("<u2").view(np.uint8).reshape(N8, B, 256)
    # zero points: nibble j of the u32 at 2304 + 8 g + 4 h, low nibble first
    z = zp_nibbles(zp, N).reshape(N8, 8, B, 8, 2).transpose(0, 2, 1, 4, 3)  # [N8, B, g, h, j]
    zb = (z[..., 0::2] | (z[..., 1::2] << 4)).astype(np.uint8).reshape(N8, B, 64)
    return np.concatenate([v, sc, zb], axis=2).reshape(-1)


def pack_bf16(w, N, K):
    """layout.h pack_bf16, vectorised: [N/8][K/128] blocks of 2048 B; lane 4 g + t's 16 B at 512 p + 16 l hold
    k = 128 b + 64 (p >> 1) + 16 t + 8 (p & 1) + 0..7 of row 8 n8 + g."""
    B, N8 = K // 128, N // 8
    x = w.reshape(N8, 8, B, 2, 4, 2, 8)  # n8, g, b, q, t, half, i
    return x.transpose(0, 2, 3, 5, 1, 4, 6).astype("<u2").reshape(-1).view(np.uint8)


def pack(sl):
    f = check_fmt(sl["fmt"])
    if f == FMT_INT4:
        b = pack_int4(sl["wp"], sl["s"], sl["zp"], sl["N"], sl["K"])
    elif f == FMT_BF16:
        b = pack_bf16(sl["w"], sl["N"], sl["K"])
    else:
        raise ValueError("pack: NVFP4 is packed by experts_fill.py / layout.h pack_nvfp4")
    assert b.size == mat_bytes(f, sl["N"], sl["K"]), (b.size, mat_bytes(f, sl["N"], sl["K"]))
    return b


def matdesc(off, fmt, s2=1.0):
    """experts.h MatDesc (64 B): off (a multiple of 256), s2 (1.0 for INT4 and BF16), fmt, 12 zero words."""
    check_fmt(fmt)
    assert off % 256 == 0
    return MATDESC.pack(off, s2 if fmt == FMT_NVFP4 else 1.0, fmt, *([0] * 12))


def expert_entry(slices):
    """(entry bytes, fmt tag gate | up << 4 | down << 8, offsets of the three matrices in the entry). A layer is
    single-format: the three matrices must share it."""
    fm = {sl["fmt"] for sl in slices}
    if len(fm) != 1:
        raise ValueError(f"expert_entry: mixed formats {sorted(fm)} (a layer has one expert format)")
    parts, moff, off = [], [], 0
    for sl in slices:
        b = pack(sl)
        moff.append(off)
        parts.append(b)
        off += b.size
    tag = slices[0]["fmt"] | slices[1]["fmt"] << 4 | slices[2]["fmt"] << 8
    return np.concatenate(parts), tag, moff


def expert_descs(slices, entry_off):
    """3 x 64 B MatDesc for an entry at entry_off in the layer's pool (gate, up, down back to back)."""
    out, off = [], entry_off
    for sl in slices:
        out.append(matdesc(off, sl["fmt"]))
        off += mat_bytes(sl["fmt"], sl["N"], sl["K"])
    return b"".join(out)


# ---- independent readers (per element, a port of layout.h read_i4 / read_b16) ---------------------------------------

def read_packed(packed, fmt, K, n, k):
    """bf16 bits of element (n, k) of a packed matrix."""
    if fmt == FMT_INT4:
        B, n8, g, b = K // 512, n // 8, n % 8, k // 512
        jj, q, t, i = (k // 128) % 4, (k // 64) % 2, (k // 16) % 4, k % 16
        l, h, j = 4 * g + t, t >> 1, 2 * jj + q
        base = (n8 * B + b) * 2368
        u = (int(packed[base + 512 * jj + 16 * l + 8 * q + i // 2]) >> (4 * (i & 1))) & 15
        s = int(packed[base + 2048 + 32 * g + 16 * h + 2 * j]) | int(packed[base + 2048 + 32 * g + 16 * h + 2 * j + 1]) << 8
        uz = (int(packed[base + 2304 + 8 * g + 4 * h + j // 2]) >> (4 * (j & 1))) & 15
        f = np.float32(u - uz) * bits_to_f32(np.array([s], np.uint16))[0]
        return int(bf16_round_bits(np.array([f], np.float32))[0])
    if fmt == FMT_BF16:
        B, n8, g, b, kk = K // 128, n // 8, n % 8, k // 128, k % 128
        q, t, rr = kk // 64, (kk % 64) // 16, kk % 16
        p, l = 2 * q + rr // 8, 4 * g + t
        o = (n8 * B + b) * 2048 + 512 * p + 16 * l + 2 * (rr % 8)
        return int(packed[o]) | int(packed[o + 1]) << 8
    raise ValueError(f"read_packed: format {fmt}")


# ---- checks ----------------------------------------------------------------------------------------------------------

def random_int4(N, K, rng):
    wp = rng.integers(-2**31, 2**31, size=(N, K // 8), dtype=np.int64).astype(np.int32)
    zp = rng.integers(-2**31, 2**31, size=(N // 8, K // GROUP), dtype=np.int64).astype(np.int32)
    # scales: positive bf16 over many binades (2^-20 .. 2^4), some with a full mantissa
    e = rng.integers(107, 131, size=(N, K // GROUP)).astype(np.uint16)
    s = (e << 7) | rng.integers(0, 128, size=(N, K // GROUP)).astype(np.uint16)
    return canon_int4(wp, s, zp)


def random_bf16(N, K, rng):
    e = rng.integers(100, 135, size=(N, K)).astype(np.uint16)
    return canon_bf16((rng.integers(0, 2, size=(N, K)).astype(np.uint16) << 15) | (e << 7) | rng.integers(0, 128, size=(N, K)).astype(np.uint16))


def selftest():
    rng = np.random.default_rng(20261014)
    bad = 0
    for fmt in (FMT_INT4, FMT_BF16):
        full = [random_int4(INTER, KH, rng), random_int4(INTER, KH, rng), random_int4(KH, INTER, rng)] if fmt == FMT_INT4 else \
               [random_bf16(INTER, KH, rng), random_bf16(INTER, KH, rng), random_bf16(KH, INTER, rng)]
        vfull = [value_bits(c) for c in full]
        for r in range(TP):
            sl = [rank_slice(full[m], m, r) for m in range(3)]
            entry, tag, moff = expert_entry(sl)
            assert tag == fmt | fmt << 4 | fmt << 8 and entry.size == sum(mat_bytes(fmt, s["N"], s["K"]) for s in sl)
            for m in range(3):
                vs = value_bits(sl[m])
                want = vfull[m][512 * r:512 * (r + 1)] if m < 2 else vfull[m][:, 512 * r:512 * (r + 1)]
                bad += int(np.count_nonzero(vs != want))  # a slice's values are the full tensor's
                pk = entry[moff[m]:moff[m] + mat_bytes(fmt, sl[m]["N"], sl[m]["K"])]
                N, K = sl[m]["N"], sl[m]["K"]
                pts = [(0, 0), (N - 1, K - 1), (7, 31), (8, 32), (N // 2 + 3, K // 2 + 17)] + \
                      [(int(rng.integers(N)), int(rng.integers(K))) for _ in range(3000)]
                for n, k in pts:
                    bad += read_packed(pk, fmt, K, n, k) != int(vs[n, k])
            d = expert_descs(sl, 4096 * 7)
            for m in range(3):
                off, s2, f, *rsv = MATDESC.unpack(d[64 * m:64 * m + 64])
                bad += off != 4096 * 7 + moff[m] or s2 != 1.0 or f != fmt or any(rsv)
        print(f"{NAME[fmt]}: 4 ranks x 3 matrices, slices = full tensor values, {3 * 4 * 3005} packed values read back")
    # the rounding: an exact product with more than 8 significant bits rounds to nearest even
    c = canon_int4(np.array([[0x0000000F] * (KH // 8)] * 8, np.int32), np.full((8, KH // GROUP), 0x3F81, np.uint16),
                   np.zeros((1, KH // GROUP), np.int32))
    v = value_bits(c)  # u = 15, u_z = 0, s = 1 + 2^-7: 15 + 15 * 2^-7 = 15.1171875 -> bf16 15.125 (0x4172)
    bad += int(v[0, 0] != 0x4172) + int(v[0, 1] != 0)
    for f in (1, 4, 0xFFFFFFFF):
        try:
            mat_bytes(f, 8, 512)
            bad += 1
        except ValueError:
            pass
    print("selftest:", "ok" if bad == 0 else f"{bad} bad")
    return bad


# ---- dumps for test_experts --real ---------------------------------------------------------------------------------

ST_DT = {"I32": np.int32, "BF16": np.uint16, "I64": np.int64, "F32": np.float32}


class Shards:
    """The checkpoint's tensors by name through model.safetensors.index.json, read with pread (no mmap)."""

    def __init__(self, d):
        self.d = d
        with open(os.path.join(d, "model.safetensors.index.json")) as f:
            self.where = json.load(f)["weight_map"]
        self.hdr = {}

    def header(self, fn):
        if fn not in self.hdr:
            with open(os.path.join(self.d, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.hdr[fn] = (8 + n, json.loads(f.read(n)))
        return self.hdr[fn]

    def get(self, name):
        fn = self.where[name]
        base, h = self.header(fn)
        e = h[name]
        a, b = e["data_offsets"]
        fd = os.open(os.path.join(self.d, fn), os.O_RDONLY)
        try:
            buf = os.pread(fd, b - a, base + a)
        finally:
            os.close(fd)
        assert len(buf) == b - a
        return np.frombuffer(buf, dtype=ST_DT[e["dtype"]]).reshape(e["shape"])

    def has(self, name):
        return name in self.where


def load_expert(S, layer, e):
    """The three canonical tensors of expert e of a layer (INT4 if weight_packed exists, else BF16)."""
    out = []
    for p in PROJ:
        n = f"model.layers.{layer}.mlp.experts.{e}.{p}_proj."
        if S.has(n + "weight_packed"):
            out.append(canon_int4(S.get(n + "weight_packed"), S.get(n + "weight_scale"), S.get(n + "weight_zero_point")))
            shp = S.get(n + "weight_shape")
            assert tuple(int(x) for x in shp) == (out[-1]["N"], out[-1]["K"]), shp
        else:
            out.append(canon_bf16(S.get(n + "weight")))
    return out


def write_matrix(out, prefix, sl):
    p = os.path.join(out, prefix)
    for k in ("wp", "s", "zp", "w"):
        if k in sl:
            sl[k].tofile(p + "." + k)
    pack(sl).tofile(p + ".packed")
    with open(p + ".desc", "wb") as f:
        f.write(matdesc(0, sl["fmt"]))
    value_bits(sl).tofile(p + ".val")
    return f"M {PROJ[int(prefix[-1])]} {sl['fmt']} 3f800000 {sl['N']} {sl['K']} {prefix}\n"


def dump(args):
    S = Shards(args.ckpt)
    os.makedirs(args.out, exist_ok=True)
    lines = []
    for e in [int(x) for x in args.experts.split(",")]:
        c = load_expert(S, args.layer, e)
        lines.append(f"S L{args.layer} {e}\n")
        for m in range(3):
            lines.append(write_matrix(args.out, f"l{args.layer}e{e}r{args.rank}m{m}", rank_slice(c[m], m, args.rank)))
    with open(os.path.join(args.out, "index.tsv"), "w") as f:
        f.writelines(lines)
    print(f"{len(lines)} lines in {args.out}/index.tsv")


def dump_random(args):
    rng = np.random.default_rng(7)
    os.makedirs(args.out, exist_ok=True)
    lines = []
    for e, fmt in enumerate((FMT_INT4, FMT_INT4, FMT_BF16)):
        mk = random_int4 if fmt == FMT_INT4 else random_bf16
        c = [mk(INTER, KH, rng), mk(INTER, KH, rng), mk(KH, INTER, rng)]
        lines.append(f"S random {e}\n")
        for m in range(3):
            lines.append(write_matrix(args.out, f"rnd{e}r2m{m}", rank_slice(c[m], m, 2)))
    with open(os.path.join(args.out, "index.tsv"), "w") as f:
        f.writelines(lines)
    print(f"{len(lines)} lines in {args.out}/index.tsv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    d = sub.add_parser("dump")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--layer", type=int, required=True)
    d.add_argument("--experts", default="0")
    d.add_argument("--rank", type=int, default=0)
    d.add_argument("--out", required=True)
    r = sub.add_parser("dump-random")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(1 if selftest() else 0)
    if a.cmd == "dump":
        dump(a)
    else:
        dump_random(a)


if __name__ == "__main__":
    main()
