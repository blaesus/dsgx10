#!/usr/bin/env python3
"""S3x2 routed experts for the converter: quantise, cut a rank's slice, pack into the expert kernels' blocks,
build the descriptors (docs/experts.md section 13; engine/experts/layout.h is the specification this ports).

S3x2 is quant/formats.py's levelset3 with the two measured subsets A = {0.5, 2, 4, 6}, B = {0.5, 1.5, 3, 6}
(docs/quantiser.md Q1): per group of 16 along K the subset (bit 7 of the group's scale byte) and the e4m3 scale
(bits 6:0) of least squared error, s2 = f32(amax) / f32(2688) over the whole tensor, codes c = sign << 2 | j
packed 16 to a little-endian 48-bit word. This module calls levelset3 itself, so its canonical tensors are the
quantiser's bytes; it then only re-tiles them (lossless) and writes MatDesc tables.

What the converter calls (experts_fill.py, for an S3 fill):

    import s3x2_pack as S
    q = S.quantise_expert(W_gate, W_up, W_down, recipe="mix")  # [(fmt, canon)] x 3, full tensors (f32 torch)
    sl = [S.rank_slice(f, c, m, r) for m, (f, c) in enumerate(q)]  # rank r's gate/up rows, down columns
    entry, tag, moff = S.expert_entry(sl)                          # bytes of the TOC entry, fmt tag, matrix offsets
    descs = S.expert_descs(sl, pool_off_of_entry)                  # 3 x 64 B MatDesc (gate, up, down)

recipes: "mix" (S3x2 gate/up + NVFP4 down, the main track), "s3x2" (S3x2 on all three, the bracket),
"nvfp4" (NVIDIA's recipe on all three, gate/up sharing amax). An S3x2 matrix is 1,376,256 B a rank, NVFP4
1,769,472 B; a mix expert 4,521,984 B, an S3x2 one 4,128,768 B (all 4 KiB multiples).

Checks:  python -I s3x2_pack.py selftest           (random and Gaussian tensors; a per-element port of layout.h)
         python -I s3x2_pack.py dump --master M --layer 3 --experts 0,51 --rank 1 --out DIR [--recipes mix,s3x2]
         python -I s3x2_pack.py dump-random --out DIR
dump writes, per matrix, the canonical slice (.w, .s), the packed blocks (.packed), the MatDesc at off 0 (.desc)
and the torch model's values without s2 (.val, bf16: exact) for engine/experts/tests/test_experts.cu "real", which
checks them byte for byte against layout.h's packers and readers and the GPU decode, then runs the kernels on them.
"""
import argparse, hashlib, json, os, struct, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
QUANT = os.path.normpath(os.path.join(ROOT, "..", "..", "quant"))
if QUANT not in sys.path:
    sys.path.insert(0, QUANT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import formats as F  # noqa: E402  (quant/formats.py)

SUBSETS = ((0.5, 2.0, 4.0, 6.0), (0.5, 1.5, 3.0, 6.0))  # experts.h kS3x2Mags: A (bit 7 clear), B (set)
FMT_NVFP4, FMT_S3 = 0, 1
BLK = {FMT_NVFP4: 2304, FMT_S3: 1792}
KH, KI = 6144, 512
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
PROJ = ("gate", "up", "down")


def mat_bytes(fmt, N, K):
    return (N // 8) * (K // 512) * BLK[fmt]


# ---- quantisation (the quantiser's own functions) -------------------------------------------------------------

def quantise(w, fmt, amax=None):
    """w: f32 torch [N][K] (a full tensor). S3: levelset3 with SUBSETS (s2 from this tensor's amax); NVFP4: nvfp4_rtn
    (amax: gate and up share max(amax_gate, amax_up))."""
    if fmt == FMT_S3:
        return F.levelset3(w, [SUBSETS[0], SUBSETS[1]])
    return F.nvfp4_rtn(w, amax)


RECIPES = {"mix": (FMT_S3, FMT_S3, FMT_NVFP4), "s3x2": (FMT_S3, FMT_S3, FMT_S3), "nvfp4": (FMT_NVFP4,) * 3}


def quantise_expert(gate, up, down, recipe="mix"):
    fm = RECIPES[recipe]
    gu = torch.maximum(gate.abs().max(), up.abs().max())
    return [(f, quantise(w, f, gu if (f == FMT_NVFP4 and m < 2) else None)) for m, (f, w) in enumerate(zip(fm, (gate, up, down)))]


def canon_arrays(fmt, c):
    """(w, s, s2): S3 codes [N][K/16][6], scales [N][K/16], s2; NVFP4 nibbles [N][K/2], scales [N][K/16], s2."""
    if fmt == FMT_S3:
        return c["codes"], c["scale"], np.float32(c["scale_2"])
    return c["weight"], c["weight_scale"], np.float32(c["weight_scale_2"])


def rank_slice(fmt, c, m, r):
    """Rank r of 4: gate/up rows [512r, +512); down input columns [512r, +512) (S3: groups [32r, +32); NVFP4:
    value bytes [256r, +256), scale columns [32r, +32))."""
    w, s, s2 = canon_arrays(fmt, c)
    if m == 2:
        if fmt == FMT_S3:
            w, s = w[:, 32 * r:32 * r + 32], s[:, 32 * r:32 * r + 32]
        else:
            w, s = w[:, 256 * r:256 * r + 256], s[:, 32 * r:32 * r + 32]
    else:
        w, s = w[KI * r:KI * r + KI], s[KI * r:KI * r + KI]
    return {"fmt": fmt, "w": np.ascontiguousarray(w), "s": np.ascontiguousarray(s), "s2": np.float32(s2)}


# ---- the torch model of the formats: values without s2 (exact in f32 and in bf16) --------------------------------

def value_nos2(sl):
    if sl["fmt"] == FMT_S3:
        p = {"codes": sl["w"], "scale": sl["s"], "scale_2": np.float32(1.0), "subsets": np.array(SUBSETS, np.float32),
             "g": np.array(16)}
        return F.levelset3_dequant(p)
    return F.nvfp4_dequant({"weight": sl["w"], "weight_scale": sl["s"], "weight_scale_2": np.float32(1.0)})


# ---- packing (layout.h pack_s3 / pack_nvfp4) ---------------------------------------------------------------------

def pack_s3(codes, scale):
    """layout.h pack_s3: codes [N][K/16][6], scale [N][K/16] -> [N/8][K/512][1792] bytes. In block (n8, b), lane
    l = 4g + t (row 8 n8 + g), unit jj: v[16q + i] = code of k = 512b + 128jj + 64q + 16t + i, i.e. group
    32b + 8jj + 4q + t, element i. P0 nibble m = (v[m] & 3) | (v[m+8] & 3) << 2, P1 the same for v[16+m], v[24+m],
    Q nibble m = the sign bits of v[m], v[m+8], v[m+16], v[m+24] in bits 0-3; word jj of plane p at
    512p + 16l + 4jj (little endian); scales at 1536 + 8l + 2jj + q = scale of group 32b + 8jj + 4q + t."""
    N, G, _ = codes.shape
    B = G * 16 // 512
    c = F.unpack3(codes).reshape(N // 8, 8, B, 4, 2, 4, 16)  # n8, g, b, jj, q, t, i
    lo = (c & 3).astype(np.uint32)
    hb = (c >> 2).astype(np.uint32)
    sh = (4 * np.arange(8, dtype=np.uint32))

    def orr(x):  # x [..., 8] nibbles -> word
        return np.bitwise_or.reduce(x << sh, axis=-1)

    P0 = orr(lo[:, :, :, :, 0, :, 0:8] | (lo[:, :, :, :, 0, :, 8:16] << 2))
    P1 = orr(lo[:, :, :, :, 1, :, 0:8] | (lo[:, :, :, :, 1, :, 8:16] << 2))
    Q = orr(hb[:, :, :, :, 0, :, 0:8] | (hb[:, :, :, :, 0, :, 8:16] << 1) | (hb[:, :, :, :, 1, :, 0:8] << 2) |
            (hb[:, :, :, :, 1, :, 8:16] << 3))
    planes = np.stack([P0, P1, Q], 1)  # n8, plane, g, b, jj, t
    planes = planes.transpose(0, 3, 1, 2, 5, 4)  # n8, b, plane, g, t, jj
    pb = np.ascontiguousarray(planes).astype("<u4").view(np.uint8).reshape(N // 8, B, 1536)
    sc = scale.reshape(N // 8, 8, B, 4, 2, 4).transpose(0, 2, 1, 5, 3, 4).reshape(N // 8, B, 256)
    return np.ascontiguousarray(np.concatenate([pb, sc], axis=2)).reshape(-1)


def pack_nvfp4(w, s):
    """layout.h pack_nvfp4 (experts_fill.py's port, repeated here so this module stands alone): w [N][K/2],
    s [N][K/16] -> [N/8][K/512][2304]."""
    N, Kb = w.shape
    B = 2 * Kb // 512
    v = w.reshape(N // 8, 8, B, 4, 2, 4, 8).transpose(0, 2, 3, 1, 5, 4, 6).reshape(N // 8, B, 2048)
    sc = s.reshape(N // 8, 8, B, 4, 2, 4).transpose(0, 2, 1, 5, 3, 4).reshape(N // 8, B, 256)
    return np.ascontiguousarray(np.concatenate([v, sc], axis=2)).reshape(-1)


def pack(sl):
    return pack_s3(sl["w"], sl["s"]) if sl["fmt"] == FMT_S3 else pack_nvfp4(sl["w"], sl["s"])


# ---- descriptors (layout.h make_desc / formats.cuh s3_table, s3_nibble_table) -------------------------------------

def _bf16_bits(x):
    return int(np.array([x], np.float32).view(np.uint32)[0] >> 16)  # exact for E2M1 values


def s3_tables(mags):
    tl0 = tl1 = th0 = th1 = nib = 0
    for c in range(8):
        v = (-1.0 if c & 4 else 1.0) * mags[c & 3]
        b = _bf16_bits(v)
        lo, hi = b & 0xFF, b >> 8
        if c < 4:
            tl0 |= lo << 8 * c
            th0 |= hi << 8 * c
        else:
            tl1 |= lo << 8 * (c - 4)
            th1 |= hi << 8 * (c - 4)
        nib |= ((8 if c & 4 else 0) | E2M1_GRID.index(mags[c & 3])) << (4 * c)
    return (tl0, tl1, th0, th1), nib


def matdesc(off, s2, fmt, subsets=SUBSETS):
    """gx::MatDesc (64 B): off u64, s2 f32, fmt u32, tab[8], nib[2], pad[2]. S3: both subsets' tables."""
    tab, nib = [0] * 8, [0, 0]
    if fmt == FMT_S3:
        for s in range(2):
            t, n = s3_tables(subsets[s if s < len(subsets) else 0])
            tab[4 * s:4 * s + 4] = t
            nib[s] = n
    return struct.pack("<QfI8I2I2I", off, float(np.float32(s2)), fmt, *tab, *nib, 0, 0)


def expert_entry(slices):
    """(entry bytes, fmt tag gate | up << 4 | down << 8, offsets of the three matrices in the entry)."""
    parts, moff, off = [], [], 0
    for sl in slices:
        b = pack(sl)
        moff.append(off)
        parts.append(b)
        off += b.size
    tag = slices[0]["fmt"] | slices[1]["fmt"] << 4 | slices[2]["fmt"] << 8
    return np.concatenate(parts), tag, moff


def expert_descs(slices, entry_off):
    """The expert's three MatDescs; entry_off: its entry's offset in the layer's pool (expert 0's entry = 0)."""
    out, off = [], entry_off
    for sl in slices:
        N, K = (sl["w"].shape[0], sl["s"].shape[1] * 16)
        out.append(matdesc(off, sl["s2"], sl["fmt"]))
        off += mat_bytes(sl["fmt"], N, K)
    return b"".join(out)


# ---- an independent per-element reader (layout.h read_s3 / read_q4), for the self-test -----------------------------

def read_packed(packed, fmt, K, n, k):
    B, n8, g, b = K // 512, n // 8, n % 8, k // 512
    jj, q, t, i = (k // 128) % 4, (k // 64) % 2, (k // 16) % 4, k % 16
    l = 4 * g + t
    blk = packed[(n8 * B + b) * BLK[fmt]:(n8 * B + b + 1) * BLK[fmt]]
    if fmt == FMT_NVFP4:
        nib = (int(blk[512 * jj + 16 * l + 8 * q + i // 2]) >> (4 * (i & 1))) & 15
        sc = int(blk[2048 + 8 * l + 2 * jj + q])
        return F.E2M1[nib & 7] * (-1 if nib & 8 else 1) * e4m3_f(sc)
    word = lambda p: int.from_bytes(bytes(blk[512 * p + 16 * l + 4 * jj:512 * p + 16 * l + 4 * jj + 4]), "little")
    P0, P1, Q = word(0), word(1), word(2)
    idx = 16 * q + i
    mm, hf = idx % 8, (idx // 8) % 2
    low2 = ((P0 if idx < 16 else P1) >> (4 * mm + 2 * hf)) & 3
    hb = (Q >> (4 * mm + idx // 8)) & 1
    sc = int(blk[1536 + 8 * l + 2 * jj + q])
    return SUBSETS[sc >> 7][low2] * (-1 if hb else 1) * e4m3_f(sc & 0x7F)


def e4m3_f(b):
    return float(torch.tensor([b], dtype=torch.uint8).view(torch.float8_e4m3fn).float()[0])


# ---- self-test and dumps ------------------------------------------------------------------------------------------

def random_s3(N, K, rng):
    G = N * K // 16
    codes = rng.integers(0, 256, size=(N, K // 16, 6), dtype=np.uint8)
    sc = (0x20 + rng.integers(0, 0x30, size=(N, K // 16))).astype(np.uint8) | (rng.integers(0, 2, size=(N, K // 16)) << 7).astype(np.uint8)
    return {"fmt": FMT_S3, "w": codes, "s": sc, "s2": np.float32(1 / 192)}


def random_nvfp4(N, K, rng):
    return {"fmt": FMT_NVFP4, "w": rng.integers(0, 256, size=(N, K // 2), dtype=np.uint8),
            "s": (0x20 + rng.integers(0, 0x30, size=(N, K // 16))).astype(np.uint8), "s2": np.float32(1 / 64)}


def selftest():
    rng = np.random.default_rng(7)
    bad = 0
    # 1. the vectorised packers against the per-element reader (layout.h's read_s3 / read_q4, ported) on random bits
    for mk in (random_s3, random_nvfp4):
        sl = mk(16, 1024, rng)
        p = pack(sl)
        v = value_nos2(sl).numpy()
        nb = sum(read_packed(p, sl["fmt"], 1024, n, k) != v[n, k] for n in range(16) for k in range(1024))
        print(f"packer + reader vs the torch model, random {('nvfp4', 's3')[sl['fmt']]} [16][1024]: {nb} differ")
        bad += nb
    # 2. Gaussian tensors through levelset3 / nvfp4_rtn: rank slices of the full tensor equal slices of its values
    w = torch.randn(2048, 6144, generator=torch.Generator().manual_seed(3)) * 0.02
    d = torch.randn(6144, 2048, generator=torch.Generator().manual_seed(4)) * 0.02
    q = quantise_expert(w, w * 0.5, d, "mix") + quantise_expert(w, w, d, "s3x2")[2:]
    fulls = [(f, c, m if m < 3 else 2) for m, (f, c) in enumerate(q)]
    for f, c, m in fulls:
        full = value_nos2({"fmt": f, "w": canon_arrays(f, c)[0], "s": canon_arrays(f, c)[1]}).numpy()
        for r in range(4):
            sl = rank_slice(f, c, m, r)
            ref = full[:, 512 * r:512 * r + 512] if m == 2 else full[512 * r:512 * r + 512]
            nb = int((value_nos2(sl).numpy() != ref).sum())
            p = pack(sl)
            K = sl["s"].shape[1] * 16
            assert p.size == mat_bytes(f, sl["w"].shape[0], K)
            pts = rng.integers(0, [sl["w"].shape[0], K], size=(2000, 2))
            nb += sum(read_packed(p, f, K, int(n), int(k)) != ref[n, k] for n, k in pts)
            bad += nb
        print(f"{PROJ[m]} {('nvfp4', 's3x2')[f]}: 4 rank slices = slices of the full tensor's values, 8,000 packed "
              f"values read back: {'ok' if not bad else bad}")
    # 3. the descriptor bytes: tables decode to the subsets
    dsc = matdesc(4096, np.float32(0.001), FMT_S3)
    tab = struct.unpack("<8I", dsc[16:48])
    for s in range(2):
        for j in range(4):
            b = ((tab[4 * s + 2] >> 8 * j) & 0xFF) << 8 | ((tab[4 * s] >> 8 * j) & 0xFF)
            v = np.array([b << 16], np.uint32).view(np.float32)[0]
            bad += v != SUBSETS[s][j]
    print(f"selftest: {'PASS' if not bad else 'FAIL'} ({bad} bad)")
    return bad == 0


def write_matrix(out, prefix, sl):
    K = sl["s"].shape[1] * 16
    N = sl["w"].shape[0]
    p = pack(sl)
    val = value_nos2(sl).to(torch.bfloat16)
    assert torch.equal(val.float(), value_nos2(sl)), "value not exact in bf16"
    for ext, data in (("w", sl["w"]), ("s", sl["s"]), ("packed", p), ("desc", np.frombuffer(matdesc(0, sl["s2"], sl["fmt"]), np.uint8)),
                      ("val", val.view(torch.int16).numpy())):
        np.ascontiguousarray(data).tofile(os.path.join(out, f"{prefix}.{ext}"))
    s2 = int(np.array([sl["s2"]], np.float32).view(np.uint32)[0])
    return f"{sl['fmt']}\t{s2:08x}\t{N}\t{K}\t{prefix}", hashlib.sha256(p.tobytes()).hexdigest()


def load_master(master, layer, e):
    man = {}
    for line in open(os.path.join(master, "manifest.jsonl")):
        x = json.loads(line)
        man[x["name"]] = x
    W = []
    for proj in PROJ:
        n = f"model.layers.{layer}.mlp.experts.{e}.{proj}_proj.weight"
        buf = open(os.path.join(master, "tensors", n + ".bin"), "rb").read()
        if hashlib.sha256(buf).hexdigest() != man[n]["sha256"]:
            raise SystemExit(f"{n}: SHA-256 differs from the fetch record")
        W.append(torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(man[n]["shape"]).float())
    return W


def dump(args):
    os.makedirs(args.out, exist_ok=True)
    idx = open(os.path.join(args.out, "index.tsv"), "w")
    rec = {"layer": args.layer, "rank": args.rank, "recipes": args.recipes, "subsets": SUBSETS, "torch": torch.__version__,
           "numpy": np.__version__, "formats_sha256": hashlib.sha256(open(os.path.join(QUANT, "formats.py"), "rb").read()).hexdigest(),
           "packed_sha256": {}}
    t0 = time.time()
    for e in [int(x) for x in args.experts.split(",")]:
        W = load_master(args.master, args.layer, e) if args.master else None
        for recipe in args.recipes.split(","):
            q = quantise_expert(*W, recipe)
            idx.write(f"S\t{recipe}\t{e}\n")
            for m, (f, c) in enumerate(q):
                sl = rank_slice(f, c, m, args.rank)
                prefix = f"L{args.layer}.r{args.rank}.e{e}.{PROJ[m]}.{('nvfp4', 's3x2')[f]}"
                line, h = write_matrix(args.out, prefix, sl)
                idx.write(f"M\t{PROJ[m]}\t{line}\n")
                rec["packed_sha256"][prefix] = h
                if f == FMT_S3:
                    nb = int((sl["s"] >> 7).sum())
                    rec.setdefault("subset_b_share", {})[prefix] = nb / sl["s"].size
            print(f"layer {args.layer} expert {e} {recipe}: {time.time() - t0:.1f} s", flush=True)
    idx.close()
    json.dump(rec, open(os.path.join(args.out, "record.json"), "w"), indent=1)


def dump_random(args):
    """Random canonical tensors (all code and scale bits, both subsets) in the three recipes, as stored experts."""
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(11)
    idx = open(os.path.join(args.out, "index.tsv"), "w")
    for e in range(4):
        for recipe in ("mix", "s3x2"):
            idx.write(f"S\t{recipe}\t{e}\n")
            for m, f in enumerate(RECIPES[recipe]):
                N, K = (KI, KH) if m < 2 else (KH, KI)
                sl = (random_s3 if f == FMT_S3 else random_nvfp4)(N, K, rng)
                sl["s2"] = np.float32(1 / 192 if m < 2 else 1 / 64)
                line, _ = write_matrix(args.out, f"rnd.e{e}.{recipe}.{PROJ[m]}", sl)
                idx.write(f"M\t{PROJ[m]}\t{line}\n")
    idx.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    d = sub.add_parser("dump")
    d.add_argument("--master", required=True)
    d.add_argument("--layer", type=int, required=True)
    d.add_argument("--experts", required=True)
    d.add_argument("--rank", type=int, default=0)
    d.add_argument("--recipes", default="mix,s3x2")
    d.add_argument("--out", required=True)
    d.add_argument("--threads", type=int, default=4)
    r = sub.add_parser("dump-random")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(getattr(a, "threads", 4))
    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    if a.cmd == "dump":
        dump(a)
    else:
        dump_random(a)


if __name__ == "__main__":
    main()
