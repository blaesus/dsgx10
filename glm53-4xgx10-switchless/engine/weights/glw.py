"""The .glw container (glw.h, mirrored field by field), the conversion plan, and the per-unit builders.

The plan is a list of units, each a group of source tensors converted together (one layer's attention, one
layer's MLP, ...). A unit lists its sources (where each comes from, whether it is quantised to FP8), the output
tensors every rank gets (name, dtype, natural shape, layout, cut), and builds a rank's bytes from the loaded
sources. Every FP8 re-cut is engine/attn/tools/pack.py's own function (the byte-for-byte reference for the
attention kernels' layouts); the quantiser is quant.fp8_rtn (zai's rule, reproduced bit for bit). Shapes and
sizes come from the plan alone, so --plan needs only the checkpoint's headers. docs/layout.md has the table.
"""
import ctypes, hashlib, os, struct, subprocess, sys, tempfile

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ATTN_TOOLS = os.path.normpath(os.path.join(ROOT, "..", "attn", "tools"))
if ATTN_TOOLS not in sys.path:
    sys.path.insert(0, ATTN_TOOLS)
import pack  # noqa: E402  (engine/attn/tools/pack.py)

MAGIC = b"GLM53W01"
VERSION = 1
ALIGN = 4096
GLOBAL = 0xFFFFFFFF
TP, D, VOCAB = 4, 6144, 154880
N_LAYERS, MTP = 78, 78
DENSE_INTER, SHARED_INTER, N_EXPERTS, EXPERT_INTER = 12288, 2048, 256, 2048

# glw.h enums
F32, BF16, F8, U8, U32 = 1, 2, 3, 4, 5
DTYPE_NAME = {F32: "F32", BF16: "BF16", F8: "F8_E4M3", U8: "U8", U32: "U32"}
DTYPE_BYTES = {F32: 4, BF16: 2, F8: 1, U8: 1, U32: 4}
REPLICATED, ROWS, COLS, RECUT, GENERATED, RESERVED = 1, 2, 3, 4, 5, 6
KIND_NAME = {1: "replicated", 2: "rows", 3: "cols", 4: "recut", 5: "generated", 6: "reserved"}
NATURAL, T64, T128, EXPERT_POOL = 0, 1, 2, 3
LAYOUT_NAME = {0: "natural", 1: "T64", 2: "T128", 3: "expert-pool"}
FMT_NVFP4, FMT_S3, FMT_NONE = 0, 1, 0xFFFFFFFF
LOSSY, RESERVED_FLAG, SCALE, INTERLEAVED = 1, 2, 4, 8

# Header: magic, version, header_bytes, rank, world, n_tensors, toc_entry_bytes, toc_offset, data_offset,
# data_bytes, file_bytes, alignment, toc_crc32c, model[64], source[256], created[32], tool[64]
HDR = struct.Struct("<8sIIIIII QQQQ II 64s256s32s64s")
# TocEntry: name[96], source[192], slice[96], dtype, ndim, shape[4], offset, bytes, kind, layout, crc32c, layer,
# blk_rows, blk_k, fmt, flags, rel_err, pad0, reserve_bytes
ENT = struct.Struct("<96s192s96s II 4Q QQ IIII IIII f I Q")
HDR_BYTES, ENT_BYTES = 4096, 512
assert HDR.size <= HDR_BYTES and ENT.size == ENT_BYTES - 24, (HDR.size, ENT.size)

# experts/experts.h: NVFP4 block 2304 B per 8 rows x 512 k; S3 1792 B; MatDesc 64 B
Q4_BLK, S3_BLK, BLK_K, MATDESC_BYTES = 2304, 1792, 512, 64


def align_up(x, a=ALIGN):
    return (x + a - 1) // a * a


# ---- CRC32C (crc32c.c through ctypes; built on first use when CMake has not built it) ---------------------------
_crc = None


def _crc_lib():
    global _crc
    if _crc is None:
        cands = [os.environ.get("GLW_CRC_LIB", ""),
                 os.path.join(ROOT, "..", "build", "weights", "libglw_crc.so")]
        lib = next((c for c in cands if c and os.path.exists(c)), None)
        if lib is None:
            d = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "glw")
            os.makedirs(d, exist_ok=True)
            src = os.path.join(ROOT, "crc32c.c")
            tag = hashlib.sha256(open(src, "rb").read()).hexdigest()[:12]
            lib = os.path.join(d, f"libglw_crc-{tag}.so")
            if not os.path.exists(lib):
                tmp = tempfile.mktemp(dir=d, suffix=".so")
                subprocess.check_call(["cc", "-O2", "-shared", "-fPIC", "-o", tmp, src])
                os.rename(tmp, lib)
        _crc = ctypes.CDLL(lib)
        _crc.glw_crc32c.restype = ctypes.c_uint32
        _crc.glw_crc32c.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    return _crc


def crc32c(buf, crc=0):
    mv = memoryview(buf).cast("B")
    if not mv.nbytes:
        return crc
    arr = np.frombuffer(mv, np.uint8)
    return _crc_lib().glw_crc32c(arr.ctypes.data, arr.nbytes, crc)


def _check_crc():
    assert crc32c(b"123456789") == 0xE3069283, "CRC32C self-test failed"


# ---- tile layouts -----------------------------------------------------------------------------------------------
def t64(codes):
    """[N][K] e4m3 -> the attention GEMV tile (pack.tiles itself)."""
    return pack.tiles(codes)


def t128(codes):
    """[N][K] e4m3 -> the expert kernels' shared-expert tile (experts/layout.h pack_fp8): [N/8][K/128][32][32 B],
    lane 4g + t: row 8 n8 + g, k = 128 j + 64 q + 16 t + i at byte 16 q + i."""
    N, K = codes.shape
    assert N % 8 == 0 and K % 128 == 0, (N, K)
    return np.ascontiguousarray(codes.reshape(N // 8, 8, K // 128, 2, 4, 16).transpose(0, 2, 1, 4, 3, 5)).reshape(-1)


# ---- the plan ---------------------------------------------------------------------------------------------------
class Src:
    """A source tensor: `origin` "nv" (NVIDIA's checkpoint) or "bf16" (zai's BF16 mirror, fetched); `fp8`: quantised
    to (codes, scales) by quant.fp8_rtn when loaded."""

    def __init__(self, name, origin="nv", fp8=False):
        self.name, self.origin, self.fp8 = name, origin, fp8


class Out:
    def __init__(self, name, dtype, shape, layer=GLOBAL, layout=NATURAL, kind=REPLICATED, flags=0, blk=(0, 0),
                 sources=(), slice_="", fmt=FMT_NONE, reserve_bytes=0):
        self.name, self.dtype, self.shape, self.layer, self.layout = name, dtype, tuple(shape), layer, layout
        self.kind, self.flags, self.blk, self.sources, self.slice, self.fmt = kind, flags, blk, list(sources), slice_, fmt
        self.reserve_bytes = reserve_bytes
        self.bytes = 0 if kind == RESERVED else int(np.prod(self.shape)) * DTYPE_BYTES[dtype]
        self.offset = 0
        self.rel_err = 0.0
        self.component = ""

    def entry(self, crc):
        enc = lambda s, n: s.encode()[:n]  # noqa: E731
        if len(self.name) > 95 or len(",".join(self.sources)) > 191 or len(self.slice) > 95:
            raise SystemExit(f"{self.name}: a TOC string is too long")
        shape = list(self.shape) + [0] * (4 - len(self.shape))
        return ENT.pack(enc(self.name, 96), enc(",".join(self.sources), 192), enc(self.slice, 96), self.dtype,
                        len(self.shape), *shape, self.offset, self.bytes, self.kind, self.layout, crc, self.layer,
                        self.blk[0], self.blk[1], self.fmt, self.flags, self.rel_err, 0, self.reserve_bytes) + bytes(24)


def short(n):
    return n[len("model."):] if n.startswith("model.") else n


def bf16_np(t):
    import torch
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


class Unit:
    """Sources converted together and the outputs they give each rank."""

    def __init__(self, name, component, srcs, outs, build):
        self.name, self.component, self.srcs = name, component, srcs
        self._outs, self._build = outs, build  # outs(r) -> [Out]; build(S, r) -> {name: np.ndarray}

    def outs(self, r):
        os_ = self._outs(r)
        for o in os_:
            o.component = self.component
        return os_

    def build(self, S, r):
        return self._build(S, r)


def fp8_pair(name, layer, rows, K, rs, ks, layout, kind, sources, slice_, scale_shape=None, flags=0):
    """An FP8 weight (natural shape [rows][K], tiled) and its f32 scale table [rows/rs][K/ks]."""
    sshape = scale_shape or (rows // rs, K // ks)
    return [Out(name + ".weight", F8, (rows, K), layer, layout, kind, LOSSY | flags, (rs, ks), sources, slice_),
            Out(name + ".scale", F32, sshape, layer, NATURAL, kind, LOSSY | SCALE | flags, (rs, ks), sources,
                slice_ + " (scales)")]


def global_unit(ck):
    srcs = [Src("model.embed_tokens.weight"), Src("lm_head.weight"), Src("model.norm.weight")]
    vr = VOCAB // TP

    def outs(r):
        return [Out("embed.weight", BF16, (vr, D), kind=ROWS, sources=["embed_tokens.weight"],
                    slice_=f"vocab rows [{vr * r}, +{vr})"),
                Out("head.weight", BF16, (vr, D), kind=ROWS, sources=["lm_head.weight"],
                    slice_=f"vocab rows [{vr * r}, +{vr})"),
                Out("norm.weight", BF16, (D,), sources=["norm.weight"]),
                Out("rope.table", U32, (1 << 20, 32), kind=GENERATED, sources=["attn/tools/rope_table.py"],
                    slice_="[pos][32] bf16 cos | bf16 sin << 16")]

    def build(S, r):
        import rope_table
        if "rope" not in S:
            S["rope"] = rope_table.table(1 << 20)
        return {"embed.weight": bf16_np(S["model.embed_tokens.weight"][vr * r:vr * (r + 1)]),
                "head.weight": bf16_np(S["lm_head.weight"][vr * r:vr * (r + 1)]),
                "norm.weight": bf16_np(S["model.norm.weight"]), "rope.table": S["rope"]}
    return Unit("global", "embed, head, final norm, rope", srcs, outs, build)


def attn_unit(ck, L):
    p = f"model.layers.{L}."
    a = p + "self_attn."
    full = (a + "indexer.wk.weight") in ck
    names = {k: a + v for k, v in dict(q_a="q_a_proj.weight", kv_a="kv_a_proj_with_mqa.weight", q_b="q_b_proj.weight",
                                       kv_b="kv_b_proj.weight", o="o_proj.weight").items()}
    if full:
        names.update(wk=a + "indexer.wk.weight", wq_b=a + "indexer.wq_b.weight")
    norms = {"attn_norm.weight": p + "input_layernorm.weight", "ffn_norm.weight": p + "post_attention_layernorm.weight",
             "attn.q_norm.weight": a + "q_a_layernorm.weight", "attn.kv_norm.weight": a + "kv_a_layernorm.weight"}
    if full:
        norms.update({"indexer.k_norm.weight": a + "indexer.k_norm.weight",
                      "indexer.k_norm.bias": a + "indexer.k_norm.bias"})
    srcs = [Src(n, fp8=True) for n in names.values()] + [Src(n) for n in norms.values()]
    if full:
        srcs.append(Src(a + "indexer.weights_proj.weight"))
    ln = f"layers.{L}."
    xrows = 656 + (32 if full else 0)
    H = 16

    def outs(r):
        o = []
        for k, n in norms.items():
            o.append(Out(ln + k, BF16, ck.shape(n), L, sources=[short(n)]))
        xs = [short(names["q_a"]), short(names["kv_a"])] + ([short(names["wk"])] if full else [])
        o += fp8_pair(ln + "attn.xproj", L, xrows, D, 16, 128, T64, RECUT, xs,
                      f"q_a [{512 * r},+512) kv_a [{128 * r},+128) [{512 + 16 * r},+16)" +
                      (f" wk [{32 * r},+32)" if full else ""))
        o += fp8_pair(ln + "attn.q_b", L, 4096, 2048, 128, 128, T64, ROWS, [short(names["q_b"])],
                      f"rows [{4096 * r}, +4096): heads {16 * r}..{16 * r + 15}")
        o += fp8_pair(ln + "attn.w_uk", L, H * 512, 192, 128, 64, T64, RECUT, [short(names["kv_b"])],
                      "heads' rows 448H..+192 transposed [512][192]")
        o += fp8_pair(ln + "attn.w_uv", L, H * 256, 512, 64, 128, T64, RECUT, [short(names["kv_b"])],
                      "heads' rows 448H+192..+256 [256][512]")
        o += fp8_pair(ln + "attn.o_proj", L, D, 4096, 128, 128, T64, COLS, [short(names["o"])],
                      f"cols [{4096 * r}, +4096)")
        if full:
            o += fp8_pair(ln + "indexer.wq_b", L, 1024, 2048, 128, 128, T64, ROWS, [short(names["wq_b"])],
                          f"rows [{1024 * r}, +1024): index heads {8 * r}..{8 * r + 7}")
            o.append(Out(ln + "indexer.weights_proj.weight", BF16, (8, D), L, kind=ROWS,
                         sources=[short(a + "indexer.weights_proj.weight")], slice_=f"rows [{8 * r}, +8)"))
        return o

    def build(S, r):
        d = {}
        for k, n in norms.items():
            d[ln + k] = bf16_np(S[n])
        packs = {"attn.xproj": pack.xproj(S[names["q_a"]], S[names["kv_a"]], S[names["wk"]] if full else None, r),
                 "attn.q_b": pack.rows_block(S[names["q_b"]], r, 4096),
                 "attn.w_uk": pack.w_uk(S[names["kv_b"]], r), "attn.w_uv": pack.w_uv(S[names["kv_b"]], r),
                 "attn.o_proj": pack.cols_block(S[names["o"]], r, 4096)}
        if full:
            packs["indexer.wq_b"] = pack.rows_block(S[names["wq_b"]], r, 1024)
            d[ln + "indexer.weights_proj.weight"] = bf16_np(S[a + "indexer.weights_proj.weight"][8 * r:8 * r + 8])
        for k, (t, s, _, _) in packs.items():
            d[ln + k + ".weight"], d[ln + k + ".scale"] = t, s
        return d
    return Unit(f"layer {L} attention", "attention" if L != MTP else "MTP attention", srcs, outs, build)


def mlp_unit(ck, L, shared_layout="t128"):
    p = f"model.layers.{L}.mlp."
    ln = f"layers.{L}."
    if L < 3:  # dense: the master from zai's BF16 mirror (NVIDIA's file holds a per-tensor FP8 requantisation)
        g, u, dn = p + "gate_proj.weight", p + "up_proj.weight", p + "down_proj.weight"
        srcs = [Src(g, "bf16", True), Src(u, "bf16", True), Src(dn, "bf16", True)]
        inter = DENSE_INTER // TP
        comp, nm = "dense MLP", "mlp"
    else:
        s = p + "shared_experts."
        g, u, dn = s + "gate_proj.weight", s + "up_proj.weight", s + "down_proj.weight"
        srcs = [Src(g, fp8=True), Src(u, fp8=True), Src(dn, fp8=True), Src(p + "gate.weight"),
                Src(p + "gate.e_score_correction_bias")]
        inter = SHARED_INTER // TP
        comp, nm = "shared expert, router" if L != MTP else "MTP shared expert, router", "shared"
    mk = lambda n: "zai-bf16:" + short(n) if L < 3 else short(n)  # noqa: E731
    t128_ = L >= 3 and shared_layout == "t128"

    def outs(r):
        o = []
        if t128_:
            for k, n in (("gate", g), ("up", u)):
                o += fp8_pair(ln + f"{nm}.{k}", L, inter, D, 128, 128, T128, ROWS, [mk(n)],
                              f"rows [{inter * r}, +{inter})")
        else:
            o += fp8_pair(ln + f"{nm}.gate_up", L, 2 * inter, D, 128, 128, T64, RECUT, [mk(g), mk(u)],
                          f"rows [{inter * r},+{inter}) interleaved gate/up", (inter // 128, 2, D // 128), INTERLEAVED)
        o += fp8_pair(ln + f"{nm}.down", L, D, inter, 128, 128, T128 if t128_ else T64, COLS, [mk(dn)],
                      f"cols [{inter * r}, +{inter})")
        if L >= 3:
            o.append(Out(ln + "router.weight", BF16, (N_EXPERTS, D), L, sources=[short(p + "gate.weight")]))
            o.append(Out(ln + "router.bias", F32, (N_EXPERTS,), L, sources=[short(p + "gate.e_score_correction_bias")]))
        return o

    def build(S, r):
        d = {}
        if t128_:
            for k, n in (("gate", g), ("up", u)):
                c, s_ = S[n]
                d[ln + f"{nm}.{k}.weight"] = t128(c[inter * r:inter * (r + 1)])
                d[ln + f"{nm}.{k}.scale"] = s_[inter * r // 128:inter * (r + 1) // 128].copy()
            c, s_ = S[dn]
            d[ln + f"{nm}.down.weight"] = t128(np.ascontiguousarray(c[:, inter * r:inter * (r + 1)]))
            d[ln + f"{nm}.down.scale"] = np.ascontiguousarray(s_[:, inter * r // 128:inter * (r + 1) // 128])
        else:
            t, s_, _, _ = pack.gate_up(S[g], S[u], r, inter)
            d[ln + f"{nm}.gate_up.weight"], d[ln + f"{nm}.gate_up.scale"] = t, s_
            t, s_, _, _ = pack.cols_block(S[dn], r, inter)
            d[ln + f"{nm}.down.weight"], d[ln + f"{nm}.down.scale"] = t, s_
        if L >= 3:
            d[ln + "router.weight"] = bf16_np(S[p + "gate.weight"])
            d[ln + "router.bias"] = S[p + "gate.e_score_correction_bias"].numpy()
        return d
    return Unit(f"layer {L} mlp", comp, srcs, outs, build)


def mtp_unit(ck):
    p = f"model.layers.{MTP}."
    ln = f"layers.{MTP}.mtp."
    norms = {"enorm.weight": p + "enorm.weight", "hnorm.weight": p + "hnorm.weight",
             "head_norm.weight": p + "shared_head.norm.weight"}
    eh = p + "eh_proj.weight"
    srcs = [Src(eh, fp8=True)] + [Src(n) for n in norms.values()]
    rows = D // TP

    def outs(r):
        o = [Out(ln + k, BF16, (D,), MTP, sources=[short(n)]) for k, n in norms.items()]
        o += fp8_pair(ln + "eh_proj", MTP, rows, 2 * D, 128, 128, T64, ROWS, [short(eh)],
                      f"rows [{rows * r}, +{rows}) (FP8 by the same rule; BF16 in zai's release)")
        return o

    def build(S, r):
        d = {ln + k: bf16_np(S[n]) for k, n in norms.items()}
        t, s, _, _ = pack.rows_block(S[eh], r, rows)
        d[ln + "eh_proj.weight"], d[ln + "eh_proj.scale"] = t, s
        return d
    return Unit("MTP extras", "MTP eh_proj, norms", srcs, outs, build)


def expert_fmt(gate=FMT_NVFP4, up=FMT_NVFP4, down=FMT_NVFP4):
    """An expert entry's format tag: one nibble a matrix (gx::Fmt: 0 NVFP4, 1 S3), gate | up << 4 | down << 8."""
    return gate | up << 4 | down << 8


def mat_bytes(fmt, n, k):
    """experts.h mat_bytes: [N/8][K/512] blocks of 2304 (NVFP4) or 1792 (S3) bytes."""
    return (n // 8) * (k // BLK_K) * (Q4_BLK if fmt == FMT_NVFP4 else S3_BLK)


def expert_entry_bytes(tag):
    i = EXPERT_INTER // TP
    return mat_bytes(tag & 15, i, D) + mat_bytes(tag >> 4 & 15, i, D) + mat_bytes(tag >> 8 & 15, D, i)


def reserved_unit(L):
    """A MoE layer's routed experts: the descriptor table, then one entry per expert (gate, up, down blocks back to
    back), reserved until an expert fill writes them. Every matrix size is a multiple of 4 KiB in either format, so
    the 256 expert entries lie back to back: the layer's pool starts at expert 0's entry and MatDesc::off is
    relative to it (docs/layout.md § 6)."""
    ln = f"layers.{L}.experts."
    i = EXPERT_INTER // TP

    def outs(r):
        tag = expert_fmt()
        o = [Out(ln + "desc", U8, (N_EXPERTS, 3, MATDESC_BYTES), L, NATURAL, RESERVED, RESERVED_FLAG, (0, 0),
                 [f"layers.{L}.mlp.experts.{{e}}.*_proj.weight_scale_2"], "gx::MatDesc [256][gate, up, down]", tag,
                 N_EXPERTS * 3 * MATDESC_BYTES)]
        for e in range(N_EXPERTS):
            o.append(Out(ln + str(e), U8, (expert_entry_bytes(tag),), L, EXPERT_POOL, RESERVED, RESERVED_FLAG, (0, 0),
                         [f"layers.{L}.mlp.experts.{e}.{{gate,up,down}}_proj"],
                         f"gate, up rows / down cols [{i * r},+{i}); 8x512 blocks", tag, expert_entry_bytes(tag)))
        return o
    return Unit(f"layer {L} routed experts (reserved)", "routed experts (reserved)", [], outs, lambda S, r: {})


def make_plan(ck, shared_layout="t128"):
    """The units in file order."""
    units = [global_unit(ck)]
    for L in range(N_LAYERS + 1):
        units.append(attn_unit(ck, L))
        units.append(mlp_unit(ck, L, shared_layout))
        if L == MTP:
            units.append(mtp_unit(ck))
    units += [reserved_unit(L) for L in range(3, N_LAYERS + 1)]
    return units


def layout_rank(units, r):
    """The rank's outputs in file order with offsets; returns (outs, toc_offset, data_offset, file_bytes)."""
    outs = [o for u in units for o in u.outs(r)]
    names = set()
    for o in outs:
        if o.name in names:
            raise SystemExit(f"plan lists {o.name} twice")
        names.add(o.name)
    toc_offset = HDR_BYTES
    data_offset = align_up(toc_offset + len(outs) * ENT_BYTES)
    off = data_offset
    for o in outs:
        if o.kind == RESERVED:
            o.offset = 0
            continue
        o.offset = off
        off += align_up(o.bytes)
    return outs, toc_offset, data_offset, off


def coverage(ck, units):
    """Every checkpoint tensor is either a source of exactly one unit, a routed expert (reserved entries), or a
    dense-MLP tensor replaced by its master from the BF16 mirror. Returns (used, reserved, replaced)."""
    import re
    used = {}
    for u in units:
        for s in u.srcs:
            if s.origin == "nv":
                if s.name in used:
                    raise SystemExit(f"{s.name} is a source of two units")
                used[s.name] = u.name
    exp = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(weight|weight_scale|weight_scale_2|input_scale)$")
    dense = re.compile(r"model\.layers\.[012]\.mlp\.(gate|up|down)_proj\.(weight|weight_scale|input_scale)$")
    reserved, replaced, unknown = [], [], []
    for n in ck.names():
        if n in used:
            continue
        if exp.match(n):
            reserved.append(n)
        elif dense.match(n):
            replaced.append(n)
        else:
            unknown.append(n)
    missing = [n for n in used if n not in ck.meta]
    if unknown or missing:
        raise SystemExit(f"coverage: {len(unknown)} checkpoint tensors in no unit (e.g. {unknown[:3]}), "
                         f"{len(missing)} planned sources missing (e.g. {missing[:3]})")
    return used, reserved, replaced


# ---- reading a file ---------------------------------------------------------------------------------------------
def cstr(b):
    return b.split(b"\0", 1)[0].decode()


def read_toc(path):
    with open(path, "rb") as f:
        h = HDR.unpack(f.read(HDR_BYTES)[:HDR.size])
        hd = dict(zip(["magic", "version", "header_bytes", "rank", "world", "n", "entry_bytes", "toc_offset",
                       "data_offset", "data_bytes", "file_bytes", "alignment", "toc_crc"], h[:13]))
        hd.update(model=cstr(h[13]), source=cstr(h[14]), created=cstr(h[15]), tool=cstr(h[16]))
        if hd["magic"] != MAGIC:
            raise SystemExit(f"{path}: not a .glw file")
        f.seek(hd["toc_offset"])
        raw = f.read(hd["n"] * ENT_BYTES)
    ents = []
    for i in range(hd["n"]):
        e = ENT.unpack(raw[i * ENT_BYTES:i * ENT_BYTES + ENT.size])
        ents.append(dict(name=cstr(e[0]), source=cstr(e[1]), slice=cstr(e[2]), dtype=e[3], ndim=e[4],
                         shape=tuple(e[5:5 + e[4]]), offset=e[9], bytes=e[10], kind=e[11], layout=e[12], crc=e[13],
                         layer=e[14], blk=(e[15], e[16]), fmt=e[17], flags=e[18], rel_err=e[19],
                         reserve_bytes=e[21]))
    return hd, ents, raw


def tool_files():
    return [os.path.join(ROOT, f) for f in ("glw.py", "convert.py", "quant.py", "sources.py", "fetch.py", "crc32c.c")] + \
        [os.path.join(ATTN_TOOLS, f) for f in ("pack.py", "rope_table.py")]


def tool_version():
    h = hashlib.sha256()
    for f in tool_files():
        h.update(os.path.basename(f).encode() + b"\0" + open(f, "rb").read())
    return h.hexdigest()
