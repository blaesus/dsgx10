"""The .glw container (glw.h, mirrored field by field), altar-1's conversion plan, and the per-unit builders.

The plan is a list of units, each a group of source tensors converted together (one layer's attention, one layer's
MLP, ...). A unit lists its sources, the output tensors every rank gets (name, dtype, natural shape, layout, cut),
and builds a rank's bytes through a reader of the checkpoint. Nothing is requantised (DESIGN decision 1): BF16 and
F32 tensors are copied (cut, stacked, transposed or interleaved), the INT4 tensors' packed values, scales and zero
points are cut on group boundaries and re-tiled by the kernels' packers, and the one derived form, an INT4 layer's
W_UK^T when the attention side serves it in BF16, is the contract's exact dequantisation (int4.dequant_bf16).
Shapes and sizes come from the plan alone, so --plan needs only the checkpoint's headers. docs/layout.md has the
table.

The INT4 tile layouts belong to the kernels: the routed experts' blocks to engine/experts/layout.h (its python
packer registers in EXPERT_FORMATS), q_b / W_UV / o_proj's GEMV tile to engine/attn/tools/pack.py (ATTN_INT4).
Until a layout is registered its tensors cannot be written: the converter's --bf16-only leaves the INT4 attention
out and the expert fill refuses an unregistered format.
"""
import ctypes, hashlib, os, re, struct, subprocess, sys, tempfile

import numpy as np

import int4

ROOT = os.path.dirname(os.path.abspath(__file__))
ATTN_TOOLS = os.path.normpath(os.path.join(ROOT, "..", "attn", "tools"))
if ATTN_TOOLS not in sys.path:
    sys.path.insert(0, ATTN_TOOLS)

MAGIC = b"ALTR1W01"
VERSION = 1
ALIGN = 4096
GLOBAL = 0xFFFFFFFF
TP, D, VOCAB = 4, 6144, 154880
N_LAYERS, MTP = 78, 78
DENSE_INTER, SHARED_INTER, N_EXPERTS, EXPERT_INTER = 12288, 2048, 168, 2048
INT4_LAYERS = range(4, 77)  # routed experts and q_b, kv_b, o_proj in INT4 (checked against the headers)
HEADS, NOPE, VD, KVL = 16, 192, 256, 512  # heads a rank, W_UK rows, W_UV rows, latent

# glw.h enums
F32, BF16, F8, U8, U32 = 1, 2, 3, 4, 5
DTYPE_NAME = {F32: "F32", BF16: "BF16", F8: "F8_E4M3", U8: "U8", U32: "U32"}
DTYPE_BYTES = {F32: 4, BF16: 2, F8: 1, U8: 1, U32: 4}
NP_DTYPE = {F32: np.float32, BF16: np.uint16, F8: np.uint8, U8: np.uint8, U32: np.uint32}
REPLICATED, ROWS, COLS, RECUT, GENERATED, RESERVED = 1, 2, 3, 4, 5, 6
KIND_NAME = {1: "replicated", 2: "rows", 3: "cols", 4: "recut", 5: "generated", 6: "reserved"}
NATURAL, T64, T128, EXPERT_POOL, INT4_TILE = 0, 1, 2, 3, 4
LAYOUT_NAME = {0: "natural", 1: "T64", 2: "T128", 3: "expert-pool", 4: "int4-tile"}
FMT_NVFP4, FMT_S3, FMT_INT4, FMT_BF16, FMT_NONE = 0, 1, 2, 3, 0xFFFFFFFF
FMT_NAME = {0: "nvfp4", 1: "s3", 2: "int4", 3: "bf16"}
LOSSY, RESERVED_FLAG, SCALE, INTERLEAVED, DEQUANT = 1, 2, 4, 8, 16

# Header: magic, version, header_bytes, rank, world, n_tensors, toc_entry_bytes, toc_offset, data_offset,
# data_bytes, file_bytes, alignment, toc_crc32c, model[64], source[256], created[32], tool[64]
HDR = struct.Struct("<8sIIIIII QQQQ II 64s256s32s64s")
# TocEntry: name[96], source[192], slice[96], dtype, ndim, shape[4], offset, bytes, kind, layout, crc32c, layer,
# blk_rows, blk_k, fmt, flags, rel_err, pad0, reserve_bytes
ENT = struct.Struct("<96s192s96s II 4Q QQ IIII IIII f I Q")
HDR_BYTES, ENT_BYTES = 4096, 512
assert HDR.size <= HDR_BYTES and ENT.size == ENT_BYTES - 24, (HDR.size, ENT.size)
MATDESC_BYTES = 64


def align_up(x, a=ALIGN):
    return (x + a - 1) // a * a


def int4_layer(L):
    return L in INT4_LAYERS


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


# ---- the pluggable INT4 / BF16 tile packers -----------------------------------------------------------------------
class ExpertFormat:
    """A routed-expert matrix format of the expert kernels (experts.h Fmt): its bytes for a rank's [n][k] matrix,
    the packer from the canonical slice (int4 dict, or uint16 [n][k] for BF16) to those bytes, its exact inverse
    (the fill's read-back), and the 64-byte gx::MatDesc for a matrix at pool offset `off`."""

    def __init__(self, fmt, name, mat_bytes, pack, unpack, desc, source=""):
        self.fmt, self.name, self.mat_bytes, self.pack, self.unpack, self.desc = fmt, name, mat_bytes, pack, unpack, desc
        self.source = source  # the packer's module file (hashed into the manifests)


EXPERT_FORMATS = {}  # fmt -> ExpertFormat; filled by expert_formats.register() (the expert kernels' packers)


class AttnInt4:
    """The attention GEMV's INT4 tile (attn/tools/pack.py): for a canonical INT4 cut (int4 dict) of q_b
    [4096][2048] (rows), W_UV [16 * 256][512] (each head's value rows, stacked) or o_proj [6144][4096] (cols), the
    entries it becomes: outs(name, N, K, grouped) -> [(suffix, dtype, shape, layout, blk, flags)], pack(t, N, K,
    grouped) -> {suffix: array}, unpack({suffix: bytes}, N, K, grouped) -> int4 dict. `w_uk_bf16`: True when the
    attention side serves an INT4 layer's W_UK^T as the contract's dequantised BF16."""

    def __init__(self, outs, pack, unpack, w_uk_bf16=True, source=""):
        self.outs, self.pack, self.unpack, self.w_uk_bf16, self.source = outs, pack, unpack, w_uk_bf16, source


ATTN_INT4 = None  # set by attn_formats.register()


def expert_fmt(gate, up=None, down=None):
    """An expert entry's format tag: one nibble a matrix (gx::Fmt), gate | up << 4 | down << 8."""
    up = gate if up is None else up
    down = gate if down is None else down
    return gate | up << 4 | down << 8


def layer_expert_fmt(L):
    return FMT_INT4 if int4_layer(L) else FMT_BF16


def mat_bytes(fmt, n, k):
    if fmt not in EXPERT_FORMATS:
        raise SystemExit(f"expert format {FMT_NAME.get(fmt, fmt)}: no packer registered (engine/experts/layout.h's "
                         "python packer has not landed, or expert_formats.register() was not called)")
    b = EXPERT_FORMATS[fmt].mat_bytes(n, k)
    if b % ALIGN:
        raise SystemExit(f"expert format {FMT_NAME[fmt]}: a [{n}][{k}] matrix is {b} B, not a multiple of 4 KiB")
    return b


def expert_entry_bytes(tag):
    i = EXPERT_INTER // TP
    return mat_bytes(tag & 15, i, D) + mat_bytes(tag >> 4 & 15, i, D) + mat_bytes(tag >> 8 & 15, D, i)


# ---- the plan ---------------------------------------------------------------------------------------------------
class Src:
    """A source tensor of the checkpoint (`rows`: only a rank's rows are read, per rank)."""

    def __init__(self, name, rows=False):
        self.name, self.rows = name, rows


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


class Reader:
    """A unit's view of the checkpoint: whole tensors cached for the unit, row reads not; every read's SHA-256
    recorded in `info` (name, or name[a:b] for a row read)."""

    def __init__(self, ck, info):
        self.ck, self.info, self.cache = ck, info, {}

    def get(self, n):
        if n not in self.cache:
            t, sha = self.ck.get(n)
            self.info[n] = {"sha256": sha, "dtype": self.ck.dtype(n), "shape": list(t.shape)}
            self.cache[n] = t
        return self.cache[n]

    def rows(self, n, a, b):
        t, sha = self.ck.get_rows(n, a, b)
        self.info[f"{n}[{a}:{b}]"] = {"sha256": sha, "dtype": self.ck.dtype(n), "shape": list(t.shape)}
        return t

    def int4(self, base):
        if base not in self.cache:
            t, shas = int4.parts(self.ck, base)
            for k, v in shas.items():
                self.info[k] = {"sha256": v, "dtype": self.ck.dtype(k), "shape": list(self.ck.shape(k))}
            self.cache[base] = t
        return self.cache[base]


class Unit:
    """Sources converted together and the outputs they give each rank."""

    def __init__(self, name, component, srcs, outs, build):
        self.name, self.component, self.srcs = name, component, srcs
        self._outs, self._build = outs, build  # outs(r) -> [Out]; build(R, r) -> {name: np.ndarray}

    def outs(self, r):
        os_ = self._outs(r)
        for o in os_:
            o.component = self.component
        return os_

    def build(self, R, r):
        return self._build(R, r)


def global_units(ck):
    """The embedding and head (each rank's vocabulary rows, read as rows), the final norm, and the RoPE table, as
    separate units so that only one of them is in memory at a time."""
    vr = VOCAB // TP
    us = []
    for nm, src in (("embed.weight", "model.embed_tokens.weight"), ("head.weight", "lm_head.weight")):
        def outs(r, nm=nm, src=src):
            return [Out(nm, BF16, (vr, D), kind=ROWS, sources=[short(src)], slice_=f"vocab rows [{vr * r}, +{vr})")]

        def build(R, r, nm=nm, src=src):
            return {nm: R.rows(src, vr * r, vr * (r + 1))}
        us.append(Unit(nm.split(".")[0], "embed, head, final norm, rope", [Src(src, True)], outs, build))
    us.append(Unit("final norm", "embed, head, final norm, rope", [Src("model.norm.weight")],
                   lambda r: [Out("norm.weight", BF16, (D,), sources=["norm.weight"])],
                   lambda R, r: {"norm.weight": R.get("model.norm.weight")}))

    def rope(R, r):
        if "rope" not in R.cache:
            import rope_table  # torch: the table is torch's own cos/sin (rope_table.py)
            R.cache["rope"] = rope_table.table(1 << 20)
        return {"rope.table": R.cache["rope"]}
    us.append(Unit("rope", "embed, head, final norm, rope", [],
                   lambda r: [Out("rope.table", U32, (1 << 20, 32), kind=GENERATED, sources=["attn/tools/rope_table.py"],
                                  slice_="[pos][32] bf16 cos | bf16 sin << 16")], rope))
    return us


def w_uk_t(kvb, r):
    """W_UK^T of rank r's heads from kv_b [28672][512] (BF16 bits): head H's rows 448H..+192 transposed to
    [512][192], stacked [16 * 512][192]."""
    return np.ascontiguousarray(np.concatenate([kvb[448 * H:448 * H + NOPE].T for H in range(HEADS * r, HEADS * (r + 1))]))


def w_uv_rows(r):
    return [(448 * H + NOPE, 448 * H + 448) for H in range(HEADS * r, HEADS * (r + 1))]


def interleave(g, u):
    out = np.empty((2 * g.shape[0], g.shape[1]), g.dtype)
    out[0::2], out[1::2] = g, u
    return out


def attn_unit(ck, L, bf16_only=False):
    p = f"model.layers.{L}."
    a = p + "self_attn."
    full = (a + "indexer.wk.weight") in ck
    i4 = (a + "q_b_proj.weight_packed") in ck
    if i4 != int4_layer(L):
        raise SystemExit(f"layer {L}: the checkpoint's attention format (INT4 {i4}) is not the plan's")
    if i4 and not bf16_only and ATTN_INT4 is None:
        raise SystemExit("the attention side's INT4 tile (attn/tools/pack.py) is not registered: --bf16-only, or "
                         "attn_formats.register()")
    names = {k: a + v for k, v in dict(q_a="q_a_proj.weight", kv_a="kv_a_proj_with_mqa.weight").items()}
    for k, v in dict(q_b="q_b_proj", kv_b="kv_b_proj", o="o_proj").items():
        names[k] = a + v + ("" if i4 else ".weight")  # INT4: the base name of the three parts
    if full:
        names.update(wk=a + "indexer.wk.weight", wq_b=a + "indexer.wq_b.weight",
                     wp=a + "indexer.weights_proj.weight")
    norms = {"attn_norm.weight": p + "input_layernorm.weight", "ffn_norm.weight": p + "post_attention_layernorm.weight",
             "attn.q_norm.weight": a + "q_a_layernorm.weight", "attn.kv_norm.weight": a + "kv_a_layernorm.weight"}
    if full:
        norms.update({"indexer.k_norm.weight": a + "indexer.k_norm.weight",
                      "indexer.k_norm.bias": a + "indexer.k_norm.bias"})
    srcs = [Src(names[k]) for k in ("q_a", "kv_a") + (("wk", "wq_b", "wp") if full else ())]
    srcs += [Src(n) for n in norms.values()]
    for k in ("q_b", "kv_b", "o"):
        if i4 and bf16_only and k != "kv_b":
            continue  # pending: coverage lists them
        srcs += [Src(names[k] + s) for s in (".weight_packed", ".weight_scale", ".weight_zero_point",
                                             ".weight_shape")] if i4 else [Src(names[k])]
    ln = f"layers.{L}."
    xrows = 656 + (32 if full else 0)
    sn = lambda k: short(names[k])  # noqa: E731
    s4 = lambda k: [sn(k) + ".weight_{packed,scale,zero_point}"]  # noqa: E731
    W_UV = (HEADS * VD, KVL)

    def i4_outs(o, k, N, K, src, slice_, grouped, kind):
        if bf16_only:
            return
        for suf, dt, shp, lay, blk, fl in ATTN_INT4.outs(k, N, K, grouped):
            o.append(Out(ln + k + suf, dt, shp, L, lay, kind, fl, blk, src, slice_))

    def outs(r):
        o = [Out(ln + k, BF16, ck.shape(n), L, sources=[short(n)]) for k, n in norms.items()]
        xs = [sn("q_a"), sn("kv_a")] + ([sn("wk")] if full else [])
        o.append(Out(ln + "attn.xproj.weight", BF16, (xrows, D), L, NATURAL, RECUT, 0, (0, 0), xs,
                     f"q_a [{512 * r},+512) kv_a [{128 * r},+128) [{512 + 16 * r},+16)" +
                     (f" wk [{32 * r},+32)" if full else "")))
        qb_slice = f"rows [{4096 * r}, +4096): heads {16 * r}..{16 * r + 15}"
        o_slice = f"cols [{4096 * r}, +4096)"
        uk_slice = "heads' rows 448H..+192 transposed [512][192]"
        uv_slice = "heads' rows 448H+192..+256 [256][512]"
        if i4:
            i4_outs(o, "attn.q_b", 4096, 2048, s4("q_b"), qb_slice, False, ROWS)
            if bf16_only or ATTN_INT4.w_uk_bf16:
                o.append(Out(ln + "attn.w_uk.weight", BF16, (HEADS * KVL, NOPE), L, NATURAL, RECUT, DEQUANT, (0, 0),
                             s4("kv_b"), uk_slice + ", bf16((q-z)s)"))
            else:
                i4_outs(o, "attn.w_uk", HEADS * KVL, NOPE, s4("kv_b"), uk_slice, True, RECUT)
            i4_outs(o, "attn.w_uv", *W_UV, s4("kv_b"), uv_slice, True, RECUT)
            i4_outs(o, "attn.o_proj", D, 4096, s4("o"), o_slice, False, COLS)
        else:
            o.append(Out(ln + "attn.q_b.weight", BF16, (4096, 2048), L, NATURAL, ROWS, 0, (0, 0), [sn("q_b")], qb_slice))
            o.append(Out(ln + "attn.w_uk.weight", BF16, (HEADS * KVL, NOPE), L, NATURAL, RECUT, 0, (0, 0),
                         [sn("kv_b")], uk_slice))
            o.append(Out(ln + "attn.w_uv.weight", BF16, W_UV, L, NATURAL, RECUT, 0, (0, 0), [sn("kv_b")], uv_slice))
            o.append(Out(ln + "attn.o_proj.weight", BF16, (D, 4096), L, NATURAL, COLS, 0, (0, 0), [sn("o")], o_slice))
        if full:
            o.append(Out(ln + "indexer.wq_b.weight", BF16, (1024, 2048), L, NATURAL, ROWS, 0, (0, 0), [sn("wq_b")],
                         f"rows [{1024 * r}, +1024): index heads {8 * r}..{8 * r + 7}"))
            o.append(Out(ln + "indexer.weights_proj.weight", BF16, (8, D), L, NATURAL, ROWS, 0, (0, 0), [sn("wp")],
                         f"rows [{8 * r}, +8)"))
        return o

    def build(R, r):
        d = {ln + k: R.get(n) for k, n in norms.items()}
        qa, kva = R.get(names["q_a"]), R.get(names["kv_a"])
        xs = [qa[512 * r:512 * r + 512], kva[128 * r:128 * r + 128], kva[512 + 16 * r:512 + 16 * r + 16]]
        if full:
            xs.append(R.get(names["wk"])[32 * r:32 * r + 32])
        d[ln + "attn.xproj.weight"] = np.concatenate(xs)
        if i4:
            kvb = R.int4(names["kv_b"])
            if bf16_only or ATTN_INT4.w_uk_bf16:
                d[ln + "attn.w_uk.weight"] = w_uk_t(int4.dequant_bf16(kvb), r)
        if i4 and not bf16_only:
            qb, op = R.int4(names["q_b"]), R.int4(names["o"])
            pieces = {"attn.q_b": (int4.rows(qb, 4096 * r, 4096 * (r + 1)), 4096, 2048, False),
                      "attn.w_uv": (int4.cat_rows([int4.rows(kvb, x, y) for x, y in w_uv_rows(r)]), *W_UV, True),
                      "attn.o_proj": (int4.cols(op, 4096 * r, 4096 * (r + 1)), D, 4096, False)}
            if not ATTN_INT4.w_uk_bf16:
                pieces["attn.w_uk"] = (None, HEADS * KVL, NOPE, True)  # a transposed INT4 tile: the packer's own cut
            for k, (t, N, K, g) in pieces.items():
                if t is None:
                    t = dict(kv_b=kvb, rank=r)
                for suf, arr in ATTN_INT4.pack(t, N, K, g).items():
                    d[ln + k + suf] = arr
        elif not i4:
            qb, kvb, op = R.get(names["q_b"]), R.get(names["kv_b"]), R.get(names["o"])
            d[ln + "attn.q_b.weight"] = qb[4096 * r:4096 * (r + 1)]
            d[ln + "attn.w_uk.weight"] = w_uk_t(kvb, r)
            d[ln + "attn.w_uv.weight"] = np.concatenate([kvb[x:y] for x, y in w_uv_rows(r)])
            d[ln + "attn.o_proj.weight"] = np.ascontiguousarray(op[:, 4096 * r:4096 * (r + 1)])
        if full:
            d[ln + "indexer.wq_b.weight"] = R.get(names["wq_b"])[1024 * r:1024 * (r + 1)]
            d[ln + "indexer.weights_proj.weight"] = R.get(names["wp"])[8 * r:8 * r + 8]
        return d
    return Unit(f"layer {L} attention", ("attention" if L != MTP else "MTP attention") +
                (" (INT4)" if i4 else ""), srcs, outs, build)


def mlp_unit(ck, L):
    """Dense MLP (layers 0-2) or the shared expert and router (3-78): BF16, gate/up interleaved (rows 2i gate, 2i+1
    up: the BF16 GEMV's kSwiGLU), down by input columns."""
    p = f"model.layers.{L}.mlp."
    ln = f"layers.{L}."
    if L < 3:
        g, u, dn = p + "gate_proj.weight", p + "up_proj.weight", p + "down_proj.weight"
        inter, comp, nm = DENSE_INTER // TP, "dense MLP", "mlp"
        srcs = [Src(g), Src(u), Src(dn)]
    else:
        s = p + "shared_experts."
        g, u, dn = s + "gate_proj.weight", s + "up_proj.weight", s + "down_proj.weight"
        inter, nm = SHARED_INTER // TP, "shared"
        comp = "shared expert, router" if L != MTP else "MTP shared expert, router"
        srcs = [Src(g), Src(u), Src(dn), Src(p + "gate.weight"), Src(p + "gate.e_score_correction_bias")]

    def outs(r):
        o = [Out(ln + f"{nm}.gate_up.weight", BF16, (2 * inter, D), L, NATURAL, RECUT, INTERLEAVED, (0, 0),
                 [short(g), short(u)], f"rows [{inter * r},+{inter}) interleaved gate/up"),
             Out(ln + f"{nm}.down.weight", BF16, (D, inter), L, NATURAL, COLS, 0, (0, 0), [short(dn)],
                 f"cols [{inter * r}, +{inter})")]
        if L >= 3:
            o.append(Out(ln + "router.weight", BF16, (N_EXPERTS, D), L, sources=[short(p + "gate.weight")]))
            o.append(Out(ln + "router.bias", F32, (N_EXPERTS,), L, sources=[short(p + "gate.e_score_correction_bias")]))
        return o

    def build(R, r):
        a = inter * r
        d = {ln + f"{nm}.gate_up.weight": interleave(R.get(g)[a:a + inter], R.get(u)[a:a + inter]),
             ln + f"{nm}.down.weight": np.ascontiguousarray(R.get(dn)[:, a:a + inter])}
        if L >= 3:
            d[ln + "router.weight"] = R.get(p + "gate.weight")
            d[ln + "router.bias"] = R.get(p + "gate.e_score_correction_bias")
        return d
    return Unit(f"layer {L} mlp", comp, srcs, outs, build)


def mtp_unit(ck):
    p = f"model.layers.{MTP}."
    ln = f"layers.{MTP}.mtp."
    norms = {"enorm.weight": p + "enorm.weight", "hnorm.weight": p + "hnorm.weight",
             "head_norm.weight": p + "shared_head.norm.weight"}
    eh = p + "eh_proj.weight"
    srcs = [Src(eh)] + [Src(n) for n in norms.values()]
    rows = D // TP

    def outs(r):
        o = [Out(ln + k, BF16, (D,), MTP, sources=[short(n)]) for k, n in norms.items()]
        o.append(Out(ln + "eh_proj.weight", BF16, (rows, 2 * D), MTP, NATURAL, ROWS, 0, (0, 0), [short(eh)],
                     f"rows [{rows * r}, +{rows})"))
        return o

    def build(R, r):
        d = {ln + k: R.get(n) for k, n in norms.items()}
        d[ln + "eh_proj.weight"] = R.get(eh)[rows * r:rows * (r + 1)]
        return d
    return Unit("MTP extras", "MTP eh_proj, norms", srcs, outs, build)


def reserved_unit(L, sized=True):
    """A MoE layer's routed experts: the descriptor table, then one entry per expert (gate, up, down back to back),
    reserved until experts_fill.py writes them. Layers 4-76 INT4, 3, 77 and the MTP layer BF16. Every matrix is a
    multiple of 4 KiB, so the 168 expert entries lie back to back: the layer's pool starts at expert 0's entry and
    MatDesc::off is relative to it. `sized` False (no packer registered yet): reserve_bytes 0, declared at the fill."""
    ln = f"layers.{L}.experts."
    i = EXPERT_INTER // TP
    f = layer_expert_fmt(L)
    tag = expert_fmt(f)
    eb = expert_entry_bytes(tag) if sized else 0
    src = "{gate,up,down}_proj." + ("weight_{packed,scale,zero_point}" if f == FMT_INT4 else "weight")

    def outs(r):
        o = [Out(ln + "desc", U8, (N_EXPERTS, 3, MATDESC_BYTES), L, NATURAL, RESERVED, RESERVED_FLAG, (0, 0),
                 [f"layers.{L}.mlp.experts.{{e}}.{src}"], f"gx::MatDesc [{N_EXPERTS}][gate, up, down]", tag,
                 N_EXPERTS * 3 * MATDESC_BYTES)]
        for e in range(N_EXPERTS):
            o.append(Out(ln + str(e), U8, (eb,), L, EXPERT_POOL, RESERVED, RESERVED_FLAG, (0, 0),
                         [f"layers.{L}.mlp.experts.{e}.{src}"],
                         f"gate, up rows / down cols [{i * r},+{i}); {FMT_NAME[f]}", tag, eb))
        return o
    return Unit(f"layer {L} routed experts (reserved)", f"routed experts {FMT_NAME[f]} (reserved)", [], outs,
                lambda R, r: {})


def make_plan(ck, bf16_only=False):
    """The units in file order."""
    units = global_units(ck)
    for L in range(N_LAYERS + 1):
        units.append(attn_unit(ck, L, bf16_only))
        units.append(mlp_unit(ck, L))
        if L == MTP:
            units.append(mtp_unit(ck))
    sized = all(f in EXPERT_FORMATS for f in (FMT_INT4, FMT_BF16))
    units += [reserved_unit(L, sized) for L in range(3, N_LAYERS + 1)]
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


EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\."
                       r"(weight|weight_packed|weight_scale|weight_zero_point|weight_shape)$")
EXPERT_PARTS = {FMT_INT4: ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape"), FMT_BF16: ("weight",)}


def coverage(ck, units, bf16_only=False):
    """Every checkpoint tensor is a source of exactly one unit, or a routed expert's tensor behind the reserved
    entries (expert id < 168, the layer's format's parts only, all of them present), or with --bf16-only an INT4
    attention part left for later. Returns (used, reserved, pending)."""
    used = {}
    for u in units:
        for s in u.srcs:
            if s.name in used:
                raise SystemExit(f"{s.name} is a source of two units")
            used[s.name] = u.name
    i4att = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_b|kv_b|o)_proj\.weight_(packed|scale|zero_point|shape)$")
    reserved, pending, unknown = [], [], []
    seen = {}
    for n in ck.names():
        if n in used:
            continue
        m = EXPERT_RE.match(n)
        if m and 3 <= int(m.group(1)) <= MTP and int(m.group(2)) < N_EXPERTS and \
                m.group(4) in EXPERT_PARTS[layer_expert_fmt(int(m.group(1)))]:
            reserved.append(n)
            k = (int(m.group(1)), int(m.group(2)), m.group(3))
            seen[k] = seen.get(k, 0) + 1
        elif bf16_only and i4att.match(n):
            pending.append(n)
        else:
            unknown.append(n)
    want = {(L, e, p): len(EXPERT_PARTS[layer_expert_fmt(L)]) for L in range(3, MTP + 1) for e in range(N_EXPERTS)
            for p in ("gate", "up", "down")}
    short_ = [k for k, v in want.items() if seen.get(k) != v]
    missing = [n for n in used if n not in ck.meta]
    if unknown or missing or short_:
        raise SystemExit(f"coverage: {len(unknown)} checkpoint tensors in no unit (e.g. {unknown[:3]}), "
                         f"{len(missing)} planned sources missing (e.g. {missing[:3]}), {len(short_)} expert "
                         f"matrices without all their parts (e.g. {short_[:3]})")
    return used, reserved, pending


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
            raise SystemExit(f"{path}: not an altar-1 .glw file (magic {hd['magic']!r})")
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
    fs = [os.path.join(ROOT, f) for f in ("glw.py", "convert.py", "sources.py", "int4.py", "crc32c.c")] + \
        [os.path.join(ATTN_TOOLS, "rope_table.py")]
    extra = {F.source for F in EXPERT_FORMATS.values() if F.source}
    if ATTN_INT4 is not None and ATTN_INT4.source:
        extra.add(ATTN_INT4.source)
    return fs + sorted(extra - set(fs))


def tool_version():
    h = hashlib.sha256()
    for f in tool_files():
        h.update(os.path.basename(f).encode() + b"\0" + open(f, "rb").read())
    return h.hexdigest()
