"""The kernels' tile packers, registered into glw.EXPERT_FORMATS and glw.ATTN_INT4 (glw.py's pluggable INT4/BF16
layouts). The layouts are the kernels' own: the routed experts' INT4-g32 and BF16 blocks are engine/experts/
layout.h's (its python port), the attention GEMV's INT4 tile is engine/attn/tools/pack.py's. This module only adapts
their functions to the interface glw.py calls and says which were found.

`register(test=False)`: registers every layout whose packer is present and returns {what: source file or None}.
`test=True` registers test-only stand-ins instead (natural row-major bytes: INT4 as packed | scale | zero point,
BF16 as is): never for rank files, only for exercising the converter, fill and verifier before the kernels' layouts
exist; a file cut with them says so in its manifest (`packers`)."""
import os

import numpy as np

import glw
import int4

EXPERTS_DIR = os.path.normpath(os.path.join(glw.ROOT, "..", "experts"))


def _matdesc_test(off, fmt):
    import struct
    return struct.pack("<QfI", off, 1.0, fmt) + bytes(glw.MATDESC_BYTES - 16)


def _test_formats():
    def i4_bytes(n, k):
        return glw.align_up(n * k // 2 + n * (k // 32) * 2 + (n // 8) * (k // 32) * 4)

    def i4_pack(t):
        n, k = int4.shape(t)
        b = np.concatenate([t["packed"].view(np.uint8).reshape(-1), t["scale"].view(np.uint8).reshape(-1),
                            t["zp"].view(np.uint8).reshape(-1)])
        return np.concatenate([b, np.zeros(i4_bytes(n, k) - b.size, np.uint8)])

    def i4_unpack(b, n, k):
        b = np.frombuffer(b, np.uint8)
        a, s = n * k // 2, n * (k // 32) * 2
        return dict(packed=b[:a].view(np.int32).reshape(n, k // 8), scale=b[a:a + s].view(np.uint16).reshape(n, k // 32),
                    zp=b[a + s:a + s + (n // 8) * (k // 32) * 4].view(np.int32).reshape(n // 8, k // 32))

    def b_pack(w):
        return np.ascontiguousarray(w).view(np.uint8).reshape(-1)

    def b_unpack(b, n, k):
        return np.frombuffer(b, np.uint8)[:n * k * 2].view(np.uint16).reshape(n, k)

    glw.EXPERT_FORMATS[glw.FMT_INT4] = glw.ExpertFormat(glw.FMT_INT4, "int4 (TEST stand-in)", i4_bytes, i4_pack,
                                                        i4_unpack, lambda off: _matdesc_test(off, glw.FMT_INT4),
                                                        os.path.abspath(__file__))
    glw.EXPERT_FORMATS[glw.FMT_BF16] = glw.ExpertFormat(glw.FMT_BF16, "bf16 (TEST stand-in)", lambda n, k: n * k * 2,
                                                        b_pack, b_unpack, lambda off: _matdesc_test(off, glw.FMT_BF16),
                                                        os.path.abspath(__file__))

    def outs(k, N, K, grouped):
        return [(".weight", glw.U8, (i4_bytes(N, K),), glw.INT4_TILE, (8, 32), 0)]

    def pack(t, N, K, grouped):
        return {".weight": i4_pack(t)}

    def unpack(d, N, K, grouped):
        return i4_unpack(d[".weight"], N, K)
    glw.ATTN_INT4 = glw.AttnInt4(outs, pack, unpack, True, os.path.abspath(__file__))
    return {"experts int4": "TEST", "experts bf16": "TEST", "attention int4": "TEST"}


def source_of(F):
    return os.path.basename(F.source) if F.source else ""


def sources():
    """Every registered packer's module file (hashed into the manifests)."""
    fs = {F.source for F in glw.EXPERT_FORMATS.values() if F.source}
    if glw.ATTN_INT4 is not None and glw.ATTN_INT4.source:
        fs.add(glw.ATTN_INT4.source)
    return sorted(fs)


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _real_experts(path):
    """engine/weights/int4_pack.py (the expert kernels' port of engine/experts/layout.h): INT4 and BF16 blocks."""
    P = _load(path, "int4_pack")

    def i4_pack(t):
        n, k = int4.shape(t)
        return P.pack(dict(P.canon_int4(t["packed"], t["scale"], t["zp"]), N=n, K=k))

    def i4_unpack(b, n, k):
        wp, s, zp = P.untile(b, P.FMT_INT4, n, k)
        return dict(packed=np.ascontiguousarray(wp).view("<i4").reshape(n, k // 8).astype(np.int32),
                    scale=s, zp=zp.astype(np.uint32).view(np.int32))

    def b_pack(w):
        return P.pack(P.canon_bf16(w))

    def b_unpack(b, n, k):
        return P.untile(b, P.FMT_BF16, n, k)

    src = os.path.abspath(path)
    glw.EXPERT_FORMATS[glw.FMT_INT4] = glw.ExpertFormat(glw.FMT_INT4, "int4 (layout.h pack_int4)",
                                                        lambda n, k: P.mat_bytes(P.FMT_INT4, n, k), i4_pack, i4_unpack,
                                                        lambda off: P.matdesc(off, P.FMT_INT4), src)
    glw.EXPERT_FORMATS[glw.FMT_BF16] = glw.ExpertFormat(glw.FMT_BF16, "bf16 (layout.h pack_bf16)",
                                                        lambda n, k: P.mat_bytes(P.FMT_BF16, n, k), b_pack, b_unpack,
                                                        lambda off: P.matdesc(off, P.FMT_BF16), src)


def _real_attn(path):
    """engine/attn/tools/pack.py's int4_tiles / int4_untile: q_b, W_UV and o_proj in the 2,368-byte INT4 blocks
    (one U8 entry [N/8][K/512][2368], layout INT4_TILE); W_UK^T of an INT4 layer in BF16 (dequantised)."""
    P = _load(path, "attn_pack")
    if not hasattr(P, "int4_tiles"):
        return False

    def outs(k, N, K, grouped):
        return [(".weight", glw.U8, (N // 8, K // 512, P.INT4_BLK), glw.INT4_TILE, (1, 32), 0)]

    def pack(t, N, K, grouped):
        b = P.int4_tiles(t["packed"], t["scale"], t["zp"])
        assert b.size == N // 8 * (K // 512) * P.INT4_BLK, (b.size, N, K)
        return {".weight": b.reshape(N // 8, K // 512, P.INT4_BLK)}

    def unpack(d, N, K, grouped):
        p, s, z = P.int4_untile(np.frombuffer(d[".weight"], np.uint8), N, K)
        return dict(packed=p.view(np.int32), scale=s, zp=z.view(np.int32))
    glw.ATTN_INT4 = glw.AttnInt4(outs, pack, unpack, True, os.path.abspath(path))
    return True


def register(test=False, experts=None, attn=None):
    """Registers the kernels' packers: `experts` the int4_pack.py to load (default the in-tree one), `attn` the
    attention pack.py (default attn/tools/pack.py, used if it has int4_tiles). Returns {what: file or None}."""
    if test:
        return _test_formats()
    found = {"experts int4": None, "experts bf16": None, "attention int4": None}
    ep = experts or os.path.join(glw.ROOT, "int4_pack.py")
    if os.path.exists(ep):
        _real_experts(ep)
        found["experts int4"] = found["experts bf16"] = os.path.basename(ep)
    ap = attn or os.path.join(glw.ATTN_TOOLS, "pack.py")
    if os.path.exists(ap) and _real_attn(ap):
        found["attention int4"] = os.path.basename(ap)
    return found
