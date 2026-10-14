"""EXL3-style trellis code (K bits a weight, 16x16 tiles), data-free, on the CPU.

The definition follows exllamav3 (turboderp-org/exllamav3, commit 3a513ac9f91aae777f317ca2341b8e445b62c64e,
exllamav3/modules/quant/exl3_lib/quantize.py and exllamav3_ext/quant/*.cuh):

  W is taken as exllamav3 holds it, (in, out) = weight.T, k = in, n = out.
  1. sign vectors su (k) and sv (n), +-1 from a seeded generator;
  2. output-channel scales: r_j = rms of column j, divided by their mean; sv <- sv * r; W <- W / sv;
  3. W <- W @ blockdiag(H128) (Sylvester Hadamard, scaled 1/sqrt(128)) along n;
  4. input-channel scales: s_i = rms of row i; su <- su * s / (-1.24371088); W <- W / su;
  5. W <- blockdiag(H128) @ W along k;
  6. global scale g from a grid search of the trellis error on a sample of tiles (sample_scale_tiles and
     g_scale_search_batch, with the LDLQ drift factor of K); W <- W * g, su <- su / g;
  7. 16x16 tiles, rows 16 at a time, each tile permuted to the tensor-core order (tensor_core_perm) and
     quantised as one tail-biting ring of 256 words by Viterbi (trellis.c);
  8. stored: the packed trellis (16*K u16 words a tile, the branch bits MSB-first in 16 spans of 16 and the
     u16 halves of every u32 swapped, as ext.pack_trellis), suh = f16(su), svh = f16(sv).
  Dequantisation: Q (decoded tiles) -> H128 along k -> * suh -> H128 along n -> * svh, then transposed back.
  Data-free means H = I: no LDLQ, no scale refit, which is exllamav3's fallback_quant path.

Codebooks (the 16-bit word x to an f16 value):
  mul1 (exllamav3's default, "1MAD"): t = (x * 0x83DCD12D) mod 2^32; h = f16(1024 + sum of t's four bytes);
       value = fma_f16(h, f16 0x1EEE, f16 0xC931)
  3inst: t = ((x * 89226354 + 64248484) mod 2^32 & 0x8FFF8FFF) ^ 0x3B603B60; value = f16(lo16) + f16(hi16)
  mcg:   the same with t = (x * 0xCBAC1FED) mod 2^32
"""
import ctypes, hashlib, math, os, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODEBOOK_SCALE = 1.24371088
LDLQ_DRIFT = {1: 1.08, 2: 1.018, 3: 1.004}
_lib = None


def lib():
    global _lib
    if _lib is None:
        src = os.path.join(HERE, "trellis.c")
        tag = hashlib.sha256(open(src, "rb").read()).hexdigest()[:16]
        d = os.path.expanduser("~/.cache/glm53-quant")
        os.makedirs(d, exist_ok=True)
        so = os.path.join(d, f"trellis-{tag}.so")
        if not os.path.exists(so):
            subprocess.check_call(["gcc", "-O3", "-mcpu=native", "-ffp-contract=off", "-fno-fast-math", "-fopenmp", "-fPIC",
                                   "-shared", src, "-o", so + ".tmp"])
            os.replace(so + ".tmp", so)
        _lib = ctypes.CDLL(so)
        _lib.viterbi_tiles.restype = ctypes.c_int
    return _lib


def codebook(name="mul1"):
    """f16 value of every 16-bit word, as float32 (exact)."""
    x = np.arange(65536, dtype=np.uint64)
    if name == "mul1":
        t = (x * 0x83DCD12D) & 0xFFFFFFFF
        s = (t & 255) + ((t >> 8) & 255) + ((t >> 16) & 255) + (t >> 24)
        h = (1024 + s).astype(np.float64)
        kinv = float(np.array([0x1EEE], np.uint16).view(np.float16)[0])
        kbias = float(np.array([0xC931], np.uint16).view(np.float16)[0])
        return (h * kinv + kbias).astype(np.float16).astype(np.float32)
    if name == "3inst":
        t = (x * 89226354 + 64248484) & 0xFFFFFFFF
    elif name == "mcg":
        t = (x * 0xCBAC1FED) & 0xFFFFFFFF
    else:
        raise ValueError(name)
    t = ((t & 0x8FFF8FFF) ^ 0x3B603B60).astype(np.uint32)
    lo = (t & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64)
    hi = (t >> 16).astype(np.uint16).view(np.float16).astype(np.float64)
    return (lo + hi).astype(np.float16).astype(np.float32)


def viterbi(tiles, K, cb, threads=8):
    """tiles: (T, L) float32. Returns (words u16 (T, L), values f32 (T, L))."""
    tiles = np.ascontiguousarray(tiles, dtype=np.float32)
    T, L = tiles.shape
    words = np.zeros((T, L), np.uint16)
    vals = np.zeros((T, L), np.float32)
    cb = np.ascontiguousarray(cb, dtype=np.float32)
    P = ctypes.POINTER
    lib().viterbi_tiles(tiles.ctypes.data_as(P(ctypes.c_float)), T, L, K, cb.ctypes.data_as(P(ctypes.c_float)),
                        words.ctypes.data_as(P(ctypes.c_uint16)), vals.ctypes.data_as(P(ctypes.c_float)), threads)
    return words, vals


def check_ring(words, K):
    """Every word's top 16-K bits equal the previous word's low 16-K bits, wrapping around the tile."""
    w = words.astype(np.int64)
    return bool(np.all((w >> K) == (np.roll(w, 1, axis=1) & ((65536 >> K) - 1))))


# ---- layout -----------------------------------------------------------------------------------------

def tensor_core_perm():
    p = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        r1, r2, r3 = r0 + 1, r0 + 8, r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        p[t * 8:t * 8 + 8] = [r0 * 16 + c0, r1 * 16 + c0, r2 * 16 + c0, r3 * 16 + c0,
                              r0 * 16 + c1, r1 * 16 + c1, r2 * 16 + c1, r3 * 16 + c1]
    return np.array(p)


PERM = tensor_core_perm()
PERM_I = np.argsort(PERM)


def to_tiles(W):
    """(k, n) -> (k/16, n/16, 256) in tensor-core order."""
    k, n = W.shape
    t = W.reshape(k // 16, 16, n // 16, 16).transpose(0, 2, 1, 3).reshape(k // 16, n // 16, 256)
    return t[..., PERM]


def from_tiles(T):
    a, b, _ = T.shape
    return T[..., PERM_I].reshape(a, b, 16, 16).transpose(0, 2, 1, 3).reshape(a * 16, b * 16)


def hadamard(n=128):
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


def had_rows(W, d=128):
    """blockdiag(H) @ W along axis 0 (k)."""
    k, n = W.shape
    return np.einsum("ij,bjn->bin", hadamard(d), W.reshape(k // d, d, n)).reshape(k, n)


def had_cols(W, d=128):
    """W @ blockdiag(H) along axis 1 (n); H is symmetric."""
    k, n = W.shape
    return np.einsum("kbj,ji->kbi", W.reshape(k, n // d, d), hadamard(d)).reshape(k, n)


# ---- packing (ext.pack_trellis) ---------------------------------------------------------------------

def pack(words, K):
    """words (..., 256) u16 -> (..., 16*K) u16, exllamav3's bit order."""
    br = (words.astype(np.uint32) & ((1 << K) - 1)).reshape(-1, 16, 16)          # 16 spans of 16
    bits = ((br[..., None] >> np.arange(K - 1, -1, -1, dtype=np.uint32)) & 1).astype(np.uint8)
    bits = bits.reshape(br.shape[0], 16, 16 * K)                                    # span = K u16 words
    u16 = np.packbits(bits.reshape(br.shape[0], 16, K, 16), axis=-1, bitorder="big")
    u16 = (u16[..., 0].astype(np.uint16) << 8) | u16[..., 1]                        # (T, 16, K)
    u16 = u16.reshape(br.shape[0], 16 * K)
    u16 = u16.reshape(-1, 8 * K, 2)[..., ::-1].reshape(br.shape[0], 16 * K)         # SWAP16 per u32
    return np.ascontiguousarray(u16).reshape(words.shape[:-1] + (16 * K,))


def unpack(packed, K):
    """Inverse of pack: the 256 branch values of each tile, then the 16-bit words of the ring."""
    shp = packed.shape[:-1]
    u16 = packed.reshape(-1, 8 * K, 2)[..., ::-1].reshape(-1, 16, K)
    by = np.stack([(u16 >> 8).astype(np.uint8), (u16 & 255).astype(np.uint8)], -1)
    bits = np.unpackbits(by.reshape(-1, 16, 2 * K), axis=-1, bitorder="big").reshape(-1, 16, 16, K)
    br = (bits.astype(np.uint32) << np.arange(K - 1, -1, -1, dtype=np.uint32)).sum(-1).reshape(-1, 256)
    words = np.zeros_like(br)
    for j in range((16 + K - 1) // K):
        words |= np.roll(br, j, axis=1) << (K * j)
    return (words & 0xFFFF).astype(np.uint16).reshape(shp + (256,))


# ---- the quantiser ----------------------------------------------------------------------------------

def sample_scale_tiles(Wr, width=3):
    k, n = Wr.shape
    tk, tn = k // 16, n // 16
    w4 = Wr.reshape(tk, 16, tn, 16)
    dl = max(tk, tn)
    ii = np.repeat(np.arange(dl), width)
    ww = np.tile(np.arange(width), dl)
    kk, nn = ii % tk, (ii + ww) % tn
    nx = max(8, (dl * width) // 16)
    ms = (w4 ** 2).mean(axis=(1, 3)).reshape(-1)
    nx = min(nx, (ms.shape[0] + 1) // 2)
    hi = np.argsort(-ms, kind="stable")[:nx]
    lo = np.argsort(ms, kind="stable")[:nx]
    x = np.concatenate([hi, lo])
    tk_i, tn_i = np.concatenate([kk, x // tn]), np.concatenate([nn, x % tn])
    t = w4[tk_i, :, tn_i, :].reshape(-1, 256)
    return np.ascontiguousarray(t[:, PERM])


def g_scale_search(tiles, K, cb, threads):
    def mse(ts, s):
        _, v = viterbi(ts * s, K, cb, threads)
        return float(((v.astype(np.float64) / s - ts) ** 2).mean())
    coarse = [0.1 + 0.2 * i for i in range(10)]
    sub = tiles[::3]
    c = coarse[int(np.argmin([mse(sub, s) for s in coarse]))]
    step = 0.075
    fine = [c + step * (i - 2) for i in range(5)]
    m = [mse(tiles, s) for s in fine]
    b = int(np.argmin(m))
    off = 0.0
    if 0 < b < 4:
        y0, y1, y2 = m[b - 1], m[b], m[b + 1]
        den = y0 - 2 * y1 + y2
        off = max(-0.5, min(0.5, 0.5 * (y0 - y2) / den if den > 0 else 0.0))
    return max(fine[b] + off * step, 0.01), m[b]


def quantise(weight, K, cb_name="mul1", seed=0, threads=8, cb=None):
    """weight: torch-layout (out, in) array. Returns dict with packed trellis, suh, svh (f16), g, stats."""
    if cb is None:
        cb = codebook(cb_name)
    W = np.asarray(weight, dtype=np.float64).T.copy()        # (k = in, n = out)
    k, n = W.shape
    rng = np.random.Generator(np.random.PCG64(seed))
    su = np.where(rng.standard_normal(k) >= 0, 1.0, -1.0)
    sv = np.where(rng.standard_normal(n) >= 0, 1.0, -1.0)
    r = np.sqrt((W ** 2).mean(axis=0))
    r = r / r.mean()
    r[np.abs(r) < 1e-30] = 0.1
    sv = sv * r + 1e-10
    W = had_cols(W / sv)
    s = np.sqrt((W ** 2).mean(axis=1))
    s[np.abs(s) < 1e-30] = 0.1
    su = su * s / (-CODEBOOK_SCALE) + 1e-10
    W = had_rows(W / su[:, None])
    tiles = sample_scale_tiles(W) * LDLQ_DRIFT.get(K, 1.0)
    g, gmse = g_scale_search(tiles.astype(np.float32).astype(np.float64), K, cb, threads)
    W = W * g
    su = su / g
    T = to_tiles(W).astype(np.float32)
    words, vals = viterbi(T.reshape(-1, 256), K, cb, threads)
    assert check_ring(words, K)
    words = words.reshape(T.shape)
    return {"trellis": pack(words, K), "suh": su.astype(np.float16), "svh": sv.astype(np.float16), "K": K,
            "g": g, "shape": tuple(np.shape(weight)), "codebook": cb_name}


def dequantise(q, cb=None):
    """Packed trellis + suh + svh -> torch-layout (out, in) float64."""
    if cb is None:
        cb = codebook(q["codebook"])
    words = unpack(q["trellis"], q["K"])
    Q = from_tiles(cb[words].astype(np.float64))
    Q = had_rows(Q) * q["suh"].astype(np.float64)[:, None]
    Q = had_cols(Q) * q["svh"].astype(np.float64)[None, :]
    return Q.T
