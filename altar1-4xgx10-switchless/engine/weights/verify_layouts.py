"""verify.py's own description of the kernels' INT4 and BF16 tile layouts: un-tilers written out again from the byte
formulas in engine/experts/layout.h's comment ("the engine layout") and engine/attn/tools/pack.py's docstring (the
same 2,368-byte INT4 block), without calling their packers or their inverses. `register(verify)` puts them
into verify.UNTILE_EXPERT, verify.UNTILE_ATTN, verify.MAT_BYTES and verify.DESC_EXPECT under the names packers.py
registers.

The formulas (block (n8, b) at ((n8 * K/kBK) + b) * size; lane l = 4 g + t holds row n = 8 n8 + g; in unit jj the
lane's values are k = kBK b + 128 jj + 64 q + 16 t + i):
  INT4, kBK 512, 2368 B: value nibble of (n, k) at byte 512 jj + 16 l + 8 q + i / 2, low nibble = even i;
    group c = 16 b + 4 jj + 2 q + (t >> 1): scale (bf16) at 2048 + 32 g + 16 (t >> 1) + 2 (2 jj + q),
    zero point nibble (2 jj + q) of the u32 at 2304 + 8 g + 4 (t >> 1), low nibble first.
  BF16, kBK 128, 2048 B: lane l's 16 B at 512 p + 16 l, p = 0..3, hold k = 128 b + 64 (p >> 1) + 16 t + 8 (p & 1) + 0..7.
  MatDesc (experts.h, 64 B): u64 off, f32 s2 (1.0 for INT4 and BF16), u32 fmt, 12 zero words.

`selftest()` checks the fast (reshape) un-tilers against a per-element gather written straight from the formulas,
on random matrices packed by the kernels' own packers (when importable) and by this module's formula-gather."""
import struct

import numpy as np

import glw
import int4


def _int4_fast(b, n, k):
    """[N/8][K/512][2368] bytes -> int4 dict (packed I32 [N][K/8], scale u16 [N][K/32], zp I32 [N/8][K/32])."""
    B, N8 = k // 512, n // 8
    blk = np.frombuffer(b, np.uint8).reshape(N8, B, 2368)
    # values: block (n8, b) byte (jj, g, t, q, x) holds canonical byte (n = 8 n8 + g, k/2 = 256 b + 64 jj + 32 q + 8 t + x)
    v = blk[:, :, :2048].reshape(N8, B, 4, 8, 4, 2, 8)        # n8 b jj g t q x
    wp = v.transpose(0, 3, 1, 2, 5, 4, 6).reshape(n, k // 2)   # n8 g b jj q t x
    packed = np.ascontiguousarray(wp).view("<i4").astype(np.int32)
    # scales: (g, h, j) -> group c = 16 b + 2 j + h
    s = np.ascontiguousarray(blk[:, :, 2048:2304]).view("<u2").reshape(N8, B, 8, 2, 8)   # n8 b g h j
    scale = s.transpose(0, 2, 1, 4, 3).reshape(n, k // 32).astype(np.uint16)              # n8 g b j h
    # zero points: byte (g, h, j // 2), nibble j & 1
    zb = blk[:, :, 2304:].reshape(N8, B, 8, 2, 4)                                         # n8 b g h jb
    zn = np.stack([zb & 15, zb >> 4], axis=-1).reshape(N8, B, 8, 2, 8)                    # n8 b g h j
    znib = zn.transpose(0, 2, 1, 4, 3).reshape(n, k // 32)                                # n8 g b j h
    return dict(packed=packed, scale=np.ascontiguousarray(scale), zp=int4.pack_zp_nibbles(znib))


def _bf16_fast(b, n, k):
    B, N8 = k // 128, n // 8
    x = np.frombuffer(b, np.uint8).view("<u2").reshape(N8, B, 2, 2, 8, 4, 8)  # n8 b ph pl g t e
    return np.ascontiguousarray(x.transpose(0, 4, 1, 2, 5, 3, 6).reshape(n, k)).astype(np.uint16)  # n8 g b ph t pl e


def _int4_gather(b, n, k, pts):
    """(nibble, scale bits, zero-point nibble) of the elements pts, straight from the formulas."""
    b = np.frombuffer(b, np.uint8)
    out = []
    B = k // 512
    for nn, kk in pts:
        n8, g, bb = nn // 8, nn % 8, kk // 512
        jj, q, t, i = (kk // 128) % 4, (kk // 64) % 2, (kk // 16) % 4, kk % 16
        l, h, j = 4 * g + t, t >> 1, 2 * jj + q
        base = (n8 * B + bb) * 2368
        u = (int(b[base + 512 * jj + 16 * l + 8 * q + i // 2]) >> (4 * (i & 1))) & 15
        s = int(b[base + 2048 + 32 * g + 16 * h + 2 * j]) | int(b[base + 2048 + 32 * g + 16 * h + 2 * j + 1]) << 8
        word = int.from_bytes(bytes(b[base + 2304 + 8 * g + 4 * h:base + 2304 + 8 * g + 4 * h + 4]), "little")
        out.append((u, s, (word >> (4 * j)) & 15))
    return out


def _bf16_gather(b, n, k, pts):
    b = np.frombuffer(b, np.uint8)
    out = []
    B = k // 128
    for nn, kk in pts:
        n8, g, bb, r = nn // 8, nn % 8, kk // 128, kk % 128
        ph, t, pl, e = r // 64, (r % 64) // 16, (r % 16) // 8, r % 8
        o = (n8 * B + bb) * 2048 + 512 * (2 * ph + pl) + 16 * (4 * g + t) + 2 * e
        out.append(int(b[o]) | int(b[o + 1]) << 8)
    return out


def _desc(off, fmt):
    return struct.pack("<QfI", off, 1.0, fmt) + bytes(48)


def _i4_bytes(n, k):
    return n // 8 * (k // 512) * 2368


def _b16_bytes(n, k):
    return n // 8 * (k // 128) * 2048


def register(verify):
    for name in ("int4 (layout.h pack_int4)",):
        verify.UNTILE_EXPERT[(glw.FMT_INT4, name)] = _int4_fast
        verify.MAT_BYTES[name] = _i4_bytes
        verify.DESC_EXPECT[name] = _desc
    name = "bf16 (layout.h pack_bf16)"
    verify.UNTILE_EXPERT[(glw.FMT_BF16, name)] = _bf16_fast
    verify.MAT_BYTES[name] = _b16_bytes
    verify.DESC_EXPECT[name] = _desc
    # the attention tile is one U8 entry [N/8][K/512][2368] of the same block (attn/tools/pack.py int4_tiles)
    verify.UNTILE_ATTN["pack.py"] = lambda d, N, K, g: _int4_fast(d[".weight"], N, K)



def selftest(verify, seed=5):
    """Fast un-tilers against the per-element formula gather; with the kernels' packers registered (packers.py),
    pack random matrices with them and un-tile with these."""
    rng = np.random.default_rng(seed)
    bad = 0
    for N, K in ((512, 6144), (6144, 512), (64, 1024)):
        b = rng.integers(0, 256, N // 8 * (K // 512) * 2368, dtype=np.uint8).tobytes()
        t = _int4_fast(b, N, K)
        pts = [(0, 0), (N - 1, K - 1)] + [(int(rng.integers(N)), int(rng.integers(K))) for _ in range(2000)]
        nib, zn = int4.nibbles(t["packed"]), int4.zp_nibbles(t["zp"])
        for (n, k), (u, s, z) in zip(pts, _int4_gather(b, N, K, pts)):
            bad += (nib[n, k] != u) + (t["scale"][n, k // 32] != s) + (zn[n, k // 32] != z)
        b = rng.integers(0, 256, N // 8 * (K // 128) * 2048, dtype=np.uint8).tobytes()
        w = _bf16_fast(b, N, K)
        for (n, k), x in zip(pts, _bf16_gather(b, N, K, pts)):
            bad += w[n, k] != x
    for f, F in glw.EXPERT_FORMATS.items():
        un = verify.UNTILE_EXPERT.get((f, F.name))
        if un is None:
            continue
        for N, K in ((512, 6144), (6144, 512)):
            if f == glw.FMT_INT4:
                t = dict(packed=rng.integers(-2**31, 2**31, (N, K // 8), dtype=np.int64).astype(np.int32),
                         scale=rng.integers(0, 1 << 16, (N, K // 32), dtype=np.int64).astype(np.uint16),
                         zp=rng.integers(-2**31, 2**31, (N // 8, K // 32), dtype=np.int64).astype(np.int32))
                bad += not int4.equal(un(np.asarray(F.pack(t)).tobytes(), N, K), t)
            else:
                w = rng.integers(0, 1 << 16, (N, K), dtype=np.int64).astype(np.uint16)
                bad += not np.array_equal(un(np.asarray(F.pack(w)).tobytes(), N, K), w)
            print(f"  {F.name} [{N}][{K}]: the kernels' packer, un-tiled here", flush=True)
    if glw.ATTN_INT4 is not None and verify.UNTILE_ATTN.get("pack.py"):
        for N, K in ((4096, 2048), (4096, 512), (6144, 4096)):
            t = dict(packed=rng.integers(-2**31, 2**31, (N, K // 8), dtype=np.int64).astype(np.int32),
                     scale=rng.integers(0, 1 << 16, (N, K // 32), dtype=np.int64).astype(np.uint16),
                     zp=rng.integers(-2**31, 2**31, (N // 8, K // 32), dtype=np.int64).astype(np.int32))
            d = {k: np.asarray(v).tobytes() for k, v in glw.ATTN_INT4.pack(t, N, K, False).items()}
            bad += not int4.equal(verify.UNTILE_ATTN["pack.py"](d, N, K, False), t)
            print(f"  attention INT4 tile [{N}][{K}]: pack.py's packer, un-tiled here", flush=True)
    print("verify_layouts selftest:", "ok" if not bad else f"{bad} bad")
    return bad
