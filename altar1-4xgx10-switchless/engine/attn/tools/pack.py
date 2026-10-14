"""The engine's per-rank attention-side weight layouts: the reference for the converter (docs/attention.md,
"Weight tiles"). Every function is a lossless re-cut of the source tensors, except W_UK^T of an INT4 kv_b, which is
served as its exact dequantisation (BF16, the contract's weight bit for bit).

altar-1 serves two formats here (docs/attention.md § 2):

* BF16 (q_a, kv_a, the indexer's wk/wq_b/weights_proj, the dense MLPs, and q_b/kv_b/o_proj of layers 0-3, 77, 78):
  row-major [N][K] bf16, re-cut by rows or columns. bf16 values are numpy uint16 bit patterns here.
  Optionally (kBF16T, bf16_tiles) in the GEMV's lane order: same values, contiguous 1 KB warp chunks.
* INT4 (q_b, kv_b, o_proj of layers 4-76): compressed-tensors `pack-quantized`, group 32 along K, asymmetric.
  Canonical (the checkpoint): weight_packed I32 [N][K/8] (k = 8j + i at bits 4i of word j), weight_scale BF16
  [N][K/32], weight_zero_point I32 [N/8][K/32] (row n = 8j + i at bits 4i of word j). Stored nibbles carry +8
  (q = nib - 8, z = znib - 8), so q - z = nib - znib. The weight (docs/model.md § 4.2, compressed-tensors'
  `_dequantize`): W(n, k) = bf16_rne((nib(n, k) - znib(n, k / 32)) * s(n, k / 32)), one rounding.
  Engine INT4 tile: blocks of 8 rows x 512 k, 2368 bytes, block (n8, b) at byte ((n8 * K / 512) + b) * 2368;
  lane l = 4g + t holds row n = 8 n8 + g; unit jj (0..3) is k = 512 b + 128 jj .. + 127, in which the lane's 32
  values are (k = 512 b + 128 jj + 64 q + 16 t + i), q = 0, 1, i = 0..15 (the GEMV's lane map, two 64-k chunks):
    values  at 512 jj + 16 l + 8 q + i / 2, low nibble = even i (the canonical nibbles, +8 kept): 2048 B
    scales  at 2048 + 32 g + 16 (t >> 1) + 2 (2 jj + q): bf16 s(n, 16 b + 4 jj + 2 q + (t >> 1)): 256 B
    zero pts at 2304 + 8 g + 4 (t >> 1) + jj, nibble q (low = q 0): znib of the same group: 64 B
  (lanes t and t ^ 1 share their group: 16 values of a 64-k chunk lie in one group of 32). The values are the
  expert block of docs/reuse.md § 2.3 (NVFP4's value order), so one packer serves both.

glm53's FP8 (zai's format: codes e4m3 [N][K] (uint8 here), weight_scale_inv f32 [ceil(N/128)][K/128];
W = e4m3 * scale) is kept for the kernels' FP8 path and its tests: tiles [N/8][K/64][32 lanes][16 B]; lane 4g + t
holds row 8i + g, k = 64j + 16t + b at byte b; scales f32 [N/rs][K/ks] so that every 8-row x 64-k chunk of a tile
lies in one of zai's blocks.
"""
import numpy as np

TP, D, QL, KL, R, NOPE, VD, IH, ID = 4, 6144, 2048, 512, 64, 192, 256, 32, 128
HEADS = 64 // TP
INT4_BLK_K, INT4_BLK = 512, 2368


# ---- bf16 bit patterns ------------------------------------------------------------------------------------------
def bf16_to_f32(b):
    return (np.asarray(b, np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f):
    """round to nearest even (finite values), as torch's .to(bfloat16)"""
    u = np.ascontiguousarray(f, np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & 1)) >> 16).astype(np.uint16)


# ---- INT4 (compressed-tensors pack-quantized, group 32, asymmetric) ---------------------------------------------
def int4_nibbles(packed):
    """weight_packed [N][K/8] (int32 or uint32) -> stored nibbles uint8 [N][K] (q + 8)"""
    p = np.ascontiguousarray(packed).view(np.uint32)
    sh = np.arange(8, dtype=np.uint32) * 4
    return ((p[:, :, None] >> sh) & 0xF).astype(np.uint8).reshape(p.shape[0], -1)


def int4_zp_nibbles(zp):
    """weight_zero_point [N/8][G] -> stored nibbles uint8 [N][G] (z + 8)"""
    z = np.ascontiguousarray(zp).view(np.uint32)
    sh = np.arange(8, dtype=np.uint32) * 4
    return ((z[:, None, :] >> sh[None, :, None]) & 0xF).astype(np.uint8).reshape(-1, z.shape[1])


def int4_pack_nibbles(nib):
    """inverse of int4_nibbles: uint8 [N][K] -> uint32 [N][K/8]"""
    N, K = nib.shape
    v = nib.reshape(N, K // 8, 8).astype(np.uint32) << (np.arange(8, dtype=np.uint32) * 4)
    return np.bitwise_or.reduce(v, axis=2).astype(np.uint32)


def int4_pack_zp(znib):
    """inverse of int4_zp_nibbles: uint8 [N][G] -> uint32 [N/8][G]"""
    N, G = znib.shape
    v = znib.reshape(N // 8, 8, G).astype(np.uint32) << (np.arange(8, dtype=np.uint32) * 4)[None, :, None]
    return np.bitwise_or.reduce(v, axis=1).astype(np.uint32)


def int4_dequant(packed, scale, zp):
    """The contract's weight, bf16 bits [N][K]: bf16_rne((nib - znib) * s). The f32 product of an integer in
    [-15, 15] and a bf16 is exact, so the one rounding is the final one, as compressed-tensors' bf16 multiply."""
    q = int4_nibbles(packed).astype(np.int32)
    N, K = q.shape
    z = int4_zp_nibbles(zp)[:N].astype(np.int32)
    d = (q.reshape(N, K // 32, 32) - z[:, :, None]).astype(np.float32)
    return f32_to_bf16(d * bf16_to_f32(scale)[:, :, None]).reshape(N, K)


def int4_rows(t, a, n):
    """rows [a, a + n) of a canonical INT4 tensor t = (packed, scale, zp); a and n multiples of 8"""
    p, s, z = t
    assert a % 8 == 0 and n % 8 == 0
    return p[a:a + n], s[a:a + n], z[a // 8:(a + n) // 8]


def int4_cols(t, a, n):
    """columns (k) [a, a + n) of a canonical INT4 tensor; a and n multiples of 32"""
    p, s, z = t
    assert a % 32 == 0 and n % 32 == 0
    return p[:, a // 8:(a + n) // 8], s[:, a // 32:(a + n) // 32], z[:, a // 32:(a + n) // 32]


def int4_cat(ts):
    """canonical INT4 tensors stacked by rows"""
    return tuple(np.ascontiguousarray(np.concatenate([t[i] for t in ts])) for i in range(3))


def int4_tiles(packed, scale, zp):
    """canonical INT4 [N][K] -> the engine tile (module docstring): uint8 [N/8 * K/512 * 2368]"""
    nib = int4_nibbles(packed)
    N, K = nib.shape
    assert N % 8 == 0 and K % INT4_BLK_K == 0, (N, K)
    B = K // INT4_BLK_K
    v = nib.reshape(N // 8, 8, B, 4, 2, 4, 16).transpose(0, 2, 3, 1, 5, 4, 6)  # n8 b jj g t q i
    vb = (v[..., 0::2] | (v[..., 1::2] << 4)).astype(np.uint8).reshape(N // 8, B, 2048)
    s = np.ascontiguousarray(scale, np.uint16).reshape(N // 8, 8, B, 4, 2, 2).transpose(0, 2, 1, 5, 3, 4)  # n8 b g th jj q
    sb = np.ascontiguousarray(s).view(np.uint8).reshape(N // 8, B, 256)
    z = int4_zp_nibbles(zp)[:N].reshape(N // 8, 8, B, 4, 2, 2).transpose(0, 2, 1, 5, 3, 4)  # n8 b g th jj q
    zb = (z[..., 0] | (z[..., 1] << 4)).astype(np.uint8).reshape(N // 8, B, 64)
    return np.ascontiguousarray(np.concatenate([vb, sb, zb], axis=2)).reshape(-1)


def int4_untile(t, N, K):
    """the engine tile -> canonical (packed uint32 [N][K/8], scale uint16 [N][K/32], zp uint32 [N/8][K/32]): an
    independent reader of int4_tiles' layout"""
    B = K // INT4_BLK_K
    blk = np.asarray(t, np.uint8).reshape(N // 8, B, INT4_BLK)
    vb = blk[:, :, :2048].reshape(N // 8, B, 4, 8, 4, 2, 8)  # n8 b jj g t q byte
    v = np.empty(vb.shape[:-1] + (16,), np.uint8)
    v[..., 0::2], v[..., 1::2] = vb & 15, vb >> 4
    nib = v.transpose(0, 3, 1, 2, 5, 4, 6).reshape(N, K)  # n8 g b jj q t i
    s = np.ascontiguousarray(blk[:, :, 2048:2304]).view(np.uint16).reshape(N // 8, B, 8, 2, 4, 2)  # n8 b g th jj q
    scale = s.transpose(0, 2, 1, 4, 5, 3).reshape(N, K // 32)  # n8 g b jj q th
    zb = blk[:, :, 2304:].reshape(N // 8, B, 8, 2, 4)  # n8 b g th jj
    z = np.stack([zb & 15, zb >> 4], axis=-1)  # ... q
    znib = z.transpose(0, 2, 1, 4, 5, 3).reshape(N, K // 32)
    return int4_pack_nibbles(nib), np.ascontiguousarray(scale), int4_pack_zp(znib)


# ---- the per-rank tensors in BF16 and INT4 ----------------------------------------------------------------------
def bf16_xproj(qa, kva, wk, r):
    """Rank r's x-projection rows, BF16 row-major: q_a [512r, +512), kv_a latent [128r, +128), kv_a rope
    [512 + 16r, +16), and wk [32r, +32) when given (full layers): [688 or 656][6144]."""
    parts = [qa[512 * r:512 * r + 512], kva[128 * r:128 * r + 128], kva[512 + 16 * r:512 + 16 * r + 16]]
    if wk is not None:
        parts.append(wk[32 * r:32 * r + 32])
    return np.ascontiguousarray(np.concatenate(parts))


def bf16_rows(w, r, n):
    """rows [n r, n (r + 1)) (q_b: 4096, the indexer's wq_b: 1024, weights_proj: 8, eh_proj: 1536)"""
    return np.ascontiguousarray(w[n * r:n * (r + 1)])


def bf16_cols(w, r, n):
    """columns [n r, n (r + 1)) (o_proj: 4096; the dense down projection: the rank's 3072 intermediates)"""
    return np.ascontiguousarray(w[:, n * r:n * (r + 1)])


def bf16_w_uk(kvb, r):
    """W_UK^T of rank r's 16 heads from a BF16 kv_b [28672][512]: head h's rows 448H .. 448H + 191 transposed to
    [512 (c)][192 (d)], heads stacked: [8192][192]"""
    return np.ascontiguousarray(np.concatenate([kvb[448 * H:448 * H + NOPE].T for H in range(HEADS * r, HEADS * (r + 1))]))


def bf16_w_uv(kvb, r):
    """W_UV of rank r's heads from a BF16 kv_b: head h's rows 448H + 192 .. + 255 [256 (v)][512 (c)], stacked
    [4096][512]"""
    return np.ascontiguousarray(np.concatenate([kvb[448 * H + NOPE:448 * H + NOPE + VD] for H in range(HEADS * r, HEADS * (r + 1))]))


def bf16_gate_up(gate, up, r, inter):
    """rank r's intermediates [inter r, +inter) of gate and up, rows interleaved (2i gate, 2i + 1 up)"""
    a = inter * r
    out = np.empty((2 * inter, gate.shape[1]), np.uint16)
    out[0::2], out[1::2] = gate[a:a + inter], up[a:a + inter]
    return out


def bf16_tiles(w):
    """kBF16T: a BF16 [N][K] matrix (any of the above) in the GEMV's lane order, [N/8][K/64][2 halves][32 lanes]
    [8 values]: in chunk j of tile i, lane 4g + t's half h holds row 8i + g, k = 64j + 16t + 8h .. + 7. The same
    values as the row-major form (the kernels give the same outputs bit for bit); each warp's 64-k chunk is one
    contiguous 1 KB. For K % 128 == 0 this is experts/layout.h's BF16 expert block (2048 B = two chunks)."""
    N, K = w.shape
    assert N % 8 == 0 and K % 64 == 0, (N, K)
    return np.ascontiguousarray(w.reshape(N // 8, 8, K // 64, 4, 2, 8).transpose(0, 2, 4, 1, 3, 5)).reshape(N, K)


def bf16_untile(t, N, K):
    """the inverse of bf16_tiles"""
    return np.ascontiguousarray(t.reshape(N // 8, K // 64, 2, 8, 4, 8).transpose(0, 3, 1, 4, 2, 5)).reshape(N, K)


def int4_q_b(qb, r):
    """INT4 q_b rows [4096r, +4096) -> tile"""
    return int4_tiles(*int4_rows(qb, 4096 * r, 4096))


def int4_w_uv(kvb, r):
    """W_UV of rank r's heads from an INT4 kv_b (canonical [28672][512]): head rows 448H + 192 .. + 255 (multiples
    of 8: the zero-point words cut cleanly), stacked [4096][512] -> tile (one 512-k block a tile)"""
    return int4_tiles(*int4_cat([int4_rows(kvb, 448 * H + NOPE, VD) for H in range(HEADS * r, HEADS * (r + 1))]))


def int4_w_uk(kvb, r):
    """W_UK^T of rank r's heads from an INT4 kv_b: the contract's dequantised weight (bf16) of rows 448H .. +191,
    transposed as bf16_w_uk: the groups of 32 run along the output there (docs/reuse.md § 3)"""
    rows = int4_cat([int4_rows(kvb, 448 * H, NOPE) for H in range(HEADS * r, HEADS * (r + 1))])
    w = int4_dequant(*rows).reshape(HEADS, NOPE, KL)
    return np.ascontiguousarray(w.transpose(0, 2, 1).reshape(HEADS * KL, NOPE))


def int4_o_proj(op, r):
    """INT4 o_proj columns [4096r, +4096) of [6144][16384] -> tile [6144 rows][4096 k]"""
    return int4_tiles(*int4_cols(op, 4096 * r, 4096))


def fp8_from_master(w):
    """zai's FP8 from a BF16 master (torch tensor [N][K]): per 128x128 block scale = f32(amax) * f32(1/448) (a
    multiply by the rounded reciprocal, as zai's release and engine/weights/quant.py; amax / 448 differs by one ulp
    in about half the blocks), codes = e4m3(f32(w) / scale) round-to-nearest-even. Returns (codes uint8 [N][K],
    scales f32 [ceil(N/128)][K/128]). An all-zero block (random test masters only) gets scale 1e-12."""
    import torch
    w = w.float()
    N, K = w.shape
    nb, kb = (N + 127) // 128, K // 128
    pad = torch.zeros(nb * 128, K)
    pad[:N] = w
    blk = pad.view(nb, 128, kb, 128)
    amax = blk.abs().amax(dim=(1, 3))
    scale = (amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)).clamp_min(1e-12)
    q = (blk / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    codes = q.view(torch.uint8).reshape(nb * 128, K)[:N].contiguous()
    return codes.numpy(), scale.numpy().astype(np.float32)


def dequant(codes, scales):
    """f32 W = e4m3 * scale, as the engine computes it."""
    import torch
    v = torch.from_numpy(codes).view(torch.float8_e4m3fn).float().numpy()
    N, K = codes.shape
    s = np.repeat(np.repeat(scales, 128, axis=0)[:N], 128, axis=1)
    return (v * s).astype(np.float32)


def tiles(codes):
    N, K = codes.shape
    assert N % 8 == 0 and K % 64 == 0, (N, K)
    return np.ascontiguousarray(codes.reshape(N // 8, 8, K // 64, 4, 16).transpose(0, 2, 1, 3, 4)).reshape(-1)


def untile(t, N, K):
    return np.ascontiguousarray(t.reshape(N // 8, K // 64, 8, 4, 16).transpose(0, 2, 1, 3, 4)).reshape(N, K)


def xproj(qa, kva, wk, r):
    """Rank r's x-projection: q_a rows [512r, +512), kv_a latent [128r, +128), kv_a rope [512 + 16r, +16), and wk
    [32r, +32) when wk is given. Each argument (codes, scales). Scales per 16 rows: [rows/16][48]."""
    parts = [(qa, 512 * r, 512), (kva, 128 * r, 128), (kva, 512 + 16 * r, 16)]
    if wk is not None:
        parts.append((wk, 32 * r, 32))
    codes = np.concatenate([c[a:a + n] for (c, _), a, n in parts])
    sc = np.concatenate([np.stack([s[(a + 16 * i) // 128] for i in range(n // 16)]) for (_, s), a, n in parts])
    return tiles(codes), sc.astype(np.float32), 16, 128


def rows_block(t, r, n):
    """Rows [n r, n (r+1)) of a 128-aligned split (q_b: n = 4096, the indexer wq_b: n = 1024)."""
    c, s = t
    return tiles(c[n * r:n * (r + 1)]), s[n * r // 128:n * (r + 1) // 128].copy(), 128, 128


def w_uk(kvb, r):
    """W_UK^T of rank r's 16 heads: head h's [512 (c)][192 (d)], heads stacked: [8192][192]; scales [64][3]
    (rs 128 over c, ks 64 over d): [4h + c/128][j] = S[(448 H + 64 j) / 128][c / 128]."""
    c, s = kvb
    codes, sc = [], []
    for h in range(HEADS):
        H = HEADS * r + h
        codes.append(c[448 * H:448 * H + NOPE].T)
        for cb in range(4):
            sc.append([s[(448 * H + 64 * j) // 128][cb] for j in range(3)])
    return tiles(np.ascontiguousarray(np.concatenate(codes))), np.array(sc, np.float32), 128, 64


def w_uv(kvb, r):
    """W_UV of rank r's heads: head h's [256 (v)][512 (c)], stacked [4096][512]; scales [64][4] (rs 64, ks 128):
    [4h + v/64][cb] = S[(448 H + 192 + 64 (v/64)) / 128][cb]."""
    c, s = kvb
    codes, sc = [], []
    for h in range(HEADS):
        H = HEADS * r + h
        codes.append(c[448 * H + NOPE:448 * H + NOPE + VD])
        for vb in range(4):
            sc.append(s[(448 * H + NOPE + 64 * vb) // 128])
    return tiles(np.concatenate(codes)), np.array(sc, np.float32), 64, 128


def cols_block(t, r, n):
    """Columns [n r, n (r+1)) (o_proj: n = 4096 for heads 16r..16r+15; down projections: the rank's intermediates)."""
    c, s = t
    return tiles(np.ascontiguousarray(c[:, n * r:n * (r + 1)])), s[:, n * r // 128:n * (r + 1) // 128].copy(), 128, 128


def gate_up(gate, up, r, inter):
    """Rank r's intermediates [inter r, +inter) of gate and up, rows interleaved (2i gate, 2i+1 up); scales
    [inter/128][2][K/128] (gate's block row, then up's)."""
    (gc, gs), (uc, us) = gate, up
    a = inter * r
    codes = np.empty((2 * inter, gc.shape[1]), np.uint8)
    codes[0::2] = gc[a:a + inter]
    codes[1::2] = uc[a:a + inter]
    sc = np.stack([gs[a // 128:(a + inter) // 128], us[a // 128:(a + inter) // 128]], axis=1)
    return tiles(codes), np.ascontiguousarray(sc, np.float32), 128, 128
