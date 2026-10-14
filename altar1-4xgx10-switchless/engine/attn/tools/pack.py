"""The engine's per-rank attention-side weight layouts, made from zai's FP8 tensors: the reference for the G2
converter (docs/attention.md, "Weight tiles"). Every function is a lossless re-cut of the codes and scales.

zai's format: codes e4m3 [N][K] (uint8 here), weight_scale_inv f32 [ceil(N/128)][K/128]; W = e4m3 * scale.
Engine FP8 tiles: [N/8][K/64][32 lanes][16 B]; lane 4g + t holds row 8i + g, k = 64j + 16t + b at byte b.
Scales f32 [N/rs][K/ks] so that every 8-row x 64-k chunk of a tile lies in one of zai's blocks.
"""
import numpy as np

TP, D, QL, KL, R, NOPE, VD, IH, ID = 4, 6144, 2048, 512, 64, 192, 256, 32, 128
HEADS = 64 // TP


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
