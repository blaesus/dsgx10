"""BF16 -> FP8 as zai's FP8 release does it (docs/layout.md § 3, measured with fp8check.py): per 128x128 block,
scale = f32(amax) * f32(1/448) (a multiply by the rounded reciprocal; amax / 448 differs by one ulp in about half
the blocks), codes = e4m3(f32(w) / scale), round to nearest even. The scale is stored as zai stores
weight_scale_inv (W = e4m3 * scale). Reproduces the release bit for bit on every tensor checked."""
import numpy as np
import torch

INV448 = torch.tensor(1.0 / 448.0, dtype=torch.float32)


def fp8_rtn(w):
    """w: [N][K] tensor (bf16 master). Returns (codes uint8 [N][K], scales f32 [ceil(N/128)][K/128]) as numpy.
    Dies on an all-zero block (the release's rule is unknown there; none occurs in GLM-5.3)."""
    N, K = w.shape
    assert K % 128 == 0, (N, K)
    nb, kb = (N + 127) // 128, K // 128
    codes = np.empty((N, K), np.uint8)
    scales = np.empty((nb, kb), np.float32)
    for b in range(nb):  # a block row at a time: ~0.2 GB of f32 temporaries at most
        r0, r1 = 128 * b, min(N, 128 * b + 128)
        blk = w[r0:r1].float().view(r1 - r0, kb, 128)
        amax = blk.abs().amax(dim=(0, 2))
        if bool((amax == 0).any()):
            raise SystemExit(f"an all-zero 128x128 block in block row {b}")
        s = amax * INV448
        q = (blk / s[None, :, None]).to(torch.float8_e4m3fn)
        codes[r0:r1] = q.view(torch.uint8).reshape(r1 - r0, K).numpy()
        scales[b] = s.numpy()
    return codes, scales


def dequant_rel_err(w, codes, scales):
    """Relative RMS error ||deq - w|| / ||w|| of the FP8 tensor against its master, in f64 sums, block row by row."""
    N, K = codes.shape
    num = den = 0.0
    for b in range((N + 127) // 128):
        r0, r1 = 128 * b, min(N, 128 * b + 128)
        v = torch.from_numpy(codes[r0:r1]).view(torch.float8_e4m3fn).float()
        d = v * torch.from_numpy(scales[b]).repeat_interleave(128)[None, :]
        m = w[r0:r1].float()
        num += float(((d - m).double() ** 2).sum())
        den += float((m.double() ** 2).sum())
    return (num / den) ** 0.5 if den else 0.0
