"""Pure-torch stand-ins for inference/kernel.py (tilelang). Same math, fp32 accumulation, no TF32.

TORCHREF_ACT perturbs the GEMM activations for sensitivity runs only (golden data uses the default):
"bf16" skips the fp8 rounding, "g128" quantizes per 128 with fp32 scales (a common FP8 serving
recipe) instead of per 32 with power-of-two scales, and "splitk" keeps the quantization but sums
every GEMM in two K halves (another fp32 accumulation order, as a different kernel would). "exact"
goes further than "bf16": it also skips the in-place fp8 rounding of the KV caches and the fp4 rounding
of the indexer's q/k and latent, so only the stored weights stay quantized.
"""
import os

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def fast_round_scale(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    """2**ceil(log2(amax * max_inv)) via the exponent bits, like the tilelang kernel (fp32)."""
    v = amax.float() * torch.tensor(max_inv, dtype=torch.float32, device=amax.device)
    bits = v.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    e = exp - 127 + (man != 0).to(torch.int32)
    return torch.ldexp(torch.ones_like(v), e.float())


def _fp8_rt(x):  # round-trip through e4m3 (RNE, saturating after clamp)
    return x.to(torch.float8_e4m3fn).float()


def fp4_round(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 (already clamped to [-6, 6]) to the nearest e2m1 value, ties to even mantissa."""
    return x.to(torch.float4_e2m1fn_x2) if False else _fp4_rne(x)


def _fp4_rne(x):
    a = x.abs()
    # representable magnitudes and decision thresholds with ties-to-even
    # 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 ; midpoints 0.25(->0) .75(->1) 1.25(->1) 1.75(->2) 2.5(->2) 3.5(->4) 5(->4)
    r = torch.zeros_like(a)
    r = torch.where(a > 0.25, torch.full_like(a, 0.5), r)
    r = torch.where(a >= 0.75, torch.full_like(a, 1.0), r)
    r = torch.where(a > 1.25, torch.full_like(a, 1.5), r)
    r = torch.where(a >= 1.75, torch.full_like(a, 2.0), r)
    r = torch.where(a > 2.5, torch.full_like(a, 3.0), r)
    r = torch.where(a >= 3.5, torch.full_like(a, 4.0), r)
    r = torch.where(a > 5.0, torch.full_like(a, 6.0), r)
    return torch.copysign(r, x)


ACT_MODE = os.environ.get("TORCHREF_ACT", "")


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    N = x.size(-1)
    if ACT_MODE == "exact":
        return x if inplace else (_QAct(x.float()), None)
    if not inplace and ACT_MODE == "bf16":
        return _QAct(x.float()), None
    if not inplace and ACT_MODE == "g128":
        block_size, scale_fmt = 128, None
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    if scale_fmt is not None:
        s = fast_round_scale(amax, 1.0 / 448.0)
    else:
        s = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32, device=x.device)
    q = (xf / s).clamp(-448.0, 448.0)
    if inplace:
        y = (_fp8_rt(q) * s).flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    # GEMM path: return the dequantized activation in fp32 plus the scale (we fold scales here)
    deq = (_fp8_rt(q) * s).flatten(-2)
    return _QAct(deq), s.squeeze(-1)


class _QAct:
    """Carrier for an already-dequantized fp32 activation (the gemm stand-ins multiply in fp32)."""
    def __init__(self, deq):
        self.deq = deq


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert inplace
    if ACT_MODE == "exact":
        return x
    N = x.size(-1)
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * (2 ** -9))
        s = _fp8_rt(amax / 6.0)
    else:
        amax = amax.clamp_min(6 * (2 ** -126))
        s = fast_round_scale(amax, 1.0 / 6.0)
    q = _fp4_rne((xf / s).clamp(-6.0, 6.0))
    y = (q * s).flatten(-2).to(x.dtype)
    x.copy_(y)
    return x


def dequant_fp8_weight(w, scale, block=32):
    out, inn = w.shape
    s = scale.float().repeat_interleave(block, 0)[:out].repeat_interleave(block, 1)[:, :inn]
    return w.float() * s


def dequant_fp4_weight(w, scale, block=32):
    raw = w.view(torch.uint8)
    lo = (raw & 0x0F).long(); hi = (raw >> 4).long()
    tab = FP4_VALUES.to(w.device)
    vals = torch.stack([tab[lo], tab[hi]], dim=-1).flatten(-2)  # [out, in]
    s = scale.float().repeat_interleave(block, 1)
    return vals * s


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    w = getattr(b, "_deq", None)
    if w is None:
        w = dequant_fp8_weight(b, b_s, block_size)
    y = _mm(a.deq.reshape(-1, a.deq.size(-1)), w)
    return y.view(*a.deq.shape[:-1], -1).to(torch.get_default_dtype())


def _mm(a, w):
    if ACT_MODE == "splitk":
        h = a.size(-1) // 2
        return a[:, :h] @ w[:, :h].t() + a[:, h:] @ w[:, h:].t()
    return a @ w.t()


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    w = dequant_fp4_weight(b, b_s)
    y = _mm(a.deq.reshape(-1, a.deq.size(-1)), w)
    return y.view(*a.deq.shape[:-1], -1).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, q_chunk=256, block=64):
    """q [b,m,h,d] bf16, kv [b,n,d] bf16, idxs [b,m,k] int32 (-1 = none).

    Mirrors the tilelang kernel step by step: keys in index order, 64 at a time; an online softmax
    whose running max starts at -1e30; P = exp(s - running max) summed in fp32 but cast to bf16 for
    the PV product; acc_o rescaled in fp32 when the max moves; exp(sink - max) added to the
    denominator at the end. An all-invalid row gives zeros."""
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    sink = attn_sink.float().view(1, h)
    for bi in range(b):
        for s0 in range(0, m, q_chunk):
            s1 = min(m, s0 + q_chunk)
            idx_all = topk_idxs[bi, s0:s1].long()             # [c, k]
            c, k = idx_all.shape
            qs = q[bi, s0:s1].float()                         # [c, h, d]
            acc_o = torch.zeros(c, h, d, dtype=torch.float32, device=q.device)
            sum_exp = torch.zeros(c, h, dtype=torch.float32, device=q.device)
            mx = torch.full((c, h), -1e30, dtype=torch.float32, device=q.device)
            for t0 in range(0, k, block):
                idx = idx_all[:, t0:t0 + block]
                valid = idx >= 0
                kvg = kv[bi][idx.clamp_min(0)]                # [c, blk, d] bf16
                kvg = torch.where(valid.unsqueeze(-1), kvg, torch.zeros_like(kvg)).float()
                sc = torch.einsum("chd,ckd->chk", qs, kvg) * softmax_scale
                sc = sc.masked_fill(~valid.unsqueeze(1), float("-inf"))
                mx_prev = mx
                mx = torch.maximum(mx_prev, sc.amax(-1))
                rescale = torch.exp(mx_prev - mx)
                p = torch.exp(sc - mx.unsqueeze(-1))
                sum_exp = sum_exp * rescale + p.sum(-1)
                acc_o = acc_o * rescale.unsqueeze(-1) + torch.einsum(
                    "chk,ckd->chd", p.to(torch.bfloat16).float(), kvg)
            sum_exp = sum_exp + torch.exp(sink - mx)
            out[bi, s0:s1] = (acc_o / sum_exp.unsqueeze(-1)).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
