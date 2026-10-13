"""Compare our pure-torch stand-ins (ref/torchref/kernel.py) with the publisher's tilelang kernels
(ref/inference/kernel.py) on random data, op by op. Run in the SGLang image (it has tilelang):

  golden/run_ref.sh py /work/ref/torchref/check_kernels.py
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


K = load("ours", os.path.join(HERE, "kernel.py"))
PK = load("publisher", os.path.join(HERE, "..", "inference", "kernel.py"))
torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)  # as model.py runs; GEMM outputs take it
dev = "cuda"
E8M0 = torch.float8_e8m0fnu


def report(name, a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    exact = (a == b).float().mean().item()
    rel = (d.norm() / b.norm().clamp_min(1e-30)).item()
    print(f"{name:34s} bit-equal {exact*100:7.3f}%  max|d| {d.max().item():.3e}  rel-l2 {rel:.3e}", flush=True)


def rand_scales(rows, cols):
    return (2.0 ** torch.randint(-9, -3, (rows, cols), device=dev).float()).to(E8M0)


def main():
    M, Kd, N = 64, 5120, 1024
    x = (torch.randn(M, Kd, device=dev) * 3).to(torch.bfloat16)

    a = x.clone(); b = x.clone()
    K.act_quant(a, 32, "ue8m0", E8M0, True)
    PK.act_quant(b, 32, "ue8m0", E8M0, True)
    report("act_quant inplace (ue8m0)", a, b)

    q = (torch.randn(M, 8, 128, device=dev)).to(torch.bfloat16)
    a = q.clone(); b = q.clone()
    K.fp4_act_quant(a, 32, True)
    PK.fp4_act_quant(b, 32, True)
    report("fp4_act_quant inplace (e8m0/32)", a, b)
    lat = (torch.randn(M, 512, device=dev) * 0.5).to(torch.bfloat16)
    a = lat.clone(); b = lat.clone()
    K.fp4_act_quant(a, 16, True, scale_dtype=torch.float8_e4m3fn)
    PK.fp4_act_quant(b, 16, True, scale_dtype=torch.float8_e4m3fn)
    report("fp4_act_quant inplace (e4m3/16)", a, b)

    # fp8 GEMM, weights e4m3 [N, K] with 32x32 ue8m0 blocks
    w = (torch.randn(N, Kd, device=dev) * 0.5).to(torch.float8_e4m3fn)
    ws = rand_scales(N // 32, Kd // 32)
    xa, xs = PK.act_quant(x, 32, "ue8m0", E8M0)
    ref = PK.fp8_gemm(xa, xs, w, ws, E8M0, block_size=32)
    qa, qs = K.act_quant(x, 32, "ue8m0", E8M0)
    ours = K.fp8_gemm(qa, qs, w, ws, E8M0, block_size=32)
    report("fp8_gemm (act ue8m0/32, w 32x32)", ours, ref)

    # fp4 GEMM, weights e2m1 [N, K/2] with ue8m0 per 32 along K
    wq = torch.randint(0, 256, (N, Kd // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    wqs = rand_scales(N, Kd // 32)
    ref = PK.fp4_gemm(xa, xs, wq, wqs, E8M0, act_block_size=32)
    ours = K.fp4_gemm(qa, qs, wq, wqs, E8M0, act_block_size=32)
    report("fp4_gemm (act ue8m0/32, w e2m1)", ours, ref)

    # sparse attention, 128 window + 512 compressed keys, some -1. Heads are independent, and the
    # tilelang kernel needs 141 KB of shared memory at 64 heads (GB10 allows ~100 KB): use 16,
    # a rank's share at TP=4
    b_, m, h, d, n, k = 1, 96, 16, 512, 900, 640
    qq = (torch.randn(b_, m, h, d, device=dev) * 0.5).to(torch.bfloat16)
    kv = (torch.randn(b_, n, d, device=dev) * 0.5).to(torch.bfloat16)
    sink = torch.randn(h, device=dev, dtype=torch.float32)
    idx = torch.randint(0, n, (b_, m, k), device=dev, dtype=torch.int32)
    idx[:, :, -50:] = -1
    idx[:, 0, :] = -1  # an all-invalid row
    ref = PK.sparse_attn(qq, kv, sink, idx, d ** -0.5)
    ours = K.sparse_attn(qq, kv, sink, idx, d ** -0.5)
    report("sparse_attn (online softmax, sink)", ours, ref)

    mixes = torch.randn(1, M, 24, device=dev, dtype=torch.float32)
    scale = torch.rand(3, device=dev, dtype=torch.float32)
    base = torch.randn(24, device=dev, dtype=torch.float32)
    r = PK.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    o = K.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    for nm, a_, b2 in zip(("pre", "post", "comb"), o, r):
        report(f"hc_split_sinkhorn {nm}", a_, b2)


if __name__ == "__main__":
    sys.exit(main())
