# Findings

What altar-1's measurements say, as a delta from glm53's [findings.md](../../glm53-4xgx10-switchless/docs/findings.md):
only what is new or different is here. Everything is **measured** unless marked *estimate*; each number's job,
date and sources are in [experts.md § 9](experts.md#9-runs) and the [experiments index](experiments.md).

## A0: the INT4 weight and the decode GEMV (2026-10-14)

**The rounding point.** compressed-tensors 0.19.0 (the checkpoint was written by 0.1.dev537+g1cee215) dequantises
in `quantization/lifecycle/forward_helpers.py` lines 554–559: `x_q.to(scale.dtype) - zero_point.to(scale.dtype)`,
then `* scale`, both in BF16 — the difference is exact, the product rounded once to nearest even (0.18.0 has the
same code at lines 562–567, quoted in [model.md](model.md)). Values unpack along K (`packed_dim` 1) and zero points
along N (`packed_dim` 0), both with the offset 8 (`compressors/pack_quantized/base.py`, `helpers.py`
`unpack_from_int32`, line 104).

**The library check** (`tools/ct_check.py`, every element; `~/dsgb10-data/altar1-experts/a0/ct_check.tsv`, not in
the repo): five INT4 tensors, 62,914,560 values, decompressed by the library and by `int4_pack.py`: **0 mismatches**
in q, z and the weight; the rank slices 0 mismatches; BF16 layer 3 expert 0 gate and down 0 mismatches. 24.1–25.4 %
of the products are inexact in bf16, so the rounding point matters: the alternative bf16(u·s) − bf16(u_z·s)
disagrees on 6.58–6.66 M values a tensor. Zero-point nibbles seen: 1–14 (2–13 on most tensors). Real scales
2^-15..2^-5; the largest |W| 0.39.

**The decode GEMV** was measured on the engine's own kernels (`bench_experts`) rather than `micro/` prototypes,
since what ships is what counts. Default variant, four runs each (both orders):

| Rows | INT4 % of 235 | BF16 % of 235 | target ≥ 93 % |
|---:|---|---|---|
| 1 | 90.2–92.3 (best variant 94.6) | 97.3–98.1 | INT4 misses |
| 6 | 94.6–95.6 | 99.0–99.3 | both meet |
| 16 | 89.7–90.6 (prefill order wk1: 96.6–97.9) | 96.7–97.0 | INT4 misses on the default |

The new finding is at 16 rows: **glm53's default K split (wk6, decision 11) is the slow one for INT4**, by 7 points,
while the split that equals prefill bit for bit (wk1, `kUpPrefillOrder`) is fast. With wk6 each warp owns two
512-k INT4 blocks and the two-ahead prefetch has nothing left to overlap the decode's ALU work (*estimate*, from
the BF16 kernel, whose warps own eight 128-k blocks, not dropping). glm53 chose wk6 for NVFP4 at 1 row; for altar-1
reversing decision 11 would cost 2–4 points at 1 row and gain 7 at 16. Not changed: it changes output bits.

## A3e: kernels (2026-10-14)

- **INT4 = BF16.** The INT4 kernels equal the BF16 kernels on the dequantised matrices bit for bit (h and outputs,
  13 decode variants × 5 row counts, prefill up to 2500 rows, a real layer): one template, the same k order, and
  the `i4_dec16` decode (prmt, `| 0x4300`, `hsub2`, `hmul2`) equal to the library on every (u, u_z) pair with 145
  scales.
- **The prefill h smoke share is lower on full-mantissa data**: 99.07–99.51 % of h bit-exact against double, against
  glm53's ≥ 99.5 % bar on NVFP4 (2–4 significant bits), with 0 outside the f32-sum range. Prefill sums 6144 k in one
  chain, decode in six; the bar is now 98.5 %.
- **Prefill.** INT4 39.7 TFLOPS at 8k (31.1 ms a layer), level with glm53's NVFP4 (38.0); the down is bound by its
  ordered f32 adds (3.5 ms without them, 15.4 with). BF16 24.3 TFLOPS at 8k, bound by weight re-reads per 128-pick
  tile (12.7 GB, *estimate*, 54 ms at 235 GB/s against 50.8 ms measured): only 2 layers (+ MTP).
- **The BF16 shared expert** through attn's GEMV: 85 µs a layer at 1–16 rows (93.7–94.7 % of 235) with split-K 4 on
  gate/up; 117–141 µs (57–69 %) without split. ~6.4 ms a token over 75 MoE layers.
- **torch.topk.** On CUDA the selected set among ties is always the lowest ids (the engine's rule); the returned order
  with `sorted=False` is not descending and not the engine's pick order, which matters only for the order of the f32
  sum of the eight scores. On the CPU the set differs on ties at the 8th place (662 of 2000 rows match lowest ids).
  Ties are real with f32 keys near 28 (ulp 1.9e-6): one in 4096 random rows.
