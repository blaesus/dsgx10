# The routed experts (A0, A3e): a delta from glm53's

What `engine/experts/` computes for one rank of altar-1 at TP=4, the layout the converter emits, how it was tested
and what it measured, written as the changes against glm53's
[docs/experts.md](../../glm53-4xgx10-switchless/docs/experts.md) (cited below as "glm53 § n"). Written 2026-10-14.
The API and every constant are in [engine/experts/experts.h](../engine/experts/experts.h); the layout's executable
specification is [engine/experts/layout.h](../engine/experts/layout.h), and its Python port, which the converter
calls, is [engine/weights/int4_pack.py](../engine/weights/int4_pack.py).

## Contents

1. [What is unchanged](#1-what-is-unchanged)
2. [What changed](#2-what-changed)
3. [Formats and the layout the converter emits](#3-formats-and-the-layout-the-converter-emits)
4. [The kernels: one template, three formats](#4-the-kernels-one-template-three-formats)
5. [Routing at 168](#5-routing-at-168)
6. [Numerics contract](#6-numerics-contract)
7. [Tests](#7-tests)
8. [Measurements](#8-measurements)
9. [Runs](#9-runs)
10. [Failures and what is left](#10-failures-and-what-is-left)
11. [Tuning: the fused BF16 shared expert and k_up's cross-task prefetch](#11-tuning-the-fused-bf16-shared-expert-and-k_ups-cross-task-prefetch)

## 1. What is unchanged

- **Shapes** per rank: gate and up [512][6144], down [6144][512] (glm53 § 1); top-8; ×2.5; ≤ 16 decode rows;
  ≤ 8192-row prefill chunks.
- **Decode** (glm53 § 4): `k_up` (union in expert-id order, gate+up fused, SwiGLU, WK warps splitting K, raw blocks
  streamed two ahead, bf16 m16n8k16 MMAs) and `k_down` (expert-id order, `fmaf(s2·k·w, c, acc)`), PDL, the dynamic
  task counter, fused routing (`DecodeCfg::logits`) and posted outputs (`DecodeCfg::post`). The same variants, names
  and defaults: up nt1 wk6 p2, down nt2 p1. **glm53 decision 11 is kept**: decode's default K order is not
  prefill's; `DecodeCfg::up = kUpPrefillOrder` (6, "nt1 wk1 p2") and the other wk1 variants equal prefill bit for bit.
- **Prefill** (glm53 §§ 5, 11, 12): bucketing, the warp-specialised gate/up `k_pup_ws` (default, `up = 1`) and
  `k_pup` (`up = 0`), the chained down `k_pdown_c` (default, `down = 1`) and `k_pdown_r` + `k_combine`
  (`down = 0`), the same tiles, swizzles and the ordered f32 adds in expert-id order.
- **The numerics contract** (glm53 § 6): f32 sums of exact bf16 products, gate and up rounded to bf16, h =
  bf16(bf16(silu(g))·u), each expert's down partial added in f32 in expert-id order, one rounding after the
  all-reduce.
- **Batch invariance and determinism** (glm53 decision 9; altar1 DESIGN decision 8).

## 2. What changed

| Where | glm53 | altar1 |
|---|---|---|
| `experts.h` | `kE` 256; formats NVFP4 (0), S3 (1); `MatDesc` with S3 prmt tables | `kE` 168, `kEWords` 6; formats **NVFP4 (0), INT4 (2), BF16 (3)**; tag 1 (S3) retired and refused like any unknown tag; `MatDesc {off, s2, fmt, rsv[12]}`; `LayerW::fmt`, the layer's one format |
| `formats.cuh` | NVFP4 widen, S3 prmt decode, the FP4 activation quantiser | NVFP4 widen kept; **`i4_dec16`** (the contract's INT4 weight, § 3); format traits `FmtNVFP4`, `FmtINT4`, `FmtBF16` (block size, k a block, `Raw`, `load`, `dec`, the epilogue factor `kMul`); `route_warp` over 6 slots a lane, ids ≥ 168 keyed −∞ |
| `layout.h` | NVFP4 and S3 packers and readers | NVFP4, **INT4 and BF16** packers, readers (`read_i4_parts`, `read_i4`, `read_b16`, `read_value`), `canon_i4`, `make_desc(off, s2, fmt)`, `desc_ok`, `scale_bytes_check` (INT4: bf16 finite and positive) |
| `decode.cu` | NVFP4/S3 chosen per matrix at run time | `k_up<F, NT, WK, P, R16>`, `k_down<F, NT, P, R16>` per format; the host dispatches on `LayerW::fmt` and dies on an unknown one; each kernel traps on a descriptor of another format; the fused FP8 shared expert optional (`SharedW::fused()`): without it `ys` is the caller's (the BF16 shared expert's output), forwarded into a posted collective's buffer |
| `prefill.cu` | NVFP4/S3 per matrix; the FP4-activation path; `up = 2`, `down = 2` | the routed kernels templated on F; **the FP4-activation path removed** (it changed outputs), and the two slower variants (`k_pup_w2`, `k_pdown_cw`); `k_scan` launched with `kE` threads (glm53's 256 would read past 168); the fused FP8 shared expert optional |
| tests, bench | NVFP4/S3 patterns | per-format layers, INT4 = BF16(dequantised), `--real` int4_pack dumps, the BF16 shared expert through attn's GEMV (§§ 7–8) |
| `weights/kcheck.cu`, `weights/tests/expert_fill_test.cu` | S3 descriptor tables | `desc_ok` on every descriptor, one format a layer and the layer's own (BF16 on 3, 77, 78 — NVFP4 also allowed on 78 —, INT4 on 4–76), INT4 scales with `--blocks`; the fill test reads int4_pack.py's dumps |

**The shared expert.** altar-1's shared expert is BF16 (gate, up, down), so it no longer rides the routed kernels as
GLM-5.3's FP8 one did. It runs through attn's BF16 GEMV (`glm::gemv(false, …)`: gate/up interleaved with the
SwiGLU epilogue, then down to f32 partials), and its f32 output is the `ys` the routed step forwards (§ 8.3). The
fused FP8 path stays in the kernels, tested with an NVFP4 layer (`t-q4`). Since the tuning of § 11 the BF16 shared
expert is fused into `k_up`/`k_down` by default (bit-identical to the GEMVs; `ys` still a separate f32 partial).

## 3. Formats and the layout the converter emits

A matrix [N][K] is stored as blocks of 8 rows × kBK k, [N/8][K/kBK]; lane l = 4g + t of a warp holds row g of the
block and, in unit jj (128 k), the 32 values k = 128 jj + 64 q + 16 t + i (q = 0, 1; i < 16): two groups of 16 that
are the B fragments of four m16n8k16 MMAs. Every format feeds the MMAs the same lane map in the same k order, so a
matrix's sums depend only on its decoded bf16 values.

| Format | kBK | Block | Contents |
|---|---:|---:|---|
| NVFP4 | 512 | 2304 B | as glm53 § 2 |
| **INT4** (layers 4–76) | 512 | **2368 B** | 4 × [32 lanes][16 B] the checkpoint's stored nibbles u = q + 8 (unit jj at 512 jj + 16 l, byte 8q + i/2, low nibble = even i: the checkpoint's own byte order); 256 B of bf16 scales, group c' (k 32c′..) of row g at 2048 + 32g + 16(c′ & 1) + 2(c′ >> 1); 64 B of zero-point nibbles u_z = z + 8, nibble c′ >> 1 of the u32 at 2304 + 8g + 4(c′ & 1) |
| **BF16** (layers 3, 77, 78) | 128 | **2048 B** | 4 × [32 lanes][16 B]: the lane's values 8p..8p+7 at 512p + 16l, the checkpoint's bits |

An INT4 matrix is 1,818,624 B a rank (444 pages of 4 KiB), an expert 5,455,872 B; a BF16 matrix 6,291,456 B.
`MatDesc::s2` is exactly 1 for INT4 and BF16 (`desc_ok` refuses anything else). The rank slices: gate/up rows
[512r, +512) and their zero-point words [64r, +64); down input columns [512r, +512), groups [16r, +16): every cut
is group- and word-aligned (`int4_pack.rank_slice`).

**The INT4 weight (the contract, DESIGN decision 2).** W = bf16_rne((u − u_z)·s): compressed-tensors' `_dequantize`
(0.19.0 `quantization/lifecycle/forward_helpers.py` 554–559: `x_q.to(scale.dtype)`, `- zero_point.to(scale.dtype)`,
`* scale`, all in the scale's dtype, BF16) after `unpack_from_int32` of values (`packed_dim` 1) and zero points
(`packed_dim` 0) with the offset 8 (`compressors/pack_quantized/base.py`, `helpers.py` 104). The difference is exact;
the product of a ≤ 4-bit integer and a bf16 scale has ≤ 12 significant bits and is rounded once.
The device does the same in bf16x2 arithmetic (`i4_dec16`):

- `prmt` puts nibbles 2j and 2j+1 into the low bits of the two halves; `| 0x4300` makes the bf16 128 + u exactly
  (ulp 1 in [128, 256));
- `__hsub2` with 128 + u_z gives u − u_z exactly;
- `__hmul2` by the scale (both halves) rounds the exact product once to nearest even, as the library.

The host emulation of these bf16 operations and the device both equal the contract on every (u, u_z) pair with 145
scales, subnormal and overflowing ones included (§ 7).

## 4. The kernels: one template, three formats

`decode_fmt<F>` and `prefill_routed<F>` launch the kernels of the layer's format; the format traits hide the
difference:

- **K blocking.** INT4 and NVFP4 blocks are 512 k (12 along gate/up's K, 1 along down's); BF16 blocks are 128 k
  (48 and 4). The decode warps split K by the same k ranges in every format (wk6: 1024 k a warp), and prefill's
  k-steps of 64 walk a raw block in KS = kBK/64 steps (8 or 2), the prefetch running one block group ahead. So the
  INT4 kernel's MMAs see the same values in the same order as the BF16 kernel's on the dequantised matrix — the
  equivalence test of § 7 checks it bit for bit, every variant.
- **The epilogue factor** `F::kMul` (NVFP4's 2^30 from its widening; 1 for INT4 and BF16) times `MatDesc::s2`.
- **INT4 `Raw`** a lane: the four 16-B value units, its 16 B of scales (its eight groups) and one u32 of zero points,
  84 B; decode of a group: one `prmt`, a mask-or, `hsub2`, `hmul2` per value pair. Registers (ptxas, sm_121a):
  the default `k_up<INT4, 1, 6, 2>` 132 a thread, no spill; the non-default `k_up<INT4, 2, 1, 2>` spills
  136–192 B (not used by default). Prefill's default kernels spill nothing; `k_pdown_c<INT4>` has a 16-B stack frame.

## 5. Routing at 168

`route()` and the fused `route_warp`: as glm53 § 3 with 168 experts, 6 keys a lane (ids ≥ 168 keyed −∞), ties to
the lower id, picks in descending key order, the sum tree of glm53. The test matches the CPU reference bit for bit on
random rows, rows with many tied logits, and rows with a three-way tie at the 8th place (§ 7).

**torch.topk's order** (`tools/probe_topk_ties.py`, job C, 2026-10-14, torch 2.11.0+cu130). The router calls
`topk(sel, 8, sorted=False)` on [T, 168] f32:

| Case, 4096 rows | CUDA (GB10): set = lowest ids among ties | CUDA: returned order | CPU: set = lowest ids |
|---|---|---|---|
| distinct keys (28 + sigmoid) | 4096 | sorted=False: not descending (1 row) | 2000 of 2000 |
| integer keys 0..5 (ties everywhere) | 4096 | sorted=False: (key desc, id asc) in all | 0 of 2000 |
| a 3-way tie at the 8th place | 4096 | sorted=False: not descending; sorted=True: (key desc, id asc) in all | 662 of 2000 |

So on CUDA the **selected set** is the engine's in every case (lowest ids win ties); the order of the eight ids is
not, and it only matters through the order of the f32 sum of the eight scores (glm53 decided the sum's tree from
torch's CUDA reduction on DeepSeek, `route_sum8`). On the CPU, torch picks a different set among ties at the 8th
place in two thirds of the rows: a CPU reference and the engine can disagree on exact ties. With a bias of ~28 the
keys' ulp is 2^-19 ≈ 1.9e-6, and the probe's "distinct" rows already held one exact tie in 4096.

## 6. Numerics contract

glm53 § 6 with the weight column changed:

| Point | Engine |
|---|---|
| weights | INT4: bf16_rne((u − u_z)·s), the library's value (§ 3); BF16: the checkpoint's bits; NVFP4: as glm53 |
| everything else | as glm53 § 6 |

**INT4 = BF16.** The INT4 kernels are bit-equal to the BF16 kernels run on the dequantised matrices (tested,
every decode variant and row count, both prefill paths): an INT4 layer's result depends only on its contract
values. The FP4-activation path, which changed outputs, is gone; no default path changes outputs.

## 7. Tests

`test_experts` (GPU through `run.sh`); `--cpu` host only; `--fmt` the synthetic layers' formats; `--real DIR` an
int4_pack.py dump. All pass in job B (2026-10-14, sources `5275133f77797abf`, commit 31a288e) except two tolerance
checks on the random-bits dump (§ 10).

| Check | Result |
|---|---|
| CPU: NVFP4 widen; INT4 `i4_dec16` host emulation vs the contract, every (u, u_z) with 145 scales | 0 of 74,240 mismatches (30,588 products inexact in bf16) |
| CPU: the contract = one rounding of the exact product, 37,120 cases (with overflow to ±inf) | all |
| CPU: FP8 widen; descriptor verdicts (each format at its tag and at the others, S3's tag 1, an unknown tag, misalignment, s2, reserved words); packers vs canonical readers, every element (NVFP4, INT4, BF16, FP8, INT4 → BF16); scale checks (a negative INT4 scale refused) | 0 wrong |
| CPU: routing reference at 168 (order, ties, sum; 128 tied neighbours) | ok |
| GPU dequantisation INT4 / BF16 / NVFP4, gate and down shapes | 0 of 3,145,728 each |
| GPU INT4 with 145 scales (subnormal 0x0001.., the smallest and largest normals, overflow 0x7F7F) | 0 of 9,502,720 |
| route 168: random; many tied logits with equal bias; 3-way tie at the 8th place | ids and weight bits identical |
| decode, INT4 / BF16 / NVFP4 (+ fused FP8 shared) layers, M = 1, 3, 5, 6, 8, 9, 16 | h ≥ 99.9 % bit-exact against double, 0 outside the f32-sum range; routed out ≤ 2.6e-7 of Σ\|terms\|; `ys` untouched without a fused shared expert |
| decode batch invariance, determinism, every variant (same K split: identical; others: inside the range) | ok |
| prefill T = 1, 37, 300, 2500 per format | h 99.07–99.51 % bit-exact (INT4 and BF16; see § 10), 0 outside the range; outputs ≤ 3.4e-7 of Σ\|terms\|; default = 2026-10-13 kernels = narrow passes / side stream, bit for bit; vs decode ≤ 1.3e-3 of the row's largest output; **= decode wk1 (nt1 wk1 p2, nt1 wk1 p4, nt2 wk1 p2) bit for bit**; determinism; chunk invariance |
| **INT4 = BF16(dequantised)**, synthetic: decode M = 1, 2, 6, 9, 16 × 13 variants; prefill T = 1, 37, 300, 2500 × (default, up 0 down 0) | h and outputs 0 differ (e.g. 10,240,000 h and 15,360,000 outputs at T = 2500) |
| real dumps (§ 9): GPU decode of every value = the contract's values (int4_pack.value_bits) | 0 of 28,311,552 (L3 BF16), 47,185,920 (L4 INT4), 18,874,368 (L40 INT4), 28,311,552 (random) |
| real dumps: the Python packer = layout.h's, byte for byte; MatDesc; readers | 0 differ |
| real layers (L3 BF16 experts 0–2 rank 1; L4 INT4 0–3, 167 rank 0; L40 INT4 57, 58 rank 2): decode and prefill as above; INT4 = BF16(dequantised) | all pass; equivalence 0 differ |

Synthetic data: 10 stored experts mapped over the 168 ids; INT4 nibbles and zero points uniform, scales bf16 in
[2^-9, 2^-5) with random mantissas (real experts: 2^-15..2^-5, mostly 2^-10..2^-6, sampled on L4 e0, e167 and L40
e57); BF16 values with exponents 2^-12..2^-4.

## 8. Measurements

**Conditions.** One GB10 (rank 0's) through `ring.sh`, production paused; the job's MEM line 10 GiB used; 2190 MHz
SM clock unpinned; the streaming read 232.1–233.1 GB/s. Decode: a pool of ≥ 2.2 GB cycled, graphs of 64 steps, the
median of 7 replays; the step = route + `k_up` + `k_down`; bytes = the union's routed blocks (values, scales, zero
points), no shared expert. Rows pick 8 distinct uniformly random experts (union 8, 42.6, 90.6 at 1, 6, 16 rows).

### 8.1 Decode, the default (nt1 wk6 p2 dyn / nt2 p1), % of 235 GB/s

Four runs: job B's sweeps in the order i4, b16, b16, i4 and job C's `--ups` runs (default first and last).

| Rows | INT4 (µs) | INT4 % | BF16 (µs) | BF16 % | INT4 73 layers | BF16 2 layers |
|---:|---:|---|---:|---|---:|---:|
| 1 | 201.3–205.8 | **92.3, 92.3; 90.2, 91.8** | 654.7–660.7 | 98.1, 97.8; 97.3, 97.5 | 14.7 ms | 1.3 ms |
| 6 | 1033.7–1044.0 | 95.6, 95.1; 94.6, 95.0 | 3443–3454 | 99.2, 99.2; 99.0, 99.3 | 75.6 ms | 6.9 ms |
| 16 | 2320–2345 | **90.6, 90.1; 89.8, 89.7** | 7503–7523 | 97.0, 96.8; 96.7, 96.9 | 170 ms | 15.0 ms |

- **Targets (≥ 93 %):** BF16 meets them at every row count. INT4 meets them at 6 rows only: 90–92 % at 1 row and
  ~90 % at 16 rows.
- **At 1 row** the route kernel costs 7–8 µs of the step (without it: 96.0 %, 95.9 %); up alone 130 µs, down alone 61
  µs (PDL overlaps them). The best variant in the sweep was down nt2 p2 (94.6 %), within the run's spread.
- **At 16 rows** the default wk6 is the slow one: the prefill-order **wk1 p2 reaches 97.9 %, 97.2 % (sweeps) and
  96.6–96.9 % (A/B both orders)**, and wk4 p2 99.0 % in one sweep. With wk6 a warp owns 2 of the 12 INT4 blocks,
  which its two-ahead prefetch loads at once: no pipeline left to hide the decode's ALU at 16 rows (BF16, with 8
  blocks of 128 k a warp, does not drop). Changing the default K split changes output bits (glm53 decision 11), so
  it stays; the choice is the coordinator's (§ 10).
- A/B of the up variants, both orders (job C, `u-i4`: wk6, wk1, wk1, wk6): INT4 1 row 90.2 / 87.4 / 88.0 / 91.8 %;
  6 rows 94.6 / 95.7 / 95.9 / 95.0 %; 16 rows 89.8 / 96.6 / 96.9 / 89.7 %. BF16 (`u-b16`): 97.3–97.5 % at 1 row, wk6
  99.0–99.3 % against wk1 97.4–97.9 % at 6, wk6 96.7–96.9 % against wk1 95.3–95.4 % at 16.

### 8.2 Prefill (one layer of 168 distinct experts, no fused shared expert)

TFLOPS count T × 8 × 3 × 512 × 6144 MACs × 2 (job C).

| Format, chunk | old (up 0, down 0): ms, TFLOPS | default (up 1, down 1): ms, TFLOPS | default, % of 107 |
|---|---|---|---|
| INT4, 1k | 7.13, 21.7 | 5.69, 27.2 | 25.4 |
| INT4, 4k | 19.06, 32.4 | 15.18, 40.8 | 38.1 |
| INT4, 8k | 37.78, 32.7 | 31.19 / 31.06 (pd run), 39.7 | 37.1 |
| BF16, 1k | 16.41, 9.4 | 14.29, 10.8 | 10.1 |
| BF16, 4k | 31.78, 19.5 | 23.92, 25.9 | 24.2 |
| BF16, 8k | 58.87, 21.0 | 50.81, 24.3 | 22.8 |

- INT4 at 8k matches glm53's NVFP4 (38.0 TFLOPS, 36.5 ms with its shared expert): gate/up 14.5 ms (57 TFLOPS), the
  chained down 15.5 ms (26.6). Parts switched off (`pd-i4`, 8k): gate/up 14.4 ms → 11.1 without weights, 12.1 without
  MMAs, 11.9 without A loads; the chained down 15.4 ms → 3.5 without its ordered adds, 14.5 without waits: as glm53
  § 12, the down is bound by the ordered f32 adds into yr.
- BF16 is bandwidth-bound by weights: each 128-pick tile re-reads its expert's matrices, at 8k ~4 tiles × 168 experts
  × 18.9 MB = 12.7 GB, 54 ms at 235 GB/s — the measured 50.8 ms. Only layers 3 and 77 are BF16 (and the MTP
  layer), ~0.1 s a 8k chunk for the two.
- 73 INT4 layers at 8k-row chunks: 2.28 s a chunk, 0.28 s per 1k rows, before attention and the shared expert.

### 8.3 The BF16 shared expert through attn's GEMV

`bench_experts --shared`: gate/up interleaved [1024][6144] with the SwiGLU epilogue, then down [6144][512] to f32,
18.87 MB a layer, 8 copies cycled; the step's µs (gate/up alone, down alone).

| Rows | splitk 1/1 | best: up 4, down 1 | % of 235 (best) | 75 MoE layers |
|---:|---|---|---:|---:|
| 1 | 116.6 (74.4, 26.7), 68.9 % | 85.7 (58.1, 27.0) | 93.7 | 6.4 ms |
| 6 | 132.7 (76.2, 27.4), 60.5 % | 84.8 (61.5, 28.5) | 94.7 | 6.4 ms |
| 16 | 141.4 (84.5, 26.9), 56.8 % | 85.6 (59.5, 27.5) | 93.9 | 6.4 ms |

The GEMV needs split-K 4 on gate/up (only 1024 rows: 128 tiles on 48 SMs) to reach the floor; down is ~27 µs at
any split. The cost a token is ~6.4 ms over 75 MoE layers (a decode step at 1 row: 85.7 µs shared + 201 µs
routed per INT4 layer). Fusing it into `k_up`/`k_down` as glm53 fused its FP8 one would save the two launches and the
separate down pass, not bytes: done in § 11.

## 9. Runs

Results: `~/dsgb10-data/altar1-experts/results.tsv` and `.log` (not in the repo); binaries built by `install.sh`
fresh from the commit named in `bin/SOURCES`. Every job through `OWNER=altar1-experts …/ring.sh …/experts/run.sh`.

| Job | When (2026-10-14) | Sources, commit | Steps |
|---|---|---|---|
| A | 10:41:10–10:41:18 | `d68c9912203228d2`, b47297b | t-cpu, t-deq, t-route (pass); t-i4, t-b16 (prefill h smoke threshold failed, § 10); t-eq (pass) |
| B | 10:44:46–10:50:40 | `5275133f77797abf`, 31a288e | t-i4, t-b16, t-q4, r-l3r1, r-l4r0, r-l40r2 (pass); r-rnd (2 tolerance failures); bw; d-i4, d-b16, d2-b16, d2-i4 |
| C | 10:49:43–10:52 | `5275133f77797abf`, 31a288e (+ run.sh budget, 9b4abeb) | p-i4, p-b16, sh, u-i4, u-b16, pd-i4, topk |

Dumps (`int4_pack.py dump` from rank 0's read-only copy; shards SHA-256-checked against the manifest first:
00001, 00003, 00004, 00017, 00019, 00020 OK, `~/dsgb10-data/altar1-experts/a0/sha.tsv`): `real/l3r1` (layer 3,
experts 0–2, rank 1), `real/l4r0` (layer 4, 0–3 and 167, rank 0), `real/l40r2` (layer 40, 57–58, rank 2), `real/rnd`
(`dump-random`: random bits over scales 2^-20..2^4).

## 10. Failures and what is left

- **Job A: the prefill h smoke statistic** (bit-exact share against double ≥ 99.5 %, glm53's bar) failed on INT4 and
  BF16: 99.07–99.51 %, with 0 values outside the f32-sum range (the contract check). Prefill sums a strip's 6144 k
  in one f32 chain, decode in six, so more of prefill's sums land one bf16 ulp off the double's rounding on data
  with full mantissas (glm53's NVFP4 values have 2–4 significant bits). The bar is 98.5 % since 31a288e.
- **Job B: r-rnd "prefill vs decode, every row"**: max 4.6e-3 (INT4) and 2.4e-3 (BF16) of the row's largest output
  against the 2e-3 tolerance, on the random-bits dump whose scales span 2^-20..2^4: an h flip there moves an output by
  more of the row's maximum. The exact checks on that dump pass (decode = contract values, INT4 = BF16, prefill =
  decode wk1). Left failing as a record; the real layers pass (≤ 1.6e-3).
- **INT4 decode under 93 % at 1 and 16 rows** with the default wk6. The prefill-order wk1 p2 is 96.6–97.9 % at 16
  rows and 95.7–95.9 % at 6 but 87.4–88.5 % at 1 row; a different split per row count would break batch invariance.
  Options: (a) make wk1 the default (decode = prefill bit for bit, glm53 decision 11 reversed; −2 to −4 % at 1 row,
  +7 % at 16); (b) give wk6 a deeper prefetch or 512-k sub-blocks so INT4's two blocks a warp pipeline; (c) leave it.
  (b) keeps the bits; **done in § 11** as a cross-task prefetch: 96–97 % at 1 row, 99.6–100.3 % at 2–16 rows.
- **torch.topk's order** differs from the engine's (§ 5); the set agrees on CUDA. Whether the reference's sum of the
  eight scores follows the returned order is A1's to settle (glm53's tree came from DeepSeek's).
- **Not done here:** ~~the BF16 shared expert fused into the routed kernels~~ (§ 11); a 256-row prefill tile for BF16 layers
  (halves the weight re-reads); kcheck on a real converted rank file (A2's files); the A0(b) `micro/` GEMV
  prototypes were replaced by benching the engine kernels themselves, which is what ships.

## 11. Tuning: the fused BF16 shared expert and k_up's cross-task prefetch

Written 2026-10-14 (commits 2c4f6f7, c3cb8c6 and its revert c9bc785, d701398, 5b85463). Both changes keep every bit;
the step-level A/Bs, the model checks and the contract placement of `ys` are in [engine.md § 9](engine.md#9-tuning-the-fused-shared-expert-and-the-routed-gateups-prefetch).

**The fused BF16 shared expert.** `SharedW` gains `bgu` (gate/up interleaved, [1024][6144] kBF16T) and `bd`
(down, [6144][512] kBF16T) and `bf16()`; `kShTiles` 128 (8-row tiles of gate/up), `kShParts` 4 (attn's GEMV
split-K on gate/up). kBF16T's lane order is the BF16 expert block's, so the kernels reuse their fragment code.
`k_up<F, NT, WK, P, R16, SHB, XPF>` puts the shared tasks first in its queue (BT = WK / gcd(WK, 4) tiles a task,
BU = 4 / gcd parts a warp, 24 chunks a part, 8 in flight), adds the parts in part order with `__fadd_rn` and applies
the GEMV's SwiGLU epilogue, writing the shared row of h; `k_down<F, NT, P, R16, SHB>` runs one MMA chain over the
shared down strip's 512 k into `ys` (posted with `yr` when the collective is posted). Both prefetch the shared
weights into L2 before `griddepcontrol.wait`. `SHB` false compiles to the earlier kernels; the host chooses it from
`LayerW::sh.bf16()` and dies on an FP8 and a BF16 shared expert together.

**k_up's cross-task prefetch** (`DecodeCfg::xpf`, default true; the template's `XPF_`, only for WK > 1). After a
routed task's partials' barrier each warp loads the next queued task's first P blocks into the registers the next
task would load first, so the epilogue and the next task's first loads overlap. The same blocks, MMA order and
reduction: the bits do not move. It is § 10's option (b).

**Registers** (`-Xptxas -v`, sm_121a, decode.cu at 5b85463, the INT4 wk6 p2 default; no spills unless noted):

| Kernel | plain | + fused shared | + prefetch | both |
|---|---:|---:|---:|---:|
| `k_up<INT4, 1, 6, 2>`, 1–8 rows | 132 | 168 (8 B spill) | 233 | **231** |
| `k_up<INT4, 1, 6, 2>`, 9–16 rows (R16) | 132 | 207 | 248 | 248 |
| `k_up<BF16, 1, 6, 2>`, 1–8 rows | 158 | 168 (8 B spill) | 168 | 202 |
| `k_down<INT4, 2, 1>` | 155 | 218 | | |

The first fused version held the shared weights in registers across the routed tasks: 223 registers on the default
k_up even with the shared expert off; an L2 prefetch replaced it. A `__launch_bounds__` minimum-blocks argument and
an extra `Params` field also changed the unfused code; both were removed so `SHB` false matches the earlier kernel.

**Bench** (`bench_experts --shfused i4|b16`, run.sh step `shf-<fmt>`: the fused h, ys and yr against the GEMVs
and the unfused step for every up and down variant and two shared copies, 26 of 26 identical at 1–16 rows; time
A = route + GEMVs + experts, B = route + fused experts, A B B A; rank 0, job 12:24, sources `857b6ab9e50ff0a2`,
commit 2c4f6f7):

| Rows | INT4 A | INT4 B | BF16 A | BF16 B |
|---:|---|---|---|---|
| 1 | 283.4 / 278.7 µs | 277.6 / 276.7 | 742.3 / 741.7 | 734.5 / 731.7 |
| 2 | 457.6 / 458.0 | 460.6 / 458.0 | 1360.1 / 1354.5 | 1346.1 / 1345.2 |
| 4 | 795.7 / 798.2 | 798.5 / 799.5 | 2518.0 / 2518.6 | 2515.0 / 2513.2 |
| 6 | 1111.6 / 1091.8 | 1114.2 / 1100.4 | 3548.7 / 3529.4 | 3531.9 / 3528.6 |
| 16 | 2279.4 / 2283.8 | 2246.4 / 2246.9 | 7413.4 / 7419.3 | 7416.0 / 7411.4 |

**Failure: 4 chunks in flight at 9–16 rows** (c3cb8c6): 138 registers instead of 207 (two CTAs an SM), slower:
9 rows B 1503.8 / 1501.0 and 1508.9 / 1515.3 µs against 1472.7 / 1471.5 and 1472.1 / 1475.9 with 8 chunks; 16 rows
2279.9 / 2279.4 and 2280.3 / 2282.1 against 2227.6 / 2226.7 and 2230.8 / 2230.7 (two binaries, A B B A, rank 0,
job ~12:58). Reverted in c9bc785.

**The prefetch** (`bench_experts --decode --ups 0,0,0,0 --xpfs 0,1,1,0`: the decode step, route + k_up + k_down,
in one process; rank 0, job ~13:20, sources `e4268f937f5f0169`, commit 5b85463; µs and % of 235 GB/s):

| Rows | INT4 off | INT4 on | BF16 off | BF16 on |
|---:|---|---|---|---|
| 1 | 199.7 / 199.3 (93.0, 93.2) | 192.9 / 191.2 (96.3, 97.1) | 653.8 / 654.4 (98.3, 98.2) | 655.5 / 653.2 (98.0, 98.4) |
| 2 | 382.6 / 385.3 (94.9, 94.2) | 367.4 / 367.7 (98.8, 98.8) | 1269.7 / 1268.5 | 1269.4 / 1269.8 |
| 4 | 733.1 / 731.3 (95.2, 95.4) | 697.4 / 697.6 (100.1, 100.0) | 2444.1 / 2430.1 | 2435.7 / 2432.6 |
| 6 | 1031.0 / 1028.9 (95.8, 96.0) | 987.0 / 985.6 (100.1, 100.3) | 3437.7 / 3439.2 | 3443.7 / 3446.6 |
| 16 | 2325.6 / 2317.1 (90.4, 90.8) | 2102.8 / 2112.0 (100.0, 99.6) | 7506.7 / 7496.8 (96.9, 97.0) | 7285.1 / 7280.2 (99.9, 99.9) |

The trial build with the prefetch unconditional (d701398, sources `5c033306f18dc1ba`) against 2c4f6f7, two
binaries A B B A, agreed (INT4 1 row 200.8 / 200.2 → 191.0 / 193.9 µs; 16 rows 2312 / 2331 → 2115 / 2103).
`test_experts` t-i4, t-b16, t-q4 and t-eq pass on both builds (0 failures: decode against the contract values,
every variant, wk1 = prefill, INT4 = BF16 on the dequantised matrices, batch invariance, determinism).

**Left.** One CTA an SM for the default INT4 k_up; a `__launch_bounds__(192, 2)` build (≤ 170 registers) was not
tried. The fused shared expert at 1 row costs ~76 µs a layer against 80.3 µs of bytes: no further room there.

**`DecodeCfg::ysum`** (commit 6477b29, the runner's default since; engine.md § 9.1, decision 8): k_down adds the
shared expert's f32 output into each value's routed sum after the routed experts (one `__fadd_rn`; the fused path
reads back its own ys write, the GEMV path the caller's ys) and writes or posts yr alone (R rows; the post's size
halves). `bench_experts --shfused` checks it: the fused and the unfused yr with `ysum` equal the plain f32 add of the
GEMV's ys into the unfused yr, 26 of 26 cases at 1, 2, 4, 6, 9, 16 rows, INT4 and BF16 (rank 0, 14:52). Registers:
the default INT4 `k_down<2,1>` with the fused shared expert stays at 218, the BF16 one 208 → 192.
