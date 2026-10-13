# engine/kernels: decode weight-streaming kernels (M3), prefill GEMMs and the prefill attention side (M5)

These are the GEMV and skinny-GEMM kernels that stream a decode or verify step's weights
(M = 1…16 activation rows; from 9 rows the `_r16` builds, [9–16 rows](#916-rows-b4)), and the tensor-core GEMMs for prefill chunks (M = 128…8192 rows). All
of them work on per-rank TP=4 shapes (DESIGN decision 1) and are built for `sm_121a`. The public
API is `dsk.h` (namespace `dsk`). Shared device code lives in `common.cuh`.

| file | kernels |
|---|---|
| `act_quant.cu` | bf16 → e4m3 per 32, power-of-two ue8m0 scale; bit-exact with the reference `act_quant` |
| `gemv_fp8.cu` | FP8×FP8 block-scaled GEMV (`mma.sync … kind::mxf8f6f4.block_scale`), multi-matrix launches, an optional RMSNorm prologue, and a CUDA-core variant |
| `gemv16.cu` | bf16 GEMV with fp32 out (head, router gate; optional RMSNorm prologue), and wo_a (FP8 widened exactly to bf16 in registers, block-diagonal over 2 groups, act_quant fused into the epilogue) |
| `moe.cu` | MXFP4 routed experts plus the FP8 shared expert in two kernels; `moe_routed` also runs the top-6 itself; the routed scales from a 1-bit code ([The experts' scale code](#the-experts-scale-code)) |
| `norm_router.cu` | RMSNorm (+act_quant) and the router (sqrt∘softplus, bias for selection only, top-6, ÷(sum+1e-20), ×1.5) |
| `prefill.cu` | prefill GEMMs: block-scaled FP8 GEMM (SwiGLU + act_quant epilogue for the shared expert), bf16 GEMM for the gate and wo_a, the routed experts as grouped GEMMs; see [Prefill GEMMs](#prefill-gemms-m5-prefillcu) |
| `prefill_attn.cu`, `prefill_index.cu`, `prefill_comp.cu`, `prefill_mhc.cu` | the prefill attention side (`prefill_attn.h`): sparse attention, the indexer, the compressor, mHC; see [Prefill attention side](#prefill-attention-side-m5-prefill_attnh) |

## Build, test, bench

```sh
cmake -S engine -B engine/build && cmake --build engine/build -j
engine/build/kernels/test_kernels                 # all; or: "" act_quant|fp8|wq_b|moe|woa|head|norm|normgemv|router|rows16|rows16moe
engine/build/kernels/test_kernels "" prefill      # prefill GEMMs; or p_fp8|p_16|p_moe
PF_M="2048 4096" engine/build/kernels/bench_kernels prefill   # prefill GEMMs per chunk size
engine/build/kernels/bench_kernels                # all per-kernel benches; filters: fp8|<shape>|pair|moe|woa|gate|head|small
SWEEP=1 engine/build/kernels/bench_kernels wq_b   # also sweep launch shapes
STEP_FUSE="0 2" engine/build/kernels/bench_kernels step   # decode-step graph, fusion variants interleaved
engine/build/kernels/test_pattn "" [attn|index|comp|mhc]  # prefill attention side, cases in ~/dsgb10-data/attntests
OWNER=me engine/server/ring.sh engine/build/kernels/bench_pattn --ref 16384 65536 131072   # [--only attn|index|comp|mhc|dec]
```

Benchmarks rotate weight copies over more than 256 MB so nothing is reused from cache. They time
CUDA-graph replays and report the median. The GPU is shared, so check `nvidia-smi` first. The
`step` bench interleaves its variants round by round.

`prefill` environment variables:
- `PF_M`: chunk sizes (default `2048 4096`).
- `PF_ROUNDS`: interleaved rounds (default 11).
- `PF_SWEEP=1`: also sweep the dense GEMMs' tile variants.
- `PF_L2`: the MoE's L2 hint bits (`MoePArgs::l2`) for all but the hint variants.

`step` environment variables:
- `STEP_ROUNDS`: rounds per variant.
- `STEP_CASES=1|6`: which cases to run.
- `STEP_SKIP` bits: 1 norms, 2 gate+top-k, 4 MoE, 8 attention GEMVs.
- `STEP_FUSE` bits:
  - 1: attn_norm into wq_a+wkv.
  - 2: ffn_norm into the gate.
  - 4: q_norm into wq_b.
  - 8: top-6 into the MoE.

Reference data comes from real layer-3 weights and the torch stand-ins in `ref/torchref/kernel.py`.
It lands in `~/dsgb10-data/kerneltests`. To regenerate it, run from the configuration directory
(`ds41f-4xgx10-switchless/`):

```sh
docker run --rm --gpus all --ipc=host --user 1000:1000 -e HOME=/tmp --entrypoint /opt/sglang/bin/python3 \
  -v /data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8:/models:ro -v $PWD:/work -v $HOME/dsgb10-data/kerneltests:/out \
  dsv41-4x-spark:canary-roce /work/engine/kernels/tests/gen_ref.py
```

The prefill data (`p_*`, 256-row chunks) comes from `tests/gen_ref_prefill.py`, the same command
with that script. It reads back the weights `gen_ref.py` saved, so run it second.

`tests/probe_router_sum.py` found the order in which torch sums the 6 routing weights,
`((w0+w4)+w2)+((w1+w5)+w3)`. That order makes our weights bit-exact.

## Weight layouts the per-rank files should hold

Scales always stay in their natural row-major layout, except for MXFP4, which has its own layout
(L3). Tile sizes are fixed by the kernels.

- **L1, FP8 dense (`tile_fp8`).** Used for wq_a, wkv, wq_b, indexer wq_b, wo_a, wo_b, and the
  shared expert's w1, w3 and w2.
  - Source: e4m3 `[N][K]`, with N % 8 = 0 and K % 64 = 0.
  - Element (n, k) moves to byte `((n/8)*(K/64) + k/64)*512 + (n%8)*64 + k%64`.
  - Scales: ue8m0 `[N/32][K/32]`, unchanged.
- **L2, bf16 (head, router gate).** View bf16 `[N][K]` as bytes `[N][2K]` and apply L1 with K' = 2K.
- **L3, MXFP4 routed experts (`tile_fp4`).** Applies to w1 and w3 `[576][5120]` and to w2
  `[5120][576]`.
  - Source: packed e2m1 `[N][K/2]` (the low nibble holds the even k) and ue8m0 `[N][K/32]`.
  - Let P = K/64. Write n = 16·n16 + 8·h + g, k = 64·p + 32·l + 8·t + i, and j = 2h + l.
  - The 4 source bytes `[n][32p + 16l + 4t, +4)` go to byte offset `((n16*P + p)*32 + 4g + t)*16 + 4j`.
  - Scale `[n][2p + l]` goes to byte `((n16*P + p)*8 + g)*4 + j`.
  - Experts are stacked by expert id: expert e's w1 starts at `e*576*2560` bytes and its scales at
    `e*576*160`; w3 is laid out the same way. w2 starts at `e*5120*288` bytes and its scales at
    `e*5120*18`. w1 and w3 are separate arrays, not interleaved.
- **Per-rank shapes.** These are rank 0's slices as the tests take them from the checkpoint.

  | tensor | per-rank shape | slice of the full tensor |
  |---|---|---|
  | wq_a | 320 × 5120 | rows |
  | wkv | 128 × 5120 | rows |
  | wq_b | 8192 × 1280 | rows |
  | wo_a | 2048 × 4096 | rows, i.e. groups 2r and 2r+1 |
  | wo_b | 5120 × 2048 | columns |
  | gate | 96 × 5120 bf16 | rows; bias fp32 `[384]` in full |
  | head | 32320 × 5120 bf16 | rows |
  | expert and shared w1, w3 | 576 × 5120 | rows |
  | expert and shared w2 | 5120 × 576 | columns |

  The indexer's wq_b is 4096 × 1280 in full; `step` assumes 1024 rows per rank. wq_a and wkv may
  stay separate tensors, because one launch takes both as segments.

  wo_a widens FP8 to bf16 exactly only while every scale byte is ≤ 134 (scale ≤ 2^7). The
  checkpoint's largest is about 116. The converter should assert this.

## Results

Clean GPU, 2026-10-11. GB/s counts weight + scale bytes, against a 235 GB/s peak.

| kernel | M=1 | M=6 | vs reference |
|---|---|---|---|
| act_quant | – | – | bit-exact |
| wq_a+wkv (one launch, cfg 1,1,8) | 95 % (10.3 µs) | 95 % | 100 % exact |
| wq_b / wo_b / ix wq_b | 108 / 107 / 107 % | 107 / 108 / 104 % | 99.98–100 % exact, 0 beyond 1 ulp |
| wo_a | 106 % (33.6 µs) | 105 % | 99.95–99.98 % exact; torch's einsum differs from fp32 math more |
| gate 384 × 5120 bf16 | 97 % | 94 % | max rel 4.8e-7 (fp32 sum order) |
| head | 102 % (1376 µs) | 102 % | max rel 1.2e-6; closer to fp64 than torch is |
| MoE, 6 experts + shared, M=1 | 99 % (160 µs) | 95 % (165 µs) | 100 % exact (bf16 sums), M=8 max 1 ulp |
| MoE at M=6 with 12 / 24 / 36 experts | – | 98 / 97 / 95 % (284 / 532 / 798 µs) | moe_up's re-quantised h: 0 off vs fp64 |
| rmsnorm, router top-6 | 1.6 µs, 7.2 µs | 1.9 µs, 7.3 µs | bf16 100 %; ids exact, weights exact from reference logits |

Each extra distinct expert costs 21 µs: 4.70 MB per rank, with the up-projection and its scales
read once for the union of all rows' picks. CUDA cores against tensor cores at M=1/M=6:
- FP8 GEMVs: 72–98 % / 49–93 % on CUDA cores (vs ≥ 95 % with MMA).
- moe_up: 98 % / 88–93 % on CUDA cores.
- wo_a: 105 % / 84 % on CUDA cores.

Tensor cores are never slower, and they win as M grows.

**Decode step** (`step`): one rank's weight reads per token for 40 layers plus the head, as one CUDA
graph. Attention math, mHC, Engram and collectives are not included.

| case | bytes | time | % of 235 GB/s |
|---|---|---|---|
| k=1, 6 experts/layer | 3130 MB | 13.85–13.90 ms | 96 % |
| k=6, 27 experts/layer | 7079 MB | 31.2 ms | 97 % |
| k=6, 36 experts/layer | 8771 MB | 38.4–38.6 ms | 97 % |

**Launch structure per layer:**
- 9 kernels (10 on index layers), all PDL launches, each prefetching its first weights before
  `griddepcontrol.wait`:
  - rmsnorm(+quant)
  - wq_a+wkv
  - rmsnorm(+quant)
  - wq_b
  - [indexer wq_b]
  - wo_a(+quant)
  - wo_b
  - gate with ffn_norm fused (CTA 0 writes the MoE's act_quant)
  - router top-6
  - moe_up
  - moe_down
- Fused into neighbours instead of launched:
  - act_quant.
  - SwiGLU, clamp, routing weight and re-quantisation (in moe_up).
  - The expert sum and the shared expert (in moe_down).
- Measured and not used by default:
  - attn_norm in the wq_a+wkv prologue: +0.05 to +0.65 ms per step. Every CTA redoes the norm.
  - q_norm in the wq_b prologue: neutral.
  - top-6 inside the MoE: +0.1 ms. Every CTA of both kernels redoes the selection. It removes 40
    launches if launches matter more.

## 9–16 rows (B4)

A batched decode step ([batching.md](../../docs/batching.md), B4) has up to 16 rows (`dsk::kMaxRows`).
Up to 8 (`kLowRows`) every kernel is the one it was: the same names and the same SASS
(`k_gemv_fp8`, `k_gemv16`, `k_moe_up`, `k_moe_down`, norm_router; the persistent `k_gemv16p<S, false>`
differs from before only in a register renaming). From 9 rows the calls take `_r16` kernels that put
rows 8..15 in the upper half of the m16 MMA tiles: the same weight loads and MMAs, twice the
activation loads, and each row's bits the same as in an 8-row call (the MMA's rows are independent,
and the k order and folds do not change).

| kernel | registers | shared memory | notes |
|---|---|---|---|
| `k_gemv_fp8_r16<TILED, NORM>` | 128 / 110 (NORM) | dynamic: 4–8 KB, NORM at K 5120 ~87–91 KB | one tile per warp (mt 2 and 4: the same bits, they would spill) |
| `k_gemv16_r16<bf16, mt 1/2, NORM>` | 80, 116; NORM 108, 128 | dynamic: the fold only | NORM normalises x as it reads it (1/rms per row first): 16 normed rows of 5120 would take 165 KB |
| `k_gemv16_r16<wo_a, mt 1/2>` | 86, 120 | as above | |
| `k_moe_up_r16<NT, WK, KU>` | 128–254 | 24 KB (WK 4), 40 KB (WK 8) | union of up to 96 experts (`Union<16>`), two rows per warp in the epilogue when there are 8 warps |
| `k_moe_down_r16<W, R>` | 128–238 | 12–24 KB | rounds of W units, outputs folded per round in unit order: k_moe_down's order and bits without its [97][16][16R] buffer |

No 16-row build spills. `MoeCfg::up16`/`down16` pick the MoE variants from 9 rows: up16 0..4 =
`<1,4,2>`, `<2,4,4>`, `<4,8,2>`, `<2,8,1>`, `<2,4,2>`; down16 0..4 = `<16,2>`, `<8,2>`, `<8,4>`, `<8,1>`,
`<4,1>` (the builds that keep to their registers). The defaults, up16 4 and down16 2, are the fastest pair
at 9, 12 and 16 rows: bench_chain's step 60.86 → 59.96 ms at 9 rows (51 experts), 75.67 → 74.25 at 12
(66) and 96.59 → 93.88 at 16 (86) against up16 0 with down16 0; alone, down16 2 is the best down kernel
at all three and up16 4 the best up kernel (up16 2, `<4,8,2>`, is 0.6 ms behind it with down16 2). Up to 8
rows only: the CUDA-core variants, gemv16's mt 4 and the persistent GEMVs. The MoE workspace grows
to 921,896 B (`moe_workspace_bytes()`; the layout per row budget: `moe_hs_offset(M)`).

Tests (`test_kernels "" rows16`): 16-row inputs are the 8 reference rows and then the same rows
reversed; at 9, 12 and 16 rows every output row must equal that input row's output from an 8-row
call bit for bit, and the fused paths (norm + GEMV, wo_a + act_quant, the router inside the MoE)
their unfused sequences. The MoE also runs a pool of 128 random experts with unions of up to 96,
every up16 × down16 pair against two 8-row calls, its intermediate (hq/hs) slot by slot, and the
task queue against the static schedule. `bench_chain --rows 9..16` runs the chain at those rows
(experts per layer: 51 at 9 rows, 66 at 12, 86 at 16, as many as independent rows touch); `mu<N>`
and `md<N>` set the MoE variants. The head re-reads x in every CTA (2,020 of them) but costs at 16 rows
what it costs on one row: `h1` (the head on one row) 96.39 against 96.38 ms, `hsep` (x normalised once,
then the plain GEMV) 96.37, `hw2` (32 rows a CTA) 96.42; at 12 rows the same within 0.03 ms.

## The experts' scale code

Since master 2812072, the routed experts' ue8m0 scales are also held as a 1-bit code over each tensor's
pair {b, b + 1}. `dsk::scale_code` makes it at bind: 579.4 MiB a rank, with 98.11 % of the 32-scale
groups coded. The groups off the pair are flagged and read from the raw scales, which stay.
- **The readers.** These builds read the code when `MoeWeights` carries `c1`, `c3` and `c2`:
  - `k_moe_up` <1, 4, 4>, and `k_moe_down` <4, 1> and <16, 4>, up to 8 rows;
  - `k_moe_up_r16` <2, 4, 2> and `k_moe_down_r16` <8, 4> at 9–16 rows (the defaults);
  - `k_moe_up_p` and `k_moe_down_p` in prefill.

  The other variants and `k_moe_down_g` (a prefill tuning, off) read the raw scales, and dspark's
  experts carry no code.
- **Bits.** The same scale bytes go into the same MMAs in the same order. test_kernels `mcode` (in
  `moe`) checks coded against raw bit for bit, at 1–16 rows and in `moe_prefill`.
- **Time.** bench_kernels `mscale` times raw against coded on real scales (layers 20 and 39) and on a
  synthetic layer with half the groups raw.
- **The measurements and the engine A/B:**
  [engine § The experts' scales coded](../../docs/engine.md#the-experts-scales-coded-and-the-exact-compression-bundles-ab-2026-10-12).

## Prefill GEMMs (M5, `prefill.cu`)

Prefill uses the same weight layouts as decode (L1, L2, L3) and needs nothing else.

| function | what |
|---|---|
| `gemm_fp8_p` | FP8×FP8 block-scaled GEMM, bf16 out; weight segments that share one input go in one launch (wq_a+wkv) |
| `gemm_fp8_swiglu_p` | shared expert w1/w3 with SwiGLU, clamps, bf16 rounding and act_quant fused (w2's input) |
| `gemm_bf16_p` | bf16 (L2 layout) → fp32: the router gate |
| `gemm_woa_p` | wo_a, block-diagonal: FP8 widened exactly to bf16 in registers, bf16 MMA, bf16 out and/or act_quant |
| `moe_prefill` | routed experts (top-6) plus the shared expert: bucketing, grouped up/down GEMMs over 64-row tiles, combine |
| `moe_combine_rs` | for chunks `moe_prefill_fuses` says no to: a column group's combine inside its reduce-scatter by rows over the ring |
| `sum4_bf16_p` | four bf16 partials added in rank order in fp32, rounded once (wo_b's, after the model's all-to-all) |

**Design.**
- Dense FP8:
  - CTA tile 128×128 (4 warps of 64×64), k-step 64, 3-stage cp.async pipeline.
  - Fragments come by ldmatrix from XOR-swizzled 64-byte rows into `mma…kind::mxf8f6f4.block_scale`
    m16n8k32 with the ue8m0 scales of each 32 k.
  - N-tiles run along grid x, so CTAs that share an activation block run together. The gate at
    M = 4096 went from 417 to 213 µs.
  - Outputs go through shared memory as coalesced bf16 rows. The SwiGLU epilogue pairs w1 with w3
    columns and quantises 8 values per lane.
- bf16 (gate, wo_a): m16n8k16 MMA.
  - wo_a widens FP8 to bf16 per fragment, exact for scales ≤ 2^7.
  - wo_a's tiles are 128×256, 128×128 or 64×64, whichever fills the 48 SMs (M = 128: 167 → 54 µs).
  - For fp32 outputs (the gate) every pipeline stage keeps its own partial sums, added at the end.
    One fp32 chain over K = 5120 drifted 7e-6 relative; now it is 3.3e-7 (torch: 1.9e-7).
- MoE:
  - Bucketing: one CTA counts picks per expert, scans the offsets and lists the 64-row tiles,
    experts by descending count. A scatter kernel places the (token, weight) pairs.
  - Up: CTA = (64 intermediate columns, one tile). w1 and w3 stream from L3 straight into
    registers. The tile's gathered activation rows pass through shared memory. SwiGLU, clamps,
    ×routing weight, bf16 and act_quant run in the epilogue. Empty m16 sub-tiles are skipped.
  - Down: CTA = (128 output columns, one tile). The h rows sit in shared memory, w2 streams into
    registers, and each pick's output is rounded to bf16.
  - Combine: per row, the 6 outputs in ascending expert id in fp32, shared expert last, as `moe()`
    and the reference do.
    - Up to M ≈ 2.5k it runs inside the down kernel. The down kernel runs column-block-major, the
      CTA that completes a (128-column block, 64-row group) sums it, and `discard.global.L2` drops
      the dead lines.
    - Larger chunks cannot, because the h rows re-read per column block no longer fit in L2. In
      the model the combine then runs inside each column group's reduce-scatter
      (`moe_combine_rs`): the kernel that fills the ring's send buffers computes a row's four
      columns from its six bf16 outputs (positions sorted by expert id once per chunk,
      `k_moe_sortpos`) and the shared expert's, so the fp32 sums never reach DRAM. The four ranks'
      sums are added in rank order and rounded to bf16, the bits of the separate combine, the fp32
      reduce-scatter and the rounding. At 8192 rows that saves ~0.4 GB of DRAM traffic per layer.
      The separate kernel (`k_moe_combine`) remains for callers without a ring.
  - L2 priorities (`createpolicy`): weights are evict_first unless their expert has more than one
    tile, in which case the next tile re-reads them from L2. Re-read activations are evict_last.
    At M = 8192 this takes the MoE from 21.5 to 19.0 ms.
  - The shared expert can run on a side stream (`MoePCfg::side`). It measured slower, because the
    routed experts already fill the GPU, so it stays an option.

**Tests** (`prefill`) compare 256-, 200-, 129- and 1-row chunks with torch:
- Dense FP8: 99.9–100 % of outputs bit-exact, none beyond 1 ulp, for every tile variant.
- Shared SwiGLU h and its scales: 100 % exact.
- wo_a: 99.92 % exact; its fused act_quant equals act_quant of its own output.
- Gate: within 6.5e-7 of the row max of torch's; against fp64, 3.3e-7 (torch: 1.9e-7).
- MoE: 99.8 % exact, none beyond 1 ulp. It is deterministic, and fused or separate combine and the
  L2 hints give the same bits. It also matches decode `moe()` run in 8-row steps.

**Results.** rank 0, 2026-10-11, per rank, weights cold, median of 11 interleaved rounds. The
measured mma.sync peaks are 214 TFLOPS e4m3 and 107 TFLOPS bf16; L2 → SM reads run at 1.87 TB/s.
Entries are TFLOPS (time).

| kernel (per-rank shape) | M = 2048 | M = 4096 | M = 8192 |
|---|---|---|---|
| wq_a+wkv (448 × 5120) | 75 (126 µs) | 91 (206 µs) | 95 (396 µs) |
| wq_b (8192 × 1280) | 129 (333 µs) | 145 (592 µs) | 154 (1113 µs) |
| indexer wq_b (1024 × 1280) | 124 (43 µs) | 134 (80 µs) | 131 (164 µs) |
| wo_b (5120 × 2048) | 138 (312 µs) | 152 (564 µs) | 160 (1074 µs) |
| shared w1+w3, SwiGLU (2 × 576 × 5120) | 139 (173 µs) | 145 (332 µs) | 148 (654 µs) |
| shared w2 (5120 × 576) | 92 (132 µs) | 95 (255 µs) | 94 (516 µs) |
| wo_a, bf16 (2 × 1024 × 4096) | 68 (503 µs) | 67 (1022 µs) | 75 (1831 µs) |
| gate, bf16 (96 × 5120) | 30 (67 µs) | 19 (215 µs) | 19 (425 µs) |
| MoE, routed + shared | 27.5 (9.23 ms) | 39.6 (12.82 ms) | 53.4 (18.99 ms) |
| – routed up | 27 (5.45 ms) | 50 (5.84 ms) | 79 (7.30 ms) |
| – routed down (+ fused combine at 2048) | 21 (3.38 ms) | 33 (4.34 ms) | 44 (6.65 ms) |
| – separate combine | – | 1.73 ms | 3.43 ms |
| – bucketing | 27 µs | 43 µs | 74 µs |

- The gate reads its bf16 input at DRAM speed (42 MB at M = 4096). Fusing it with ffn_norm would
  save that read.
- The routed experts stream 1.8 GB per layer per rank whatever M is: 7.7 ms at 235 GB/s. That
  floor sets the throughput of small chunks. At M = 2048 the up phase streams w1/w3 at 94 % of
  DRAM speed.
- At M ≥ 4096 the next largest cost is the bf16 outputs' round trip through DRAM (P × 10 KB each
  way).

**A chunk through one layer's GEMMs:** wq_a+wkv, wq_b, indexer wq_b, wo_a, wo_b, gate, and the
MoE with the shared expert (uniform routing). The ×20 column scales this to the 20 encoder layers
that CED prefill runs over every token. GEMMs only: attention, norms and collectives are not
included.

| chunk M | one layer | ×20 layers | tokens/s | TFLOPS per rank |
|---|---|---|---|---|
| 128 | 7.21 ms | 144 ms | 888 | 3 |
| 512 | 8.65 ms | 173 ms | 2 959 | 11 |
| 1024 | 9.16 ms | 183 ms | 5 587 | 21 |
| 2048 | 10.56 ms | 211 ms | 9 698 | 37 |
| 4096 | 15.22 ms | 304 ms | 13 458 | 51 |
| 8192 | 24.15 ms | 483 ms | 16 959 | 65 |

These GEMMs cost 191 MFLOP per token per layer per rank. Over 20 layers and 4 ranks that is
15.3 GFLOP per token, against the ~17 GFLOP budget. The baseline prefills 4.2k tokens/s end to end.

**Not done yet:**
- Split-K for small chunks: wq_a+wkv and the gate take 41 and 46 µs at M = 128.
- MoE tiles of 128 rows with weights staged in shared memory, for M ≥ 4096.
- A down kernel that adds into fp32 rows held in L2, in expert order. It would remove the output
  round trip at M ≥ 4096.
- Fusing the combine into the residual add.

## Prefill attention side (M5, `prefill_attn.h`)

This is everything in a layer besides the GEMMs and the collectives: sparse attention, the
indexer, the compressor and mHC. Shapes are per rank (16 heads), for CED's 4096-row encoder chunks
and its 128-row decoder replay (docs/ced.md). State, formats and numerics are engine/attn's. These
are its decode-shaped kernels redone for many rows, and the tests hold them against attn's.

| function | file | what |
|---|---|---|
| `sparse_attn_p` | `prefill_attn.cu` | window (FP8) + top-k (FP4) attention, MQA over the 512-wide latent; `win_lo` truncates the window for CED's replay |
| `index_score_p`, `index_topk_p` | `prefill_index.cu` | indexer scores (FP4 × FP4, mxf4nvf4 MMA), top-512 and layer 20's pool, in batches of rows whose scores stay in L2 |
| `compressor_project_p`, `compressor_emit_p` | `prefill_comp.cu` | wkv and wgate as one GEMM (`gemm_bf16_rows_p`), then pool, norm, index key and FP4 entries, 16 per CTA |
| `hc_fn_parts`, `hc_boundary_p` | `prefill_mhc.cu` | one mHC sublayer boundary (hc_post, projection + Sinkhorn, hc_pre, norm, act_quant) in one pass over the stream |

**Design.**
- Attention:
  - A row is 16 heads (MMA M = 16) against its own 640 keys: 2 window blocks of 64, then kmax/64
    blocks of selected entries.
  - Keys stay packed in shared memory. Each warp widens only its own operands to f16 in registers
    (e4m3x2 / e2m1x2 → f16x2, times the scale; exact).
  - QK splits a block over 4 key groups × 2 dim halves. In PV each warp owns 64 output dims.
  - Online softmax as the reference: P rounded to bf16, the sink in the denominator. The running
    max is lazy: the CTA agrees on a new max only when a block exceeds the current one by e^8.
  - One CTA per SM walks its rows. The next block's keys and the next row's q and top-k are
    prefetched.
- Indexer:
  - A CTA holds 256 positions of keys in registers as the MMA's A operand and streams q through
    shared memory for all of its rows, so keys are read once per CTA.
  - The epilogue is attn's, in its order, so the scores are bit-identical.
  - `index_topk_p` scores a batch of rows (128–256), then selects torch.topk-exactly while the
    batch's scores are still in L2.
  - Rank split: the rows. Rank k takes rows ≡ k (mod 4), which balances the triangle, and the
    top-k lists are all-gathered (2 KB per row).
- Compressor:
  - The projection is one GEMM over both weights (`gemm_bf16_rows_p`, row-major bf16 as attn keeps
    them).
  - Emit runs 16 entries per CTA. Pool and RMSNorm follow attn's order, wk runs on tensor cores,
    then k_norm, RoPE and both FP4 formats.
  - Ratio 2 needs both rows of a pair on one rank. Chunks that start at an even position and split
    into even quarters keep the carry on one rank.
- mHC (`hc_boundary_p`):
  - Single-pass mHC: y, x and the next coefficients all come from one read of sub and of the 4-copy
    stream (50 KB in and 50 KB out per row, then a second pass over x).
  - A CTA takes 32 rows and streams blocks of 64 columns of each copy through a 3-stage cp.async
    pipeline.
  - hc_post and hc_pre are elementwise in attn's order. The x² sums are kept per column class, so
    the norm reproduces attn's thread and warp tree exactly.
  - The hc_fn projection splits the fp32 weights into three bf16 parts (exact) for bf16 MMA, with
    fresh fp32 partials per block.
    - Each warp prefetches its parts for the next block into registers.
    - Every CTA reads all 2.9 MB of parts from L2. With 16-row CTAs that cost 20 % of the
      boundary; at 32 rows it is ~10 %.
  - The Sinkhorn runs on 4 lanes per row in attn's order. Each warp then re-reads its own x rows,
    normalises them, and optionally quantises them (act_quant).
  - `y` may be null: under CED, layer 20's projection needs only x for the rows before the replay.

**Tests** (`test_pattn`, against attn's kernels and model.py's cases):
- Attention:
  - Against fp64, the per-head relative L2 error is 2.7–3.6e-3, the same as attn's kernel
    (2.5–3.6e-3).
  - Against model.py's layers 0, 2, 3, 20, 21, 24 and 25, 90–95 % of the bf16 outputs are
    bit-exact; the mean relative difference is 1.6–2.6e-4.
  - A CED replay case covers the window truncated at the chunk start.
- Indexer: scores are bit-identical to attn::index_score (67 M keys at 128k). The top-512 and
  layer 20's pool equal attn's and model.py's for every row.
- Compressor:
  - KV entries are bit-identical to attn's.
  - Index keys are equal in every tested entry. wk's summation order could move one element by one
    FP4 step.
  - The projection is within 1e-5 relative of model.py's, and within one bf16 step at ratio 1.
- mHC: y, x and act_quant are bit-identical to attn's and model.py's. pre and post are within 2e-6,
  and comb within 5e-6, relative to attn's.

**Results.** rank 0, 2026-10-11, per rank, median of 9 graph replays after an L2 flush:

| part | rows | ours | attn's |
|---|---|---|---|
| attention, window only | 4096 | 0.79 ms (21.6 TFLOPS) | 1.43 ms |
| attention, window + 512 | 4096 | 2.50 ms (34.4 TFLOPS) | 11.8 ms |
| indexer scores, 128k context | 1024 | 2.32 ms (233 TFLOPS) | 3.48 ms |
| indexer scores + top-512 at 16k / 64k / 128k | 1024 | 0.51 / 1.48 / 2.92 ms | – |
| compressor ratio 2: projection + emit | 1024 | 0.30 + 0.09 ms | 4.45 + 0.08 ms |
| compressor ratio 1 | 1024 | 0.16 + 0.06 ms | 2.26 + 0.10 ms |
| mHC boundary with act_quant | 1024 | 0.65 ms (169 GB/s) | 1.21 ms |
| mHC boundary with act_quant | 4096 | 2.52 ms (175 GB/s) | 5.02 ms |

**(a) CED encoder, 4096-row chunks.** Per rank, ms per layer. Attention runs on all rows; the
indexer, the compressor and mHC run on the rank's 1024 rows. GEMMs are 15.22 ms per layer (above).
Collectives are not included.

| context | window 0–1 | Full 2, 8, 14 | the other 15 | layer 20 | 20 layers | + GEMMs | tokens/s |
|---|---|---|---|---|---|---|---|
| 16k | 2.27 | 5.16 | 3.98 | 0.64 | 80.4 | 385 | 10 645 |
| 64k | 2.27 | 6.12 | 3.97 | 0.64 | 83.1 | 388 | 10 571 |
| 128k | 2.27 | 7.56 | 3.98 | 0.64 | 87.5 | 392 | 10 451 |

- Every layer has its attention (0.79 ms window only, else 2.50 ms), 2 mHC boundaries (1.31 ms)
  and q_norm + kv_write (0.17 ms).
- The Full layers add the indexer (0.51–2.92 ms), the compressor (0.38 ms) and weights_proj +
  index_q_prep (0.29 ms).
- Layer 20 is CED's projection of every row: one boundary without y (0.41 ms) and the ratio-1
  compressor (0.22 ms).
- Attention is the largest part: 47 of the 80–88 ms. mHC takes 26 ms.

**(b) CED decoder replay: 128 rows through layers 20–39.** Per rank, ms. Attention runs on all
128 rows. The indexer and mHC run on 32 rows (rows split) or on all 128 (not split).

| P | split | layer 20 | Reindex 24, 28, 32, 36 | the other 15 | 20 layers |
|---|---|---|---|---|---|
| 16k | rows | 0.44 | 0.44 | 0.34 | 7.3 |
| 64k | rows | 0.56 | 0.49 | 0.37 | 8.1 |
| 128k | rows | 0.67 | 0.52 | 0.39 | 8.5 |
| 16k / 64k / 128k | none | 0.73 / 1.05 / 1.52 | 0.74 / 0.78 / 0.82 | 0.53 / 0.56 / 0.57 | 11.6 / 12.5 / 13.4 |

- The GEMMs of a 128-row chunk take 7.21 ms per layer (above), 144 ms for the replay. The
  attention side adds 5–6 %.
- At 128 rows the kernels are short of work:
  - Attention takes 0.12–0.17 ms: each CTA does its 2–3 rows in turn. attn's split kernel takes
    0.51 ms.
  - hc_boundary_p takes 0.23 ms on 1–4 CTAs. The table uses attn's chain instead (0.10 ms at 32
    rows, 0.19 ms at 128).
  - Layer 20's indexer over all P keys takes 0.07–0.25 ms on 32 rows (0.17–0.91 ms on 128).
    Reindex (attn's kernels on the 16 384-position pool) takes 0.06–0.11 ms.

**Not done yet:**
- mHC for few rows: split the columns across CTAs, then finish (Sinkhorn, norm, act_quant) in a
  second small kernel. At ~25 µs per boundary on 32 rows, (b) would drop from 8.5 to ~5.6 ms.
- Attention for few rows: split each row's blocks over 2–3 CTAs and merge them.
- Attention sharing keys between neighbouring rows. Their windows overlap in 127 of 128 positions,
  and 4 rows × 16 heads would make M = 64 per key block. That is also the next step for (a), where
  attention runs at 34 TFLOPS.
- weights_proj as a GEMM (attn's gemv_bf16 takes 0.20 ms at 1024 rows).
- The rank split's collectives are not measured here: reduce-scatter of the sublayer output, then
  all-gather of FP8 x and of the top-k lists. They replace the two all-reduces per layer and move
  less data.
