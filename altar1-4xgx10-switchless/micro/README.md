# glm53 micro benchmarks (experiment 1)

Decode-GEMV and tensor-core microbenchmarks that decide the routed experts' storage format
([DESIGN.md](../DESIGN.md) decision 1, experiment 1). Results and verdict:
[docs/findings.md](../docs/findings.md).

## Build

```
./build.sh [outdir]          # default ~/dsgb10-data/glm53-micro/bin
```

One nvcc line per binary, CUDA 13.0, `-O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a -lineinfo`
(block-scaled MMA needs the arch-specific target), `gemv` with OpenMP for its CPU references.
`compile_probe.cu` is compile-only: it shows that ptxas rejects `kind::mxf8f6f4` with `ue4m3` scales.

## Run

```
gemv --cpu-check                       # host only: every decoder and lane mapping against its reference
mma --analyse results/2026-10-13.mma4x.bin   # host only: re-classify the dumped NVFP4 MMA trials
# GPU: only through ds41f's ring.sh (short, preemptible); from the repo root:
OWNER=glm53-micro ds41f-4xgx10-switchless/engine/server/ring.sh glm53-4xgx10-switchless/micro/run.sh
```

`run.sh [results.tsv]` runs every benchmark not yet marked `DONE` in the results file (default
`~/dsgb10-data/glm53-micro/results.tsv`), so a preempted run is resumed by running it again. Each
benchmark has a 240 s timeout; the whole set took 30 s on 2026-10-13. Peak device memory ~2.5 GB.

## Benchmarks

| Binary / mode | What |
|---|---|
| `gemv --bench q4` | NVFP4 as published (e2m1 two a byte, low nibble even, E4M3 scale per 16, F32 per tensor), BF16 activations, widened to bf16 in registers, m16n8k16 bf16 MMA, f32 accumulate |
| `gemv --bench s3` | E2M1-subset 3-bit code {±0.5, ±2, ±4, ±6} (quantisation.md § 3.3), E4M3 scale per 16, 3.5 bpw: 32 values in 12 B (two 2-bit planes + one 1-bit plane), two lop3 + `prmt` table decode |
| `gemv --bench tk3`, `tk4`, `tk3m` | EXL3 trellis K3 / K4 with the 3INST codebook, and K3 with MUL1 (exllamav3's `mul1`, QTIP's 1MAD), the full EXL3 linear: input Hadamard of x ⊙ suh, decode + f16 MMA (f32 accumulate), output Hadamard ⊙ svh |
| `gemv --bench dense` | BF16 and FP8 (e4m3, f32 scale per 128×128) GEMVs on q_b [4096 × 2048] and o_proj [6144 × 4096] a rank |
| `gemv --bench bw` | the window's streaming-read bandwidth (also measured at the start of every `gemv` bench) |
| `mma` | `mxf4nvf4` 4X/ue4m3 (NVFP4) bit-exactness, operand/scale layout, rate; `mxf4nvf4` 2X/ue8m0; `mxf8f6f4` e4m3×e4m3 and e4m3×e2m1; bf16 and f16 `mma.sync` rates |

MoE shapes are GLM-5.3's per rank at TP=4: gate and up [512 × 6144] each (one fused launch), down
[6144 × 512]; the union of a step's experts is read once: 8 experts at 1 row, 28 at 6 rows. A pool of
2.2 GB of experts is cycled so that L2 (24 MB) never holds a step's weights. Each step is up (+ the
trellis input transform), a small "mid" kernel (SwiGLU, and for trellis the output and down-input
transforms), down; timed as CUDA graphs of whole pool cycles, the median of 7 replays, per kernel and
per step. Every kernel is checked against a CPU reference on the data it timed: the decode bit for
bit (through the kernels' own fragment mapping), the Hadamard transforms bit for bit, the GEMVs
against double precision (|err| / Σ|terms| ≤ 1e-4).

## Files

- `common.cuh` — the decoders (host and device), MMA wrappers, the 128-point Hadamard.
- `gemv.cu` — the GEMV kernels, CPU references and timing harness; `mma.cu` — the tensor-core tests.
- `results/2026-10-13.tsv` — the GPU window's raw output; `results/2026-10-13.mma4x.bin` — the 64
  raw NVFP4 MMA trials (A, B, scale and C registers per lane, and D).

## Layouts (engine-internal; each is a lossless re-tiling of the canonical tensor)

- NVFP4 `[N/8][K/128][32 lanes][16 B]` + scales `[N/8][K/128][32][2 B]`: lane (g, t) holds row g,
  k = 128j + 64q + 16t + i in byte 8q + i/2. A lane's 16 values are one scale group and the B fragments
  of four m16n8k16 (ds41f's permuted-k trick: activations use the same k order).
- S3: the same tiling, three u32 planes per lane per 128 k (P0, P1, Q, see `common.cuh`).
- Trellis: EXL3's tiles unchanged (256 K-bit codes, MSB-first circular stream read as u32,
  `tensor_core_perm` order); only the tile order is changed from EXL3's `[k/16][n/16]` to
  `[n/16][k/16]` so a warp streams a column strip contiguously. The lane decoding positions 8l..8l+7
  holds exactly the A fragment (n = g, g+8; k = 2t, 2t+1, 2t+8, 2t+9) of m16n8k16.
