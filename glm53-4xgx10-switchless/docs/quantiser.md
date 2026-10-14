# The quantiser prototype (2026-10-13/14)

The code that turns GLM-5.3's BF16 routed experts into the candidate ~3.75-bit formats
([DESIGN](../DESIGN.md) decision 1, experiments 3 and 6), measured on every routed expert of two MoE layers
(3, the first, and 40) fetched from the master. All CPU (`nice -n 19`, ≤ 8 threads), nothing on the GPUs.
Numbers are *measured* unless marked *estimate*.

```
python -I quant/fetch_master.py OUT --layers 3 40                       # range-fetch the experts, SHA-256 each
python -I quant/run.py OUT NV RES --layers 3 40 [--threads 4]           # grid formats on every expert
python -I quant/run.py OUT NV RES --layers 3 40 --formats trellis_k3,trellis_k4,e2m1sub3x2_g16,nvfp4 \
       --experts 0,51,102,153,204,255 --trellis-experts 0,51,102,153,204,255 --tag trellis
python -I quant/subset_study.py OUT --layers 3 --experts 0,32,64,96     # the E2M1 subset choice
python -I quant/summarise.py RES/results-*.jsonl                        # the tables below
python -I quant/summarise.py --compare RES/a.jsonl RES/b.jsonl          # the reproducibility check
```

`NV` is the NVFP4 checkpoint snapshot (`nvidia/GLM-5.3-NVFP4` revision `e3b8f9f2`), `python` the CPU torch 2.11
environment. `trellis.c` is compiled on first use (gcc, `-O3 -mcpu=native -ffp-contract=off`, no fast-math) into
`~/.cache/glm53-quant/`.

## 1. Files

| File | What |
|---|---|
| `quant/fetch_master.py` | the routed experts of chosen layers from `zai-org/GLM-5.3-BF16` revision `9d2398f478cab2de883137db3a36ad2c96205e24` by HTTP range requests (index, then each shard's header, then each tensor's bytes); resumable (`.part` files renamed when complete); `manifest.jsonl` with shard, dtype, shape, bytes and SHA-256 of every tensor |
| `quant/formats.py` | `nvfp4_rtn`, `levelset3` (the E2M1-subset code, one or two subsets), `int3` (g16/g32, e4m3 or bf16 scales); each with its packer and a dequantiser that reads the packed bytes |
| `quant/trellis.py`, `quant/trellis.c` | the EXL3 trellis: codebooks (mul1, 3inst, mcg), the regularisation (signs, channel scales, Hadamard 128 both sides, global scale search), the tail-biting Viterbi in C (NEON, OpenMP over tiles), exllamav3's packing, the dequantiser |
| `quant/run.py` | the driver: every format on every expert, metrics, packed-byte hashes, NVFP4 compared byte for byte with the checkpoint, mixtures at the expert level |
| `quant/subset_study.py` | all 70 single E2M1 subsets and the best pairs, on sampled rows |
| `quant/summarise.py` | the tables; `--compare` checks two runs' packed bytes |

Every master tensor is re-hashed against the manifest when read. Every metric goes through the stored bytes
(quantise → pack → unpack → dequantise), so the decoders are part of what is measured.

## 2. The formats, exactly

Weights are torch-layout `W[out][in]`; groups run along `in` (the GEMM's K). Shapes a layer: 256 experts ×
gate and up `[2048][6144]`, down `[6144][2048]`, 12,582,912 weights each.

**`nvfp4` (NVIDIA's recipe, 4.5 bpw + 32 bits a tensor).**
- `s2 = f32(amax) × f32(1/2688)`, a multiply by the f32 reciprocal. f32 division `amax / 2688` differs in the
  last bit on about a third of tensors (found here: 3 of 9 tensors of layer 3 experts 0–2). **`amax` is taken over
  gate and up together**, `max(amax_gate, amax_up)` (model.md § 5.2 noted they share `weight_scale_2`), and over
  the tensor for down.
- block scale `sc = e4m3_rne(amax₁₆ / 6 / s2)`; value `v = W / (f32(sc) · s2)` (0/0 → 0); code = E2M1
  round-to-nearest with ties to the even code, saturating at 6; nibble = `sign << 3 | index`, `sign = (v < 0)`
  (a value rounding to 0 keeps its sign bit); byte = even column in the low nibble.
- provenance.py's single-tensor check on layer 10 used division and per-tensor `amax`, which agree with this on
  those two tensors only by chance of the last bit.

**`e2m1sub3` / `e2m1sub3x2` (the E2M1-subset 3-bit code, "S3"; 3.5 bpw + 32 bits).**
- A subset is four E2M1 magnitudes `m0 < m1 < m2 < m3` (`m0` may be 0); code `c = s << 2 | j` means `(−1)^s · m_j`.
- One subset (`e2m1sub3`: {0.5, 2, 4, 6}) or two (`e2m1sub3x2`: A = {0.5, 2, 4, 6}, B = {0.5, 1.5, 3, 6}). With two,
  **bit 7 of the group's e4m3 scale byte selects the subset** and bits 6:0 are the scale. Scales are positive, so
  the second subset costs no bits.
- `s2 = f32(amax) / f32(t · 448)`, `t` the smallest top magnitude among the subsets in use (6 for both sets here).
- Per group of 16 the quantiser tries every subset × multiplier `m ∈ {1, .95, .9, .85, .8, .75, .7}`:
  - scale `e4m3_rne(amax₁₆ / m3 · m / s2) · s2`;
  - each value to the nearest level (ties to the lower level);
  - the candidate with the smallest squared error wins, the first in (subset, multiplier) order on ties.
- Packing: codes `[N][K/16][6]`, each group a little-endian 48-bit word with value j at bits 3j…3j+2; scales
  `[N][K/16]` bytes; `s2` f32. Dequantisation: `W = (−1)^(c>>2) · mags[s>>7][c & 3] · e4m3(s & 0x7f) · s2`.
  This is the canonical S3 tensor in [experts.md](experts.md) (`kS3`), so the kernels' layout packers take this
  output as is.
- The subset choice comes from `subset_study.py`: layer 3, experts 0/32/64/96, 512 sampled rows each, mean relative RMS
  weight error gate / up / down:

| Subsets | Error |
|---|---|
| {0.5, 2, 4, 6}, the best of the 70 singles | 0.1547 / 0.1547 / 0.1638 |
| {0.5, 1.5, 3, 6} (2nd single) | 0.1598 / 0.1597 / 0.1659 |
| {0.5, 2, 4, 6} + {0.5, 1.5, 3, 6}, the best of 45 pairs among the top 10 | **0.1429 / 0.1429 / 0.1510** |
| {0.5, 2, 4, 6} + {0.5, 2, 3, 6} | 0.1466 / 0.1466 / 0.1556 |
| {0.5, 2, 4, 6} + {0, 0.5, 1, 2} | 0.1491 / 0.1491 / 0.1541 |

  The pair cuts the error 8 % at the same bytes. (experts.md's tests used {0.5, 2, 4, 6} + {0, 1, 3, 6}; any pair
  decodes through the same tables.)

**`int3_g{16,32}_{e4m3,bf16}` (signed int3 {−4…3}).**
- e4m3 scales: `s2 = amax / (4 · 448)` and a candidate scale `e4m3_rne(amax_g / t · m / s2) · s2`.
- bf16 scales: `bf16(amax_g / t · m)`.
- In both, `t ∈ {4, 3}` (the group's extreme on the −4 or the +3 side) and `m` as above: 14 candidates. Values are
  rounded to nearest-even and clamped, and the best squared error wins.
- Codes `q + 4` are packed as S3's. −4…3 are E2M1 values, so the FP4 tensor-core path is native.
- bpw: g16 e4m3 3.5, g32 e4m3 3.25, g32 bf16 3.5, g16 bf16 4.0.

**`trellis_k3` / `trellis_k4` (EXL3, data-free).**
- Follows exllamav3 at commit `3a513ac9f91aae777f317ca2341b8e445b62c64e` (`modules/quant/exl3_lib/quantize.py`
  `quantize_exl3` → `regularize` → `fallback_quant`; `exllamav3_ext/quant/quantize_tiles_kernel.cuh`, `codebook.cuh`,
  `pack.cu`; read on GitHub, not run; it needs CUDA).
- `W` is taken as exllamav3 holds it, `(in, out) = weight.T`:
  1. ±1 sign vectors `su`, `sv` (numpy PCG64, seed 0);
  2. output-channel RMS scales, divided by their mean, folded into `sv`, and `W /= sv`;
  3. a Sylvester Hadamard of 128 (scaled 1/√128) on every 128 output columns;
  4. input-row RMS scales folded into `su` with the factor −1/1.24371088, and `W /= su`;
  5. the Hadamard on every 128 input rows;
  6. a global scale `g` from exllamav3's two-stage grid:
     - sample: a wrapped diagonal of width 3 plus the highest- and lowest-RMS tiles;
     - coarse grid: 10 scales on a third of the sample;
     - fine grid: 5 scales on the whole sample, refined by a parabola;
     - the sample is multiplied by the K's LDLQ drift (1.004 at K3, 1 at K4);
     - then `W *= g`, `su /= g`.
- 16 × 16 tiles, each permuted to the tensor-core order (`tensor_core_perm`), are quantised as **one tail-biting
  ring of 256 16-bit words by Viterbi**:
  - word i's top 16−K bits are word i−1's low bits;
  - first an unconstrained pass on the ring rotated by 128, traced back to fix the wrap-around state;
  - then a pass constrained to start there, traced back from it.
- Codebook **mul1** (exllamav3's default, "1MAD"): `t = x · 0x83DCD12D mod 2³²`,
  `value = fma_f16(f16(1024 + Σ bytes of t), f16 0x1EEE, f16 0xC931)`. `3inst` and `mcg` are implemented too
  (codebook std 1.0003, 1.2437, 1.2441, matching exllamav3's constants).
- Stored: the trellis packed as `ext.pack_trellis`: the K branch bits MSB-first in 16 spans, the u16 halves of each
  u32 swapped, 16·K u16 a tile. Also `suh = f16(su)` and `svh = f16(sv)`.
- Dequantisation: decoded tiles → Hadamard on rows → `· suh` → Hadamard on columns → `· svh`.
- bpw: K + 16·(in + out)/(in·out) = K + 0.0104.
- **Differences from exllamav3's kernel:**
  - costs and inputs in f32 where the kernel converts the tile to f16 and keeps f16 costs;
  - ties go to the smallest branch and the smallest final state;
  - the RNG for the signs is numpy's, not torch's.
  The codes are the same format, and exllamav3's dequantiser would read them; the bytes would not match
  exllamav3's own output.
- Checks (`trellis.py` against exllamav3's own tests' properties):
  - every tile's words form a closed ring;
  - decode(words) equals the returned values;
  - a random valid codeword is recovered exactly (K3, K4);
  - Gaussian tiles at the codebook scale give MSE 0.0175 (K3) and 0.0046 (K4), under exllamav3's test bounds
    (0.1) and near the rate-distortion bounds 1/64 and 1/256;
  - pack → unpack is the identity.
- The C kernel: NEON, 4 lanes; the 2^K branch accumulators stay in registers per group of 2^K states; OpenMP over
  tiles; `-ffp-contract=off`, no fast-math. Each tile is independent, so the thread count cannot change a bit.
  Single-threaded, a tile takes 12.7 ms at K3 and 4.3 ms at K4 (*measured* on Gaussian tiles); an unvectorised
  first version took 71 and 52 ms.

**Metrics.**
- **bpw**: everything stored, including scales and `s2`.
- **Weight error**: `‖W − Ŵ‖ / ‖W‖` against the master.
- **Output error on Gaussian inputs**:
  - `x = g ⊙ z`, `z ~ N(0, 1)`, 512 rows, seed 1000 + layer, `g` the layer's `post_attention_layernorm.weight`
    (the RMSNorm output of a unit-RMS hidden state);
  - gate and up see `x`; down sees `h = silu(x W_gᵀ) ⊙ (x W_uᵀ)` from the master;
  - error `‖x(W − Ŵ)ᵀ‖ / ‖xWᵀ‖`.
- **Expert output error**: the same for `down(silu(gate x) ⊙ up x)` with all three projections quantised. Mixtures
  are scored only this way.
- Scale check against `input_scale` (`amax_calib = input_scale · 2688`):
  - `x`'s max is 0.23 against a calibrated 0.35 in layer 3, and 1.47 against 2.19 in layer 40;
  - `h`'s is far below the real one: 0.008 against 1.0, and 1.45 against 18.9.
  So the gate/up numbers use realistic magnitudes, and the down numbers do not carry the real activation outliers.
- `--cov DIR` adds `√(tr(EΣEᵀ) / tr(WΣWᵀ))` with a real per-expert input covariance (`layer{L}_expert{E}.pt` with
  `gate_up` and `down` matrices) once the reference's calibration statistics exist. Not run yet.

## 3. The data and reproducibility

- **Fetch** (2026-10-13 21:12 → 22:10, rank 0, 4 connections: 58 min, 11 MB/s on average against the ~24 MB/s
  assumed for one stream): the 1,536 routed-expert tensors of layers 3 and 40, 38.65 GB, from 9 of the 282 master shards. There
  were 10 retried requests and no failures. The SHA-256 of every tensor's bytes is in the fetch directory's
  `manifest.jsonl` (kept on rank 0's scratch disk for the fill and the reference; nothing deleted).
- **NVFP4 reproduced bit for bit: 1,536 of 1,536 tensors** of the two layers (codes, block scales and
  `weight_scale_2`, byte-compared with the checkpoint; 2026-10-14). The recipe took three tries, all recorded:
  1. provenance.py's rule (f32 division, per-tensor `amax`): 3 of 9 tensors on layer 3's experts 0–2;
  2. `amax` shared by gate and up, `s2` a multiply by `f32(1/2688)`: 9 of 9 there, then 1,296 of 1,320 over the
     full run. 24 tensors had block scales off by one e4m3 step at near-ties (`amax₁₆/6/s2` = 91.99999 or 168
     exactly, where NVIDIA has 96 and 176);
  3. block scale `e4m3(amax₁₆ / (6·s2))`: **all 1,536**.

  The checkpoint's experts are therefore a function of the master that we compute. This is the anchor track for
  experiment 3, no download needed.
- **Byte-identical second run**: **93 of 93** packed tensors identical (2026-10-14):
  - every format, including trellis K3 and K4, on layer 3 and layer 40, experts 0 and 255;
  - rerun with 3 threads against 2 or 4 in the first run, in a separate process, compared by the SHA-256 of the
    packed bytes (`summarise.py --compare`).

## 4. Results (layers 3 and 40, 2026-10-14, measured)

Relative RMS errors against the BF16 master. Means over experts:
- grid formats: all 256 experts of layer 3, experts 0–183 of layer 40 (the run was stopped there, see § 6);
- trellis sample: experts 0, 51, 102, 153, 204, 255 of both layers, scored with the same metrics and, for
  pairing, alongside NVFP4 and S3×2 on the same experts.

**Weight error, gate / up / down, and bits a weight:**

| Format | bpw | Layer 3 | Layer 40 |
|---|--:|---|---|
| trellis K4 (6 experts) | 4.010 | 0.0667 / 0.0667 / 0.0667 | 0.0667 / 0.0667 / 0.0667 |
| NVFP4 as published | 4.500 | 0.0946 / 0.0946 / 0.0941 | 0.0951 / 0.0951 / 0.0951 |
| trellis K3 (6 experts) | 3.010 | 0.1319 / 0.1319 / 0.1319 | 0.1319 / 0.1319 / 0.1319 |
| S3×2: E2M1 subsets {0.5,2,4,6}+{0.5,1.5,3,6}, g16 e4m3 | 3.500 | 0.1427 / 0.1427 / 0.1482 | 0.1418 / 0.1418 / 0.1418 |
| S3: E2M1 subset {0.5,2,4,6}, g16 e4m3 | 3.500 | 0.1543 / 0.1542 / 0.1599 | 0.1520 / 0.1520 / 0.1521 |
| int3 g16 bf16 | 4.000 | 0.1657 / 0.1656 / 0.1668 | 0.1618 / 0.1618 / 0.1619 |
| int3 g16 e4m3 | 3.500 | 0.1666 / 0.1665 / 0.1678 | 0.1627 / 0.1627 / 0.1628 |
| int3 g32 bf16 | 3.500 | 0.1850 / 0.1849 / 0.1881 | 0.1793 / 0.1793 / 0.1795 |
| int3 g32 e4m3 | 3.250 | 0.1856 / 0.1855 / 0.1888 | 0.1800 / 0.1799 / 0.1801 |

**Output error on Gaussian inputs**:
- per projection, gate / up / down;
- for the whole expert, mean (p90, max over experts);
- mixtures are scored at the expert level only, and their bpw is averaged over the three equal-sized projections.

| Format | bpw | L3 per projection | L3 expert | L40 per projection | L40 expert |
|---|--:|---|---|---|---|
| trellis K4 (6) | 4.010 | 0.0662 / 0.0663 / 0.0649 | 0.0992 (0.110) | 0.0667 / 0.0667 / 0.0663 | 0.1142 (0.115) |
| NVFP4 | 4.500 | 0.0894 / 0.0894 / 0.0850 | 0.1318 (0.146, 0.157) | 0.0951 / 0.0951 / 0.0939 | 0.1618 (0.164, 0.165) |
| mix K3 gate/up + K4 down (6) | 3.344 | | 0.1623 (0.186) | | 0.1952 (0.198) |
| mix S3×2 gate/up + NVFP4 down | 3.833 | | 0.1742 (0.192, 0.207) | | 0.2167 (0.220, 0.221) |
| trellis K3 (6) | 3.010 | 0.1309 / 0.1311 / 0.1285 | 0.1961 (0.216) | 0.1319 / 0.1319 / 0.1310 | 0.2246 (0.227) |
| S3×2 | 3.500 | 0.1328 / 0.1329 / 0.1236 | 0.1982 (0.217, 0.232) | 0.1417 / 0.1417 / 0.1400 | 0.2397 (0.243, 0.244) |
| S3 | 3.500 | 0.1439 / 0.1440 / 0.1351 | 0.2173 (0.235, 0.249) | 0.1519 / 0.1519 / 0.1501 | 0.2566 (0.260, 0.261) |
| int3 g16 bf16 | 4.000 | 0.1539 / 0.1540 / 0.1426 | 0.2321 | 0.1616 / 0.1616 / 0.1596 | 0.2724 |
| int3 g16 e4m3 | 3.500 | 0.1551 / 0.1552 / 0.1440 | 0.2341 | 0.1627 / 0.1627 / 0.1607 | 0.2741 |
| int3 g32 bf16 | 3.500 | 0.1767 / 0.1767 / 0.1656 | 0.2714 | 0.1791 / 0.1791 / 0.1772 | 0.3011 |
| int3 g32 e4m3 | 3.250 | 0.1776 / 0.1775 / 0.1665 | 0.2728 | 0.1799 / 0.1798 / 0.1779 | 0.3024 |

On the same six experts, layer 3 / layer 40 expert output error: NVFP4 0.1323 / 0.1629, S3×2 0.1991 / 0.2411,
trellis K3 0.1961 / 0.2246.

No KL yet: these are weight and output proxies. KL needs experiment 3's tracks in the reference.

**Spread across experts and the hardest ones:**
- **The weight error is flat.** Across all 256 experts of layer 3, NVFP4 gate runs from 0.0939 to 0.0950 and S3×2
  gate from 0.1422 to 0.1439. Data-free trellis is flat to four digits (K3 0.1318–0.1319 on every tensor), because
  after the Hadamard rotations every tensor looks Gaussian to the codebook and the error is the code's
  Gaussian MSE at the chosen scale. **Weight error cannot rank experts for a bit allocation**, least of all for
  trellis.
- **Where the spread is.** The expert output error varies in layer 3 (NVFP4 mean 0.132, max 0.157; S3×2
  0.198, max 0.232) and hardly in layer 40 (0.162–0.165). The down projection's error ranges widest:
  - S3×2 down weight error 0.142–0.188 in layer 3;
  - output error 0.082–0.141 in layer 3.
- **The ranking is the expert's, not the format's.** It is the same in every format: in layer 3 the hardest
  experts are 164, 225, 137, 252, 178 (NVFP4 and S3×2 agree on 4 of the top 5), and on the trellis sample the
  order 204 > 153 > 102 > 51 > 255 > 0 is identical for NVFP4, S3×2 and K3.
- **Layer 40 errs more than layer 3** at the expert level in every format (×1.15–1.23), for the same weight error:
  its SwiGLU amplifies more on these inputs.
- **Caveat.** These are Gaussian inputs, which carry no activation outliers (§ 2). The real per-expert covariance
  from the reference is what should rank experts and projections.

**CPU time a tensor** (12.6M weights; wall with the process's thread count, beside the other runs at nice 19):

| Method | CPU s | Wall s | Threads |
|---|--:|--:|--:|
| NVFP4 | 0.1 | 0.1 | 2 |
| S3 (one subset, 7 scales) | 1.7 | 0.9 | 2 |
| S3×2 (two subsets, 14 candidates) | 3.3 | 1.6 | 2 |
| int3 (14 candidates) | 2.8–3.1 | 1.4–1.6 | 2 |
| trellis K3 | 280–282 | 71–72 | 4 |
| trellis K4 | 285–291 | 72–74 | 4 |

A trellis tensor is 49,152 tiles in the main Viterbi plus ~10,800 tiles in the global-scale search (+22 %). On
an idle box with 8 threads, one tensor took 36 s wall (layer 3, expert 0, 2026-10-13).

## 5. Projection to all 75 MoE layers (57,600 expert tensors)

| Step | Measured base | All 75 layers |
|---|---|---|
| fetch the master's experts (1.45 TB) | 11 MB/s with 4 connections | **~37 h** (the 16.8 h in quantisation.md assumed 24 MB/s); 19.3 GB on disk a layer at a time |
| NVFP4 + S3×2 (+ int3, S3) on the CPU | 3.4 CPU-s a tensor for NVFP4 + S3×2 | ~54 CPU-h, **~7 h at 8 threads**: hidden under the fetch |
| trellis, one K a tensor, CPU | ~285 CPU-s a tensor | ~4,560 CPU-h: **~24 days at 8 threads**, ~6 days on 4 boxes × 8 threads; both K for every tensor doubles it |
| trellis on a GPU (*estimate*) | a tile is 2¹⁶ candidates × 256 steps × 2 passes ≈ 33.5M evaluations, ~100M instructions in half2; 48 SMs at ~15 T lane-instructions/s → ~7 µs a tile ideal, ~1 s a tensor with the kernel's per-step syncs | ~16 h of GPU, ~64 preemptible 15-min windows; the same as drowzeys' "days" with LDLQ added |

A GPU window changes the trellis track from weeks to hours. Nothing else here needs the GPU. Cheaper trellis on
the CPU:
- one global scale a layer instead of a tensor (−18 %);
- data-free trellis with one fixed scale per K is plausible, since its error is the same on every tensor. Not
  measured.

## 6. Recommendation for experiment 3's tracks

1. **Anchor: NVFP4 as published**, computed from the master by `nvfp4_rtn` (bit-identical to the checkpoint on
   all 1,536 tensors), or read from the checkpoint. It is the same thing.
2. **Main track: S3×2 gate/up + NVFP4 down**, the layout experiment 1 chose for speed:
   - expert output error 0.174 / 0.217 (layers 3 / 40) against NVFP4's 0.132 / 0.162;
   - 3.83 bpw, about at the 3.82 cap. Experiment 3 decides which downs move to S3×2 (or which gate/ups to NVFP4)
     from the real per-expert output error.
   - Use the pair **{0.5, 2, 4, 6} + {0.5, 1.5, 3, 6}**: 8 % less weight error than one subset at the same bytes.
     It decodes through experts.md's existing two-subset tables. The kernels' tests used {0, 1, 3, 6} as the second
     subset, so the loader's tables need the measured pair.
3. **Bracket: S3×2 everywhere** (3.5 bpw, expert error 0.198 / 0.240), the floor of the grid family.
4. **Trellis as a reference track on one or two layers only**: K3 gate/up + K4 down (3.34 bpw), expert error 0.162 /
   0.195, beats the main track's 0.174 / 0.217 at half a bit less.
   - That is the size of what a tuned trellis kernel would buy. Experiment 1 measured its decode at 55–75 % of
     the floor.
   - All 75 layers cost ~24 days of CPU or ~16 h of GPU windows. One layer is ~7.6 h at 8 threads: enough to
     measure the KL gap per layer.
5. **Drop int3.** It is worse than S3 at every bit rate: int3 g16 with bf16 scales at 4.0 bpw errs more (0.232 /
   0.272) than S3×2 at 3.5 (0.198 / 0.240).
6. **Allocate by output, not by weight error.** Weight error is flat across experts, and exactly flat for trellis.
   The expert ranking comes from the SwiGLU and the inputs and is the same in every format. Hence:
   - feed `run.py --cov` the reference's per-expert input covariances (gate/up input and down input);
   - then KL in the tracks.

## 7. What is not done, and failures

- **No KL**, and no real-input metric: the per-expert covariances from the reference on rank 2 were not available to
  this run. `--cov` is written and untested on real data.
- **Layer 40's grid run covers experts 0–183.** It was stopped on 2026-10-14 to free the CPU, after the spread had
  shown itself flat (p90 to max within 0.001). The NVFP4 check ran separately on all 256.
- **The mixture rows were scored before the NVFP4 block-scale fix** (§ 3, try 2): the NVFP4 down in
  `mix_sub3x2gu_nvfp4d` differs from the checkpoint on 2 of the 440 down tensors (layer 3 expert 180, layer 40 expert 78: one e4m3 step
  on near-tie blocks).
  The effect on the means is below the fourth digit.
- The **down projection's Gaussian inputs are unrealistic**: no activation outliers, 0.008 against a calibrated 1.0
  in layer 3.
- The **S3 subset pair** was chosen on layer 3 only (4 experts × 512 rows). Layer 40's full run confirms its gain
  (0.1418 against 0.1520) but did not re-search the pair.
- **Speed.** The first trellis C kernel was unvectorised at 71 ms a tile; NEON brought it to 12.7 ms (K3)
  single-threaded. The grid run went at ~40 s an expert beside the trellis run, so it was split into two
  2-thread processes.
- **Disk.** Rank 0's /data had 103 GB free at the start and 33 GB at 2026-10-14 04:50: 37 GB is this fetch, the
  other ~33 GB other work. The fetched tensors are kept for the fill and the reference.
- The CPU jobs were not pinned off the ring's cores 5–8 (engine.md T3). They ran at nice 19 with ≤ 8 threads.
