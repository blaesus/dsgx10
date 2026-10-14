# quant/

The quantiser prototype for GLM-5.3's routed experts. The method, the definitions, the reproducibility check and
the results are in [../docs/quantiser.md](../docs/quantiser.md) (2026-10-13/14).

| File | What |
|---|---|
| `fetch_master.py OUT --layers 3 40` | range-fetch the BF16 master's routed experts, SHA-256 each (resumable) |
| `formats.py` | `nvfp4_rtn` (NVIDIA's recipe, bit-exact on 1,536 of 1,536 tensors), `levelset3` (the E2M1-subset 3-bit code S3, one or two subsets), `int3`; packers and dequantisers |
| `trellis.py`, `trellis.c` | the EXL3 trellis (exllamav3 3a513ac9), data-free: regularisation, tail-biting Viterbi (C, NEON, OpenMP), packing, dequantisation |
| `run.py MASTER NV OUT --layers ...` | every format on every expert: bpw, weight error, output errors, packed-byte SHA-256, NVFP4 byte comparison, mixtures |
| `subset_study.py` | the E2M1 subset choice |
| `summarise.py` | tables; `--compare A B` checks two runs byte for byte |

CPU only (`python -I`, torch 2.11, numpy, gcc); `nice -n 19`, `--threads` ≤ 8. `trellis.c` builds itself into
`~/.cache/glm53-quant/` on first use.
