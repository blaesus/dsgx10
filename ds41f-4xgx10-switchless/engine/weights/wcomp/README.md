# wcomp: how much would the weights compress losslessly?

The tools behind [engine.md's lossless weight compression study](../../../docs/engine.md#lossless-weight-compression-a-feasibility-study-2026-10-12).
They run on the CPU only and read a rank file without loading it.

```sh
g++ -O2 -std=c++17 -o wcomp wcomp.cpp
./wcomp /data/dsgb10/weights/rank0.dsw --list > toc_list.txt
# the study's sample: sample_names.txt, with --max-mb 64 before embed.weight (bf16 sampled in 4 MiB chunks)
nice -n 19 ionice -c3 ./wcomp /data/dsgb10/weights/rank0.dsw --expert-stride 8 <names...> > sample.jsonl
python3 analyze.py sample.jsonl   # bits per code by weight kind -> summary.json, stats.json
python3 project.py [overhead]     # bytes per rank and ms per step by scheme (overhead: 0.0025 by default)
```

- `wcomp` streams each named tensor in 64 MiB reads and drops the pages behind it. It writes one JSON
  line of histograms per tensor:
  - FP8 bytes, per block scale, and after the previous byte;
  - MXFP4 nibbles, byte pairs, nibbles per scale, scales and scale pairs;
  - bf16 words;
  - f32 exponents.
  With `--expert-stride S`, it reads every S-th expert.
- `analyze.py` computes, per tensor kind:
  - order-0 and conditional entropies;
  - Huffman code lengths, unlimited and limited to 12 or 8 bits (package-merge);
  - DFloat11's split: the exponent coded, the sign and mantissa raw.
- `project.py` gives every tensor in the file its kind's bits. It then sums the bytes a decode step
  reads, and turns the bytes saved into ms at 235 GB/s.

Part 2's GPU bench, the fused Huffman GEMV, is [bench_wcomp.cu](../../kernels/tests/bench_wcomp.cu).
Part 3 costs fixed-length codes with these:

```sh
g++ -O2 -std=c++17 -o wfix wfix.cpp
nice -n 19 ionice -c3 ./wfix /data/dsgb10/weights/rank0.dsw --expert-stride 8 $(cat wfix_names.txt) > fix.jsonl
nvcc -std=c++20 -O3 -gencode arch=compute_121a,code=sm_121a -cubin fixdec.cu -o fixdec.cubin
cuobjdump -sass fixdec.cubin > fixdec.sass && python3 sass_loops.py fixdec.sass
python3 fixcost.py fix.jsonl      # ms at k = 1 and 6 rows, decode hidden and at v2's exposure
gcc -I/usr/local/cuda/include dequery.c -o dequery -L/usr/local/cuda/lib64/stubs -lcuda && ./dequery
```

- `wfix` streams like `wcomp`. Per tensor, it writes what fixed-length codes would leave as exceptions:
  - FP8 lane chunks (the 16 bytes a GEMV lane reads a step): their exponent range and (min, max);
  - expert-scale groups (the 32 bytes a MoE warp-step reads): their range, and whether they stay
    within the tensor's best pair or quad;
  - bf16 tiles (512 bytes): their exponent range.
- `fixdec.cu` holds the decoders, compiled only to read their SASS: the expert scales at 1 bit, the
  head's exponent at 4 bits from a tile base and at 5 bits, and FP8 with 3-bit codes. Each has a raw
  twin.
- `sass_loops.py` counts the instructions in each loop of a kernel. The study used it on today's GEMV
  and MoE kernels, bench_wcomp's decoders and fixdec's.
- `fixcost.py` turns wfix's statistics and the decoders' counts into ms at k = 1 and at 6 rows.
- `dequery.c` asks the driver whether the GPU has hardware decompression. GB10 doesn't.
