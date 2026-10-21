# Open-interval sampler regression

The Gumbel sampler requires `0 < u < 1`. The old 24-bit midpoint expression
rounds its highest midpoint to exactly `1.0f` in binary32; `-log(-log(u))`
then becomes positive infinity. Using 23-bit midpoints keeps both endpoints
excluded with equally represented bins. The mixer and greedy path are unchanged.
Stochastic outputs for the same seed can change; this is not a speed optimization.

Run without model weights, CUDA, network, or a running server:

```sh
python3 ds41f-4xgx10-switchless/micro/sampling_unit/check.py
```

The test compiles the **actual function body** extracted from `sample.cu`
(including the actual `common/spec_sampling.h` when it is a forwarding wrapper),
exhausts all 16,777,216 high-24-bit patterns, checks open bounds, finite Gumbel
values, endpoint values, low-bit invariance and uniform paired representation,
and reproduces the one endpoint failure in the old expression.

On Windows, run from an MSVC developer environment with `--cxx cl`.

On a spare GPU, the same standalone test can explicitly be run with:

```sh
python3 ds41f-4xgx10-switchless/micro/sampling_unit/check.py --cuda --nvcc /usr/local/cuda/bin/nvcc
```

The default test is host arithmetic, not proof of a complete CUDA engine build
or of arbitrary CUDA compiler flags. The GPU option initializes a CUDA context;
do not run it on a machine serving another workload without reserving resources.
