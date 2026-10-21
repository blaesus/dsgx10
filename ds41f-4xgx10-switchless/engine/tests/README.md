# Synthetic probability tests

No captured conversation, model weights, credentials, fleet configuration, or
production service is required for the offline tests.

## CPU only

From `ds41f-4xgx10-switchless`:

```sh
python3 engine/tests/distribution_oracle.py
cmake -S engine/tests -B build-probability-host -DCMAKE_BUILD_TYPE=Release
cmake --build build-probability-host --config Release
ctest --test-dir build-probability-host -C Release --output-on-failure
python3 micro/sampling_unit/check.py
```

The Fraction oracle independently checks 3,969 small rational distribution
pairs, nine named distributions, first rejection/bonus handling, and a
counterexample for proposal-dependent look-ahead gating. Regenerate its purely
synthetic header/JSON with `--freeze TEMP_DIRECTORY` and compare them before an
intentional fixture update.

The C++20 host test uses actual RNG, metadata, and residual helpers, including
all 8,388,608 representable RNG midpoints. When CUDA headers are already
installed it also includes the actual batch-verdict headers and checks batch
first rejection. Otherwise it explicitly reports that only this section was
skipped; no replacement CUDA headers are used. Neither mode links nor
initializes the CUDA runtime. The standalone project never invokes a CUDA
compiler. The CPU CI workflow runs the core helpers and Fraction oracle. The earlier
`micro/sampling_unit` regression still compiles the actual runtime function;
if it is changed to a wrapper, it compiles that wrapper and shared header rather
than substituting an expected formula. On Windows use
an MSVC developer environment and pass `--cxx cl` to that regression.

## GPU probe: explicit build and execution only

The probe is not registered with CTest. Build it in the normal Linux CUDA 13 /
SM121a engine project:

```sh
cmake -S engine -B build -DDSGB10_BUILD_GPU_PROBE=ON
cmake --build build --target probabilistic_sample_probe -j2
```

It uses the actual `ops::sample` and four-rank native ring, independent CPU
proposal draws, exact-law fixture expectations, stale/unsupported metadata,
multiple cache slots, and CUDA Graph reuse. It does not run the complete
DSpark drafter or a full model. Run four ranks only on an explicitly reserved
test fleet with a fresh port, never on an active production service. Per rank:

```sh
timeout --kill-after=5s 150s ./build/tests/probabilistic_sample_probe \
  --fleet /path/to/test-fleet.conf --rank 0 --port 29680 \
  --trials 4096 --deadline 120
```

Use rank IDs 0 through 3 and coordinate/reap all processes with your own test
infrastructure. The in-process deadline does not prove cleanup under every
peer failure; retain an external timeout. This repository adds no automatic
SSH launcher, inference requests, or service-control operations.

The clean PR combination has not yet been rebuilt or run on that fleet.
Host checks are not CUDA/communication or end-to-end quality qualification.
