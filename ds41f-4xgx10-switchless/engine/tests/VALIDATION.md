# Clean probability PR: offline validation

Recorded 2026-10-21. Base: upstream
`a92b8f44046f4435c3c29c07fbc39e6e4b157080` plus the open-interval sampler
fix `c6a7a29`. These checks used only local CPU resources and synthetic data.

## Executed

Windows x64, MSVC 19.44.35228, Python 3.12, CMake and Ninja:

- Standalone host project configured and compiled successfully. The two CTests
  (`probability_host_test`, `distribution_oracle`) passed. CUDA 12.4 headers
  enabled the real batch-verdict section, but no CUDA library was linked or
  initialized.
- A separate host compilation without CUDA include paths passed the core RNG,
  metadata, eligibility and residual checks. Its output explicitly reported
  the unavailable batch-verdict section.
- The host RNG check covered all 8,388,608 representable midpoint values.
  The independent Fraction oracle passed seven tests, including 3,969 rational
  distribution pairs and the proposal-dependent look-ahead counterexample.
- `micro/sampling_unit/check.py --cxx cl` compiled the current runtime `unit`
  function: 16,777,216 cases, zero new errors, one reproduced original endpoint
  error. The checker also supports compiling an actual shared-header wrapper;
  it does not substitute the expected expression.
- Regenerating the synthetic JSON/header reproduced both files byte for byte
  after fixing the generator to write LF on all platforms. JSON SHA-256:
  `f5e8eb0977f0d02986f24f8d44018c3aea143768822b0c3be17d8b72a7ad5181`;
  header SHA-256:
  `bb9aa6a307de137f3fe1cc1951b1275dc83a4679c0eafea79f2ea55164e2e32b`.
- `git diff --check` passed. Changed and new text files use LF. The frontend
  protocol header and CED128 constants remain identical to the PR base.

## Extraction boundaries

The drafter and shared probability helper retain the previously integrated
implementation, with attribution comments added. The target sampler retains
the same arithmetic; its existing open-interval comment differs in wording.
The extraction adds invalidation calls to the upstream lifecycle functions
without importing image, tail-replay, Creative steering or external telemetry
changes. The original `prefill(..., bool ced)` interface remains intact.

The option is disabled by default. The CPU-only GitHub workflow is included,
but its hosted execution is separate from these local results. On a runner
without CUDA headers it runs the core host helpers and Fraction oracle; the
existing sampler workflow independently checks the runtime `unit` function.

## Not executed for this clean PR

No full CUDA/SM121a build, four-rank sampler probe, complete DSpark/model run,
long-lived cache/fork/load test, application quality evaluation or performance
benchmark was performed. Earlier integrated-branch GPU results do not qualify
this different upstream combination. The included GPU probe is opt-in, has no
automatic launcher, and must run only on a separately reserved test fleet.
