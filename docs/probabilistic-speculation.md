# Optional probabilistic DSpark proposals

This change is based on upstream `a92b8f4` and the open-interval sampler fix.
It adds an opt-in probability proposal path to the existing DSpark drafter and
native four-rank sampler. **It is disabled by default.** Enable it with
`--spec-probabilistic` alongside the existing speculative-decoding options.
In serve mode, speculation remains enabled unless `--no-spec` is supplied.

## Behavior and scope

For temperature greater than zero, `top_p=1`, and `top_k=0`, the drafter samples
from its effective normalized distribution `q`. Verification accepts a draft
token with probability `min(1, p/q)`. At the first rejection it samples from the
normalized positive part of `p-q`; bonus tokens use the target sampler. The
accepted prefix ends at the explicit rejection flag, even if the replacement
token happens to equal the proposal. Draft, acceptance, and residual draws use
separate random domains indexed by request seed and absolute token position.
Identical explicit seeds across independent requests can correlate; this is not
a promise of independent randomness between sessions.

Temperature zero and unsupported truncation settings use the earlier proposal
and target-sampling path. The optional ratio gate is rejected at startup with
this feature: its look-ahead depth choice may depend on the current random
proposal. Runtime eligibility also excludes that gate. Tree-probe and
batch-graph-check diagnostics are not supported with this option. Request
sampling parameters and the service's defaults are not changed.

Proposal logits remain vocabulary-sharded. Each cache slot stores five FP32
rows of 32,320 logits plus 32 bytes of metadata: 646,432 bytes per rank/slot,
or 10,342,912 bytes for sixteen slots. Verification adds one small collective;
its additional payload is 384 bytes/rank for the eight-row sampler and 768
bytes/rank for the sixteen-row sampler. No complete vocabulary distribution is
gathered onto a single rank.

Proposal metadata is not saved in KV snapshots. Prefill, token-history reset,
given-token priming, state load/restore/end restore, session freeing, and forked
snapshot copying invalidate it. Real cache-slot indices are retained across
batch reordering. CUDA Graphs read current sampling and gate inputs. Stale
metadata and numerical zero-residual fallbacks are observable through flags
and per-request `spec_probability` log counters. `prob_accept` counts evaluated
rows, including rows later discarded after an earlier rejection; use the
existing accepted-prefix counters for end-to-end SPEC acceptance rates.

This PR keeps upstream CED128, the original frontend protocol, native ring,
attention/FFN math, and model weights. It does not add image input, tail2560,
Creative steering, external telemetry, or deployment controls. Enabling the
probability path changes seeded output trajectories; the intended contract is
the target sampling law, not equality with the old greedy-proposal trajectory.
Finite floating-point arithmetic and finite RNG precision remain limitations.

## Provenance

The probability implementation was rewritten for dsgx10 after consulting
DeepInfer revision
[`8b16149741a4935e1dec8cea27853f273b374c5b`](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b),
particularly [random draft generation](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/mtp/dflash.py#L279-L344)
and [acceptance/residual sampling](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/tilelang_ops/cuda/fused_sample.py#L375-L443).
This is not a clean-room implementation or a port of the complete DeepInfer
runtime. The C++/CUDA adaptation uses the native ring, per-slot proposal state,
Graph-compatible inputs, and stable log-difference arithmetic; it does not copy
the reference's residual penalty constant.

Existing upstream portions retain their MIT terms. DeepInfer-derived additions
are provided under Apache-2.0; retain [NOTICE](../NOTICE), the included
[Apache license](../LICENSES/Apache-2.0-DeepInfer.txt), and the source comments.
The consulted DeepInfer revision has no root NOTICE file. No model weights,
private workload fixtures, captured conversations, or service configuration
are included.

Pinned source SHA-256 values below are calculated from that revision's Git
blobs, with their original LF line endings:

| DeepInfer file | SHA-256 |
| --- | --- |
| `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `deepinfer/model_executor/models/mtp/dflash.py` | `12ebfb9c793062844d40e0fab17164d5365d965c318f8b9f0f4adf51b3afb684` |
| `deepinfer/model_executor/layers/tilelang_ops/cuda/fused_sample.py` | `2b289145a9c28bb869dcc4cb63d80eff845d923a3fc34885366a2caa0411223f` |

## Validation status

See [the synthetic test instructions](../ds41f-4xgx10-switchless/engine/tests/README.md).
The [offline validation record](../ds41f-4xgx10-switchless/engine/tests/VALIDATION.md)
separates executed host tests from the unexecuted GPU probe.
The probability arithmetic and drafter code were extracted from an integrated
branch that had bounded four-rank sampler and application tests. That branch
also contained unrelated changes. **This clean combination with upstream
CED128 has not received a new full CUDA build, four-rank GPU run, or end-to-end
model benchmark.** Earlier integrated-branch speed measurements are not a
benchmark or release qualification of this PR. Treat it as a draft until those
checks are completed on a separately reserved test fleet.
