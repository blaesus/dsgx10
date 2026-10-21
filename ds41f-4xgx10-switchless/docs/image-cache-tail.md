# Image-aware prefix policies and bounded decoder replay

This change follows the paired image-serving protocol. It adds explicit prefix
policies while keeping `legacy` as the default:

| Policy | Text prefill | Image prefill | Text-to-image prefix reuse |
| --- | --- | --- | --- |
| `legacy` | Upstream CED128 when eligible | Complete forward | Conservatively invalidated |
| `exact` | Complete forward | Complete forward | Identical causal prefix only |
| `tail-exact` | Encoder plus bounded decoder suffix when eligible | Same bounded suffix policy | Identical causal prefix only |

The added policies identify sessions separately in both the frontend namespace
and engine state. Conflicting flags and cross-policy reuse are rejected. Image
identity includes encoder/checkpoint identity, exact content, position and span
length. Append, replacement, deletion or reordering clips reuse at the first
changed image; matching placeholder token IDs alone never establish identity.

`tail-exact` keeps the full encoder/compressed-memory path and replays 2,560
decoder rows. For this model's twenty decoder layers with causal 128-row windows,
the structural minimum covering the final output and the last 128 layer-input
rows is 2,541. The rounded 2,560-row suffix meets that bound. This assumes the
model's shared decoder KV/index architecture and complete encoder state. The
policy header statically checks both declared finite dependency bounds; that
check alone does not verify the model's runtime architecture or cache contents.
It is an architectural dependency argument, not bitwise equality across GEMM
shapes or a claim that every visual reasoning answer is correct.

Only the suffix after the decoder input dependency boundary is advertised as
valid for restarting a cached session. The entire 2,560-row scratch replay is
not reusable state. Short requests use the full path when they cannot provide
the required replay rows. Replay buffers grow within an explicit capacity
check; DSpark consumes the actual replay partition rather than assuming 128
rows. A complete image layout remains available for Engram masks and replay,
even when only the current chunk's BF16 features are resident.

## Validation

The CPU `ced_tail_policy_test` includes the real policy header but independently
propagates a boolean dependency graph through twenty causal windows. It checks
the advertised input boundary, final output, last KV window, a one-row-short
counterexample, absolute position zero, legacy behavior and unsupported sizes.
It does not run CUDA, attention arithmetic, model weights or cache restoration.
The paired protocol/prefix tests additionally cover image changes and policy
isolation; checkpoint-dependent tests need their documented external tokenizer.

This split retains already integrated code but forms a new upstream combination.
It still needs a full CUDA build and reserved four-rank cold/warm numerical and
performance qualification. No integrated-branch speed figure is claimed as a
benchmark of this standalone PR, and no production service is changed by its
test or publication workflow.
