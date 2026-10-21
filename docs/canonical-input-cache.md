# Canonical input when reusing session KV

A generated token sequence can decode to `ab` as two IDs, while the tokenizer
encodes a fresh `ab` as one merged ID. Previously, the frontend could copy the
generated IDs into the next request before an added-token boundary. A warm
request therefore had different input IDs from an otherwise identical cold
request, despite equal decoded text.

The frontend now tokenizes the complete incoming prompt and reuses only the
identical token-ID prefix of the selected session. Session selection, the busy
session fork rule, wire protocol, saved-table format, model math, and sampling
settings are unchanged. Full-prompt tokenization costs CPU work on each turn;
the matching canonical prefix still reuses engine KV. No tokenizer throughput
or GPU speed claim is made here.

Run `cargo test --locked -j2` in `ds41f-4xgx10-switchless/frontend`. The three
`canonical_cache` tests generate a minimal byte-BPE tokenizer in memory and
exercise noncanonical history, busy-session forks, and canonical-prefix reuse.
They need no model files, images, GPU, endpoint, or private prompts. Existing
checkpoint-dependent tests are separate; a normal test run does not establish
that those tests executed when their tokenizer is unavailable.

This is the token-identity prerequisite for image-aware cache work. It does not
add image serving, repair semantic visual answers, or claim full-model numerical
or performance qualification of the standalone patch.
