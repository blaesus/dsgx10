# Paired native image serving (protocol 5)

This change connects the native vision worker to the Rust frontend and the CUDA
engine. It depends on the canonical-input cache fix and the native worker; both
are included as preceding commits. Build and deploy the frontend and **all four
ranks from the same revision**. A text-only old frontend/engine is not a supported
half-upgrade. Protocol 5 rejects unknown request flags, including the later
full-prefix/tail-prefix flags; it never silently promises an unsupported policy.

## Scope and conservative cache policy

- The 1017 encoder preserves image blocks, including tool-result content blocks.
  The worker returns learned BF16 image embeddings, not captions. Identity binds
  the checkpoint/encoder revision and the original image bytes.
- The Rust cache reuses immutable inline image data. Mutable URLs/paths are not
  trusted as content-addressed cache keys. Changed, removed, reordered, or resized
  images clip reuse at the first changed span; token placeholders alone cannot
  identify a picture.
- Pure text keeps the upstream 128-row CED replay. Any image request uses full
  40-layer prefill. Text/image transitions do not reuse each other's state.
  This is a deliberately conservative serving baseline, not the later 2560-row
  replay optimization. It makes no new fast-first-image or full-1M claim.
- No Creative steering, probability SPEC, attention/wo_a numerical patch, new
  administrative gate, or external telemetry subsystem is part of this change.

## Local worker and paired build

Build the existing engine with CUDA 13 on supported GB10 hardware, and the
frontend with Rust using their existing build instructions. Enable vision in
`dsfront serve` with `DSGB10_VISION_SOCKET=/tmp/vision.sock` and
`DSGB10_VISION_REVISION=<worker revision>`, retaining `--encoding 1017`.
Start `vision/encoder.py` as documented in [the worker README](../vision/README.md),
using the same checkpoint/revision. The worker endpoint is a private Unix socket,
not HTTP. Frontend and rank 0 must share the Unix filesystem and service UID.
They must not have isolated `/tmp` mount namespaces. The spool is not guaranteed
to be disk-backed by the program: use a disk-backed `/tmp` or account for tmpfs
RAM separately. Persisted engine sessions (`DSGB10V5`) and frontend tables
(`DSFT0003`) are intentionally incompatible with the old formats; they are not
a transparent old-version cache migration.

Image submissions carry a 40-byte descriptor (start, rows, digest), followed by a
u32 UTF-8 path length and an absolute path. There is no BF16 inline or ImageMiss
retry path. The frontend writes files below
`/tmp/dsgb10-vision-features/<generation>/<file>` in 0700 directories with 0400
files. Pending requests own files through Done/disconnect, including cancellation.
Rank 0 opens restricted, same-UID, regular files with no symlink traversal and an
exact expected length. Rank 0 also checks modification metadata and finite BF16
values; it does not recompute a complete embedding SHA. This boundary trusts the
private same-UID frontend/encoder, not a malicious same-UID writer. Open FDs
survive unlink. File errors fail the request,
not the engine; no rank waits for a payload before all required local reads pass.

## Bounded memory and retained history

Limits remain explicit: 4096 descriptors, at most 1M historical image rows, 1024
rows/10 MiB per image, 12 GiB live spool bytes including surviving generations,
32 MiB engine metadata frames, and 512 MiB active feature memory. Worker requests
are batched at eight images (at most 80 MiB returned features). The complete
layout stays available for Engram's three-token lookback and masking; only whole
images intersecting the current prefill chunk are read and copied. With 8192-row
chunks, disjoint images of at most 1024 rows need at most 10238 feature rows
(about 100 MiB), even if the complete history exceeds 512 MiB. No image is dropped
or downsampled by this transport to meet the historical budget.

A frontend restart may leave accounted spool files behind. This change does not
recursively delete old generations or promise persistent encoder/KV cache hits.
Operators should inspect and remove only generations proven to have no owners.
The quota belongs to one frontend process and conservatively counts old safe
generations on startup. It is not a coordinated quota for concurrent frontends.

## CPU tests and validation limits

On a disposable Linux environment (the filesystem tests use the private `/tmp`
spool root; do not run them next to a live frontend):

```sh
cargo test --manifest-path ds41f-4xgx10-switchless/frontend/Cargo.toml
cmake -S ds41f-4xgx10-switchless/engine/tests -B build/vision-host \
  -DBUILD_TESTING=ON -DDSGB10_ENABLE_FILE_TESTS=ON
cmake --build build/vision-host --parallel 2
ctest --test-dir build/vision-host --output-on-failure
```

Tests cover image identity, changed history, malformed paths, short/non-finite
files, unlink/FD lifetime, cancellation, full-layout/chunk geometry, and a
58-image logical history larger than 512 MiB. Frontend tests use synthetic
embeddings and Unix sockets; the optional 58-image transport test needs
`DSGB10_TEST_TOKENIZER` and at least 700 MiB scratch disk and is ignored by default.
The `vision_test` CUDA target is opt-in; CTest does not launch GPUs or SSH.

Extraction validation: Linux g++ host tests passed (61,102 budget assertions),
and `serve.cpp`/`cmd.cpp` passed C++20 syntax checking with existing CUDA headers.
Local Rust and CUDA toolchains are unavailable for the complete paired build;
the narrow GitHub workflow compiles/tests Rust. This upstream-plus-serving
combination has **not** received a fresh GPU numerical, long visual quality, or
performance qualification. Historical deployment evidence motivates the code,
but cannot substitute for testing this extracted branch. Keep the PR draft
until paired CUDA build and representative image serving are independently checked.
