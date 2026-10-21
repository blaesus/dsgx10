# Native vision encoder worker

This optional Python worker loads the checkpoint's vision tower, aligner and
learned image separator rows. It returns BF16 image embeddings over a private
Unix socket, with a byte-bounded CPU cache. It does not replace images with text
captions or load the language model.

This change supplies the worker independently. It does **not** connect images to
the existing Rust frontend or CUDA engine, change their wire protocol, or enable
visual chat by itself. A compatible caller must place the returned image spans
in the language-model input and preserve their order and identity. No service is
started by installation, tests or CI.

## CPU tests without weights

Python 3.12 and the packages in `requirements.txt` are sufficient for the tests.
Use an isolated environment. From the repository root:

```sh
python3 -m venv .venv-vision
. .venv-vision/bin/activate
python -m pip install 'torch>=2.4' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r ds41f-4xgx10-switchless/vision/requirements.txt
python -m unittest discover -s ds41f-4xgx10-switchless/vision/tests -v
```

The 11 tests generate small images in memory. They exercise actual CPU image
preprocessing, span layout, input validation, cache reuse/eviction and response
byte limits. Cache tests run `Encoder.encode` with an explicit fake CPU vision
tower. The large-history case checks geometry and byte accounting without
allocating that history. The tests load no model weights, call no GPU or network,
and do not establish model answer quality, GPU numerics or inference performance.

## Explicit worker launch

Supply an existing checkpoint separately. The development checkpoint was
[`deepseek-ai/DeepSeek-V4.1-Flash-1017` at
`1bae0e1088993b9b82de08f59de5be5a34e42170`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170).
The worker reads `config.json`, `model.safetensors.index.json` and the indicated
vision tensors from the original checkpoint. It asserts a text dimension of
5120, image token ID 129264 and a maximum of 1024 visual rows per image; it is not
a generic vision-model server. Choose a PyTorch build compatible with your GPU
and driver before launching.

From `ds41f-4xgx10-switchless`, after reserving the required resources:

```sh
python3 vision/encoder.py \
  --checkpoint /path/to/checkpoint \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --socket /path/to/private/vision.sock \
  --device cuda
```

This command loads weights and uses the GPU. Create the socket's parent
directory beforehand with access restricted to the service user. The worker
creates its socket with owner-only permissions and refuses to replace an
existing socket. It is a single-process, sequential Unix-socket service, not an
Internet-facing HTTP endpoint or a deployment supervisor.

`--revision` is a caller-supplied identity string, not a remote revision
verification or download. The encoder identity also hashes the local config,
Pillow version, the three runtime Python files and all loaded vision tensor
names/dtypes/bytes. Cache identity additionally includes the original image
bytes. Metadata returned for a cache hit does not mutate the stored entry.
Use the actual worker identity; do not copy an old cache identity into a new
deployment. Source hashes and line-ending details are in `PROVENANCE.json`.

## Socket contract

Each connection carries one request: a little-endian `u32` JSON byte length,
followed by UTF-8 JSON. `{"op":"health"}` returns health and cache counters.
Encoding requests use `{"images":[...records...]}`. Supported image records
contain base64 `data`, `source.data`, an image data URL in `url`, or an HTTP(S)
URL. Each record is decoded and validated; animated images and local file paths
are rejected.

The response begins with a little-endian `u32` JSON header length and that
header. A successful encoding header includes `ok`, `identity`, `revision` and
an `images` list in request order. Each image describes:

- `rows`, `dim`, `digest` and `embedding_sha256`;
- `vit_grid`, `llm_grid`, `encode_ms` and `cache_hit`.

The header is followed by one contiguous BF16 byte payload per image in the
same order, with exactly `rows * dim * 2` bytes each, including learned
start/newline/end rows. The intended deployment uses little-endian hosts.
Health responses have no payload. On error the worker sends `ok:false`, a
bounded generic error message and either `image_feature_budget` or
`image_encoding`; it does not echo URLs or image bytes. Callers must inspect
`ok` before reading payloads and enforce their own header and total-byte limits.

HTTP(S) inputs permit outbound requests and redirects within those schemes.
This is **not** a network-isolated decoder or an SSRF filter. Restrict URL inputs
and network access appropriately before allowing untrusted callers. Base64
inputs avoid remote image fetching. Per-image URL reads use a 20-second timeout;
socket I/O uses 180 seconds. These are I/O bounds, not a hard GPU-compute or
whole-request deadline; a deployment needs its own lifecycle supervision.

## Independent resource limits

| Resource | Bound |
| --- | ---: |
| Request JSON bytes | 256 MiB |
| Original compressed image bytes | 16 MiB per image |
| Decoded pixels | 40 million per image |
| Frames | One per image |
| Visual rows | At most 1024 per image for the supported checkpoint |
| Returned BF16 features | 512 MiB per exchange |
| CPU feature cache | 1 GiB, byte-accounted LRU |

These limits apply to different resources. The cache limit does not cap total
process RSS: decoded pixels, preprocessing tensors, response references and
other transient allocations also exist. A large history must be requested in
bounded batches by the caller; this worker does not implement history
pagination or a language-model KV cache. Repeating the same image bytes can
reuse encoded features, but does not establish reusable language-model state.
Cache state is in memory and is not restored after worker restart.

## Source and attribution

The three runtime files are unchanged from the public fork's qualified worker
snapshot (with the already-published LF normalization). They have not been
combined with frontend, CUDA, speculative-decoding or production configuration
changes. `reference_image_processor.py` follows the bundled DeepSeek reference;
`reference_vision.py` adds an explicit batch dimension to SDPA and moves RoPE
cosine/sine tensors to the input device. They remain under the
[DeepSeek MIT license](../ref/LICENSE-deepseek); see the root `NOTICE` and
`PROVENANCE.json`. No model weights or private evaluation transcripts are included.
