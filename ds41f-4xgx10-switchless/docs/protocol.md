# Front-end ↔ rank-0 protocol

The Rust front-end (`frontend/`, binary `dsfront`) owns HTTP, chat encoding, tokenisation, stop
strings, output parsing and session matching. Rank 0 of the engine owns everything that touches the
model: scheduling, prefill, decode, sampling and the per-session KV state. They talk over one Unix
stream socket that rank 0 listens on. The message layouts are in
[`engine/common/frontend_protocol.h`](../engine/common/frontend_protocol.h) (C++, plain structs) and
`frontend/src/protocol.rs` (Rust); this page is the contract.

## Transport

- `AF_UNIX`, `SOCK_STREAM`. Rank 0 creates the socket (default `/tmp/dsgb10-engine.sock`, mode
  0600) and accepts one front-end at a time; a new connection replaces the old one, whose requests
  are cancelled.
- Every frame is an 8-byte header followed by `size` bytes of payload:

  ```
  struct FrameHeader { uint32_t size; uint16_t type; uint16_t reserved; };   // little-endian
  ```

- Payloads are a fixed struct (a multiple of 8 bytes) followed by arrays in the order given below.
  All integers are little-endian, floats are IEEE `float`. There is no padding between arrays.
- Both sides send `HELLO` first and close the connection if the other side's `version` differs.
- Either side may close the socket at any time. The front-end reconnects with backoff and fails the
  requests that were in flight; rank 0 cancels everything the old connection submitted.
- Rank 0 must never block on the socket: the front-end reads continuously. Frames from rank 0 for
  different requests interleave freely, but for one request all `TOKENS` come before its `DONE`.

## Messages

Front-end → rank 0

| type | name | payload |
|---|---|---|
| 0x01 | `HELLO` | `{u32 version; u32 reserved}` |
| 0x02 | `SUBMIT` | `SubmitHeader`, `u32 stop_ids[n_stop_ids]`, `u32 prompt[n_prompt]` |
| 0x03 | `CANCEL` | `{u64 req_id}` |
| 0x04 | `RELEASE` | `{u64 session_id}`: the front-end will not continue this session; free its state |

Rank 0 → front-end

| type | name | payload |
|---|---|---|
| 0x81 | `HELLO` | `{u32 version; u32 max_context; u32 vocab_size; u32 eos_id; u32 max_sessions; u32 name_len}`, `char name[name_len]` |
| 0x82 | `TOKENS` | `{u64 req_id; u32 n; u32 flags}`, `u32 ids[n]`, then `f32 logprob[n]` if `flags & 1`, then `n * n_top` pairs `{u32 id; f32 logprob}` if `flags & 2` |
| 0x83 | `DONE` | `DoneHeader`, `char error[error_len]` |

```
struct SubmitHeader {            // 72 bytes
  uint64_t req_id;               // unique per connection, chosen by the front-end
  uint64_t session_id;           // KV state to continue (or create); never 0
  uint64_t base_session_id;      // 0, or a resident session a new one may start from (see below)
  uint64_t seed;                 // used when flags & SEED, otherwise rank 0 picks one
  float    temperature;          // 0 = greedy
  float    top_p;                // 1 = off
  uint32_t top_k;                // 0 = off
  uint32_t max_new_tokens;       // >= 1; the front-end has already clamped it to the context
  uint32_t flags;                // 1 SEED, 2 KEEP (keep the session after DONE), 4 LOGPROBS
  uint32_t n_top_logprobs;       // 0..20, with LOGPROBS
  uint32_t prefix_hint;          // prompt tokens the front-end expects to match the session
  uint32_t n_stop_ids;           // token ids that end generation (the front-end sends EOS here)
  uint32_t n_prompt;             // >= 1
  uint32_t reserved;
};

struct DoneHeader {              // 56 bytes
  uint64_t req_id;
  uint32_t reason;               // 0 STOP, 1 LENGTH, 2 CANCELLED, 3 CONTEXT, 4 ERROR
  uint32_t prompt_tokens;        // n_prompt
  uint32_t cached_tokens;        // prompt tokens whose KV was reused, not computed
  uint32_t completion_tokens;    // ids sent in TOKENS frames for this request
  uint64_t queue_us;             // SUBMIT received → prefill started
  uint64_t prefill_us;           // prefill started → first token sampled
  uint64_t decode_us;            // first token → DONE
  uint32_t error_len;
  uint32_t reserved;
};
```

## Requests

- Every `SUBMIT` gets exactly one `DONE`, also when it is rejected, cancelled or fails. `TOKENS`
  frames carry the sampled ids in order, as soon as they are accepted (a speculative step may send
  several at once). A sampled stop id is sent as the last token and ends the request with `STOP`.
  `LENGTH` means `max_new_tokens` ids were sent, `CONTEXT` that the context window is full.
- `CANCEL` for a finished or unknown `req_id` is ignored. Cancelling is how the front-end ends a
  request on a stop string, on a repetition loop, or when the HTTP client goes away; ids that
  rank 0 sends after it are dropped by the front-end.
- Rank 0 may reject a `SUBMIT` (prompt longer than the context, bad arguments, out of memory) with
  `DONE{reason = ERROR}` and a message, and no `TOKENS`.
- Sampling: `temperature` scales logits; then `top_k`, then `top_p` filter; the seed makes a request
  reproducible on its own (one RNG stream per request). Greedy (`temperature == 0`) ignores the rest.
- With `LOGPROBS`, `TOKENS` carries the log-probability of each sampled id under the distribution it
  was sampled from (after temperature, before top-k/top-p), and, if `n_top_logprobs > 0`, the top
  ids of that distribution. The mock engine sends zeros.

## Sessions and exact prefix reuse

DESIGN decision 6: a new turn continues from the state its conversation left behind.

- A session is rank 0's KV state (global KV, compressed KV, SWA window) for one token history,
  keyed by `session_id`. After a request with `KEEP` ends, however it ends, the session holds the
  history whose KV was computed: the prompt (or the part of it prefilled before a `CANCEL`) and the
  generated ids except the last sampled one. A request that ends with `ERROR` leaves the session
  undefined; the front-end then `RELEASE`s it.
- On `SUBMIT`, rank 0 looks up `session_id`. If it is resident, rank 0 computes the longest common
  prefix (LCP) of the session's history and the new prompt, keeps the state up to the latest position
  at or before the LCP that it can rebuild exactly (the live rings, the end snapshot or a window
  snapshot: [engine.md](engine.md#serving), "Exact reuse"), and prefills the rest. `cached_tokens` in
  `DONE` reports the reused length. If the session is not resident and `base_session_id` is, rank 0
  may start the new session from the base's state up to their LCP (a fork); otherwise it is a cold
  prefill. Either way the session then holds the new history. When device memory runs short, sessions
  leave the store least recently used first, to each rank's disk tier (`--disk-gib`, default 32 GiB),
  and come back from it on their next request; only a session dropped from the disk tier, or any
  evicted session with `--disk-gib 0`, comes back cold. The encoder-side SWA bounded replay first planned here is not
  built ([correctness review](review/correctness.md)).
- The front-end never has two requests in flight on one `session_id`, and it only sends `RELEASE`
  for sessions it will not name again. `prefix_hint` is what the front-end believes is reusable; it
  is for logging and need not be trusted.

How the front-end picks the session and the token ids (frontend/src/session.rs):

1. The prompt is rendered to text with the chat encoding. Clients such as pi send no session id,
   so the front-end compares the text with the text of each recent session (prompt text plus the
   text of the generated ids) and takes the one with the longest common prefix.
2. Token ids for the shared part are the session's own ids, i.e. what the model actually saw and
   generated, not a re-tokenisation of the text (BPE of generated text often differs from the
   sampled ids). The cut is the last position, within the shared text, that is a token boundary in
   the session's ids and sits next to an added token (`<｜User｜>`, `<｜end▁of▁sentence｜>`, `</think>`,
   ...) in the new text, so the rest tokenises exactly as it would on its own. The rest is
   tokenised normally.
3. If the shared text covers the session's prompt (a continuation, a regeneration, or a turn whose
   last reasoning was dropped by the encoding), the request continues that session. If it shares
   less (another conversation with the same system prompt, or an edited earlier turn) the request
   gets a new session id with `base_session_id` set, so the old conversation keeps its state.
4. A request that finds its best session busy gets a new session with that one as base.

With tools present the encoding keeps every turn's reasoning, so an agent loop (assistant tool
call → tool result → assistant ...) re-sends exactly the history that is in the KV state, and each
turn prefills only the new tool results. Without tools, thinking-mode history drops the previous
reasoning, so a turn reuses the prefix up to the previous assistant message.

## Front-end behaviour that rank 0 can rely on

- Prompts start with `<｜begin▁of▁sentence｜>` (id 0) and are at most `max_context - 1` tokens;
  `max_new_tokens` is clamped so that prompt + output fits `max_context`.
- `stop_ids` always holds EOS (1) unless the client asked to ignore it.
- Stop strings, repetition loops and client disconnects end a request with `CANCEL`.

## Testing without the engine

`dsfront mock-engine --socket PATH` is a rank 0 stand-in: it speaks this protocol, keeps sessions
(and reports `cached_tokens` from the LCP), and streams scripted text built from the prompt:
reasoning when the prompt ends in `<think>`, a short answer that quotes the last user message or
tool result, and a DSML tool call when the last user message contains `MOCK_TOOL <name> <json
arguments>`. `MOCK_LOOP` makes it repeat itself (to test the loop guard) and `MOCK_LONG <n>`
streams n filler tokens.

[`engine/common/frontend_echo_engine.cpp`](../engine/common/frontend_echo_engine.cpp) is the same
loop on the C++ side, written against `frontend_protocol.h` alone: it answers each `SUBMIT` with the
last 8 prompt ids reversed plus EOS, which checks the header against the Rust side end to end
(`stop`/`length` reasons, usage counts, streaming) and is a starting point for rank 0's socket
thread.
