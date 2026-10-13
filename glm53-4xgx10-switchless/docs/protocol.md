# Front end and the front-end ↔ rank-0 protocol (GLM-5.3)

The Rust front-end (`frontend/`, crate and binary `glmfront`) is ds41f's `dsfront` re-specialised
for GLM-5.3 (DESIGN decision 7, milestone G6). It owns HTTP (the OpenAI API, SSE with keep-alive),
the chat template, tokenisation, stop strings, the loop guard, output parsing and session matching.
Rank 0 owns everything that touches the model. The server, SSE, sessions and kept sessions, the
gate, the guard and the mock engine are ds41f's, unchanged except where this page says so; the
contract they follow is
[ds41f's protocol.md](../../ds41f-4xgx10-switchless/docs/protocol.md), and this page lists what
differs for this configuration.

Contents: [what changed](#what-changed-from-ds41f) · [the protocol](#the-protocol-version-2) ·
[the template port](#the-chat-template-port) · [output parsing](#output-parsing) ·
[sampling](#sampling-defaults-and-guards) · [tests](#tests-and-how-to-rerun-them) ·
[limits](#what-is-not-matched-and-what-remains)

## What changed from ds41f

| Part | ds41f (`dsfront`) | GLM-5.3 (`glmfront`) |
|---|---|---|
| Tokeniser core | byte-level BPE, merges as `"a b"` strings | the same core; merges read as strings or `["a", "b"]` arrays; `ignore_merges` (a pre-token that is a vocabulary entry is that one id; 893 entries depend on it) |
| Pre-tokeniser | DeepSeek's three chained `Split`s | GLM's one Qwen2/GLM-4 pattern, hand-written leftmost-first (`tokenizer/pretok.rs`); class table L / N / whitespace from Unicode 16.0 (`tools/gen_unicode_tables.py`) |
| Chat encoding | port of DeepSeek's `encoding.py` (DSML) | port of `chat_template.jinja` (`encoding.rs`, below) |
| Thinking | a per-request switch (`thinking`, `enable_thinking`, effort none) | always on: the template's generation prompt is `<|assistant|><think>` and it has no switch; `thinking: false` is ignored, as vLLM ignores it for this template |
| Effort | budget 1–100 in a system preamble | the template's `Reasoning Effort: Max/High/Low` system turn (below) |
| Tools in the prompt | hosted on a leading system message | the template's `tools` variable, the client's tool objects as sent (null values dropped) |
| Tool calls out | DSML blocks | `<tool_call>` blocks with vLLM's argument rules (below) |
| Image, audio, video parts | 400 | rendered as the template's `<reminder>You are unable to process this image …</reminder>` |
| Tool-call arguments in requests | as given | JSON text parsed to a mapping (the template calls `.items()`); not JSON or not an object → 400 |
| Stop ids | EOS (1) | 154820 `<|endoftext|>`, 154827 `<|user|>`, 154829 `<|observation|>` (generation_config.json), plus HELLO's `eos_id` |
| Served model name | `deepseek-v4.1-flash-uncensored` | `glm-5.3` |
| Engine socket | `/tmp/dsgb10-engine.sock` | `/tmp/dsgb10-glm53-engine.sock` |
| Session table file | magic `DSFT0001` | magic `GLMT0001` (a DeepSeek table is not taken) |
| `serve` options | `--effort N`, `--thinking` | `--effort max|high|low`, `--clear-thinking` |

## The protocol (version 2)

The frames, their layouts and their rules are ds41f's version 1 byte for byte (HELLO, SUBMIT,
CANCEL, RELEASE; HELLO, TOKENS, DONE; the same `SubmitHeader` and `DoneHeader`). Version 2 changes
what three fields mean, and the bump keeps a DeepSeek engine and a GLM front-end from talking:

- HELLO's `vocab_size` is the number of ids rank 0 may sample: **154856** (the 154820 vocabulary
  entries and the 36 added tokens). The head has padding rows above that; rank 0 masks them before
  sampling. The front-end refuses an engine that announces more, or 0.
- HELLO's `eos_id` must be one of the three stop ids (rank 0 sends 154820); the front-end refuses
  an engine that announces another.
- Every SUBMIT's `stop_ids` holds 154820, 154827 and 154829 (unless the client asked `ignore_eos`),
  plus any `stop_token_ids` of the request. A sampled stop id is sent as the last token, as before;
  its text is not part of the answer.
- If rank 0 ever sends an id ≥ 154856, the front-end cancels the request, drops and releases its
  session, and answers 500 (a fault, not a token).

Front-end behaviour rank 0 can rely on, besides ds41f's: prompts start with `[gMASK]<sop>`
(154822, 154824) and end with `<|assistant|><think>` (154828, 154841).

Sessions and exact prefix reuse work as in ds41f (text match against each session's prompt plus
output, the session's own ids up to a cut next to an added token, forks through
`base_session_id`). With GLM's template a conversation renders append-only across turns when
`clear_thinking` is false (the default): every past turn re-renders identically, reasoning included.
The stop id the model ended with is itself the next turn's role token (`<|observation|>` before
tool results, `<|user|>` before the next user message), so the session's ids match through it.
What differs between a turn's output and its re-rendering: the template strips the content of past
assistant turns (`content.strip()`), and a tool-call argument whose type the schema changes (`2.0`
for a number parameter comes back as `2`); the reuse then stops at the last added token before the
difference. The mock engine's agent turn reuses 349 of the previous turn's 350 ids (the last
sampled id has no KV yet).

## The chat template port

`encoding::render(messages, RenderOptions { tools, reasoning_effort, clear_thinking,
add_generation_prompt })` gives what transformers' `apply_chat_template` gives for the checkpoint's
`chat_template.jinja` (sandboxed Jinja, `trim_blocks`, `lstrip_blocks`, transformers' `tojson` =
`json.dumps(ensure_ascii=False)`). Messages stay JSON values; Jinja semantics are mirrored where the
template depends on them: undefined attributes render as nothing, `{{ x }}` of a non-string is
Python's `str(x)` (pyjson.rs), truthiness is Python's, `str.strip()` is Python's (it also strips
U+001C–U+001F), variables set in a loop body do not carry into the next iteration (so an assistant
turn without reasoning gets `<think></think>` even after one with it), and the port raises where the
template raises. The branches:

1. Prefix `[gMASK]<sop>`, then `<|system|>Reasoning Effort: ` + `Low`/`High` if `reasoning_effort`
   is exactly `"low"`/`"high"`, else `Max` (undefined, null, `"medium"`, `"Low"`, numbers).
2. `tools` (when truthy): the `# Tools` system turn, one line per tool: the `function` object (or the
   tool itself without one) key by key without `strict` and `defer_loading`, keys unescaped, values
   through tojson; tools with a truthy `defer_loading` left out.
3. Messages in order, no newline after a role token: `system` and `user` render
   `visible_text(content)` (strings; lists of parts: `text` parts, bare strings, media parts as the
   reminder, anything else nothing; null nothing; other values `str()`); unknown roles nothing.
4. `assistant`: the reasoning is `reasoning_content` when it is a string, else (if the content holds
   `</think>`) the text between the last `<think>` before the first `</think>` and it, the content
   becoming what follows the last `</think>`. `<think>R</think>` when there is reasoning and
   (`clear_thinking` is falsy or the turn is after the last user message), else `<think></think>`;
   then the stripped content; then each tool call as
   `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`, `V` raw for strings
   and tojson otherwise (a call without the `function` wrapper is taken as is; arguments that are
   not a mapping and names that are not strings raise).
5. `tool`: a run of consecutive tool messages renders once, as `<|observation|>` and one
   `<tool_response>…</tool_response>` per result. The results are put in the order of the preceding
   assistant's calls when every result (or every entry of a list-of-outputs result) has an id
   (`tool_call_id`, else `id`), no id repeats in the run, each matches a call, and every call has a
   distinct id; otherwise message order. Results can be strings, content parts, lists of
   `{output, id}` entries, or `tool_reference` lists, which expand to the referenced tools' lines in
   `<tool_response><tools>…</tools></tool_response>` (and raise when no tools are given, since the
   template defines that macro only then). A number or mapping as a tool result raises, as in the
   template.
6. `add_generation_prompt`: `<|assistant|><think>`. An empty message list raises
   (`apply_chat_template`'s own check).

The server maps the request onto these variables: `chat_template_kwargs.reasoning_effort` and
`.clear_thinking` go to the template as they are; otherwise OpenAI's `reasoning_effort` (or
`reasoning.effort`) is mapped (none, minimal, low → low; medium, high → high; xhigh, max → max;
anything else the server's `--effort`, default max), and `clear_thinking` is the server's
`--clear-thinking` (default false, the template's). `developer` messages become `system`, the
`reasoning` field becomes `reasoning_content`, and `tool_choice: "none"` sends no tools.

## Output parsing

`encoding::StreamParser`, after vLLM's GLM-4.7 parser (`vllm/parser/glm47_moe.py`, which vLLM also
uses for GLM-5.3, and the schema coercion in `vllm/tool_parsers/utils.py`):

- The output starts in the reasoning (the prompt ends in `<think>`). `</think>` ends it; a
  `<tool_call>` inside the reasoning also ends it and starts a call. A stray `<think>` in the
  reasoning and a stray `</think>` in the content are dropped; `<think>` in the content opens
  reasoning again. Reasoning goes out as `reasoning_content` (ds41f's convention, which pi reads),
  content as `content`, both streamed as soon as they cannot be the start of a marker.
- A `<tool_call>…</tool_call>` block is a call when its name (the text before the first `<arg_key>`,
  stripped) is a tool the request offered. Arguments are every
  `<arg_key>K</arg_key>\s*<arg_value>V</arg_value>` match in order (key stripped, value raw; a key
  seen twice keeps its first place and its last value). Each value stays a string unless the tool's
  `parameters.properties` gives that key a schema: then vLLM's `coerce_to_schema_type` over the
  schema's types (`type`, `enum`, `anyOf`/`oneOf`/`allOf`, type aliases such as `int`), in vLLM's
  priority order null > integer > number > boolean > object > array > string, with Python's `int()`
  and `float()` rules (`" +0_7 "` is 7, a whole float becomes an integer, non-finite values stay
  strings), and a final `json.loads` attempt when no listed type fits. The arguments go out as
  `json.dumps(…, ensure_ascii=False)` text. Each call is streamed whole once its block closes, with
  `finish_reason: "tool_calls"`.
- A block whose name is empty or not an offered tool, a request with no tools, and a block cut off
  by `max_tokens` or a stop string pass through as content text.

## Sampling defaults and guards

- generation_config.json has **no** sampling fields (only `eos_token_id` [154820, 154827, 154829],
  `pad_token_id` 154820 and `bos_token_id` 0; scouting.md's "temperature 1.0, top-p 0.95" are the
  model card's evaluation settings, not that file's). The front-end's defaults are the model card's:
  temperature 1.0, top-p 0.95 (`--temperature`, `--top-p`), top-k off.
- Stop on the three eos ids (SUBMIT `stop_ids`, and the front-end ends the output on any of them).
- Ids ≥ 154856 are masked by rank 0 (HELLO `vocab_size`); one arriving is a fault (above).

## Tests and how to rerun them

All CPU, in `frontend/`. `MODEL` is the checkpoint's snapshot directory (tokenizer.json,
tokenizer_config.json, chat_template.jinja); the Python needs `tokenizers` 0.22 and `transformers`
5.x (run on 2026-10-13 with tokenizers 0.22.2, transformers 5.15.0, jinja2 3.1.6).

```
cargo build --release -j4 && cargo test --release -j4          # 16 unit tests
python3 tools/gen_tokenizer_cases.py $MODEL $CASES               # tok_{vocab,corpus,fuzz,sweep}.jsonl
for s in vocab corpus fuzz sweep; do target/release/glmfront tokcheck $CASES/tok_$s.jsonl --tokenizer $MODEL/tokenizer.json; done
python3 tools/gen_encoding_cases.py $MODEL $CASES/enc_cases.jsonl
target/release/glmfront enccheck $CASES/enc_cases.jsonl --tokenizer $MODEL/tokenizer.json
target/release/glmfront mock-engine --socket $S &
target/release/glmfront serve --listen 127.0.0.1:$PORT --engine-socket $S --api-key-file $KEY 2> $LOG &
python3 tools/smoke_test.py http://127.0.0.1:$PORT $KEY $LOG
```

| Suite | Cases | Result |
|---|---|---|
| tokcheck `vocab`: every vocabulary entry's text alone and in 4 contexts; the 893 `ignore_merges` entries in 12 more | 779,296 | 0 encode, 0 decode mismatches |
| tokcheck `corpus`: Python sources, README, the repo's docs and sources, prose in 12 scripts, every added token inside text (adjacent, partial, nested), whitespace runs to 1000, long runs to 5000 | 956 (5.2 MB) | 0 / 0 |
| tokcheck `fuzz`: random strings over whitespace, digits of many scripts, CJK, marks, emoji, apostrophes and contractions (`'ſ` included), added tokens | 60,000 | 0 / 0 |
| tokcheck `sweep`: every assigned code point (every 97th unassigned) in 5 contexts | 1,476,195 | 0 / 0 |
| one 6.7 MB text; `tokbench`'s parallel encoding against the single pass | 1 | equal |
| enccheck render: targeted cases for every branch above, then 3000 random conversations (system prompts, content parts and media, reasoning by field and by `<think>`, `clear_thinking`, efforts, tool sets with deferred and bare tools, 1–3 calls with every argument shape, results in and out of call order and every reason not to reorder) | 3,063 (63 raise in both; 856k tokens) | 0 mismatches: same text, same ids |
| enccheck parse: completions with reasoning, stray tags, 0–3 calls over 15 schema shapes and 29 values, unknown tools, no tools, cut-off calls; also fed in random 1–9-char pieces to the streaming parser | 3,000 | 0 mismatches |
| smoke_test.py against `glmfront mock-engine` | 29 checks | 0 failures |

Coverage of the random render cases (from the cases file): 1810 with tools, 1947 with tool results,
570 blocks reordered, 1354 sortable and already in order, 1532 not sortable, 773 with a truthy
`clear_thinking` (188 of them clearing a past reasoning), 1150 with media reminders, 436 without the
generation prompt.

## What is not matched, and what remains

- Inside the tested domain nothing in the template differs. Outside it, by construction: error
  messages are not compared (only that both raise); Python's `str()` of a mapping or list given as
  message content uses pyjson's simplified `repr` (strings with both quote kinds may differ);
  Jinja's `==` between numbers of different JSON spellings is approximated; and the `'function' in
  tool` test on a tool that is a string or list (not a mapping) is not reproduced (the template then
  raises in `tool_to_json` anyway).
- The argument coercion takes ASCII digits only; Python's `int()`/`float()` (and so vLLM) also read
  other scripts' decimal digits (`"٣"` → 3), which stay strings here.
- The parse cases' expected values come from vLLM's argument and coercion functions transcribed
  into tools/gen_encoding_cases.py; the reasoning/content split there is a reference written for
  this suite, not vLLM's parser engine run as is (vLLM is not installed on the box).
- Thinking cannot be turned off (the template has no path for it; vLLM ignores the switch too). If
  a client needs a non-thinking answer, appending `</think>` to the prompt is the obvious extension;
  it is not built.
- No engine speaks version 2 yet: rank 0's side (masking ids ≥ 154856, the three stop ids, the HELLO
  values) is G6's engine half. pi against a test port of the real engine is G6's acceptance test.
