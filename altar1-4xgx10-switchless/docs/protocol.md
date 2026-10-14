# Front end and the front-end ↔ rank-0 protocol (altar-1)

The Rust front-end (`frontend/`, crate and binary `altarfront`) is glm53's `glmfront` copied and
specialised for AikidoSec/altar-1 (revision 5d591cd98958e4e7e429517887e80ca3bb7ce2c4), which is
GLM-5.3 pruned to 168 of 256 routed experts. This page is a delta: everything not named here is
[glm53's protocol.md](../../glm53-4xgx10-switchless/docs/protocol.md) unchanged (and, through it,
[ds41f's](../../ds41f-4xgx10-switchless/docs/protocol.md)). Written 2026-10-14.

Contents: [identical](#what-is-identical) · [what changed](#what-changed-from-glmfront) ·
[the template](#the-template-five-lines-differ) · [null contents in the server](#null-contents-in-the-server) ·
[tests](#tests-and-how-to-rerun-them) · [limits](#what-is-not-matched)

## What is identical

Checked on 2026-10-14 against the GLM-5.3 files the glm53 port is tested on (the NVIDIA GLM-5.3-NVFP4
snapshot e3b8f9f2fad29727dfa95e766c5cac18ea1414bf):

| File | altar-1 | GLM-5.3 | |
|---|---|---|---|
| tokenizer.json | sha256 `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` | the same | byte-identical (`cmp`) |
| tokenizer_config.json | sha256 `77af7d4769cd62c107b90495cac9b0ba81573c86486821bfba2980c04285ec7a` | the same | byte-identical |
| chat_template.jinja | sha256 `69bb3ab52067898e2466b855407636de559568947f367945842aabcb7fcc1705` | sha256 `3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c` | five lines differ (below) |
| generation_config.json | eos [154820, 154827, 154829], pad 154820, `do_sample` true, temperature 1.0, top_p 0.95 | eos and pad the same; no sampling fields, `bos_token_id` 0 | stop ids the same |

So, unchanged from glmfront:

- **The protocol**: version 2, the same frames and meanings: HELLO's `vocab_size` is the 154856
  sampleable ids (ids ≥ 154856 are masked padding rows), `eos_id` one of the three stop ids, every
  SUBMIT's `stop_ids` holds 154820, 154827, 154829. No bump: the vocabulary and stop ids are the
  same, and the default socket (below) keeps an altar-1 and a GLM-5.3 engine apart.
- **The tokeniser** (`tokenizer/`) and its default path: altar-1's tokenizer.json is the GLM-5.3 file
  byte for byte, so `--tokenizer` still defaults to the GLM-5.3 snapshot's copy.
- **Output parsing** (vLLM's GLM-4.7 rules), the server, SSE, sessions and kept sessions, the gate,
  the guard, the mock engine, the request mapping (efforts, `developer` → `system`, `reasoning` →
  `reasoning_content`), the thinking behaviour (always on).
- **Sampling defaults**: temperature 1.0, top-p 0.95, top-k off. glm53 took them from the model card;
  altar-1's generation_config.json now states the same two values.

## What changed from glmfront

| Part | GLM-5.3 (`glmfront`) | altar-1 (`altarfront`) |
|---|---|---|
| Crate, binary | `glmfront` | `altarfront` |
| Served model name (`--model-name`) | `glm-5.3` | `altar-1` |
| Engine socket (`--engine-socket`, mock `--socket`) | `/tmp/dsgb10-glm53-engine.sock` | `/tmp/dsgb10-altar1-engine.sock` |
| Session table file | magic `GLMT0001` | magic `ALTR0001` (a GLM-5.3 table is not taken: its kept sessions live in another model's engine) |
| Template port (`encoding.rs`) | GLM-5.3's template | altar-1's: a null content renders `None`; the tool-result sortability check without `break`s |
| Message mapping (`server.rs`) | a missing or null `content` goes to the template as null | goes as `""` (below) |
| `enccheck` | | also counts the cases GLM-5.3's template renders differently (`glm53_differs`) |

## The template: five lines differ

`diff` of GLM-5.3's chat_template.jinja (`<`) against altar-1's (`>`), line numbers GLM-5.3's:

```
65c65
<     {%- elif content is not none -%}
---
>     {%- else -%}
180d179
<             {%- if not ns_chk.can_sort -%}{%- break -%}{%- endif -%}
184d182
<                     {%- if not ns_chk.can_sort -%}{%- break -%}{%- endif -%}
206d203
<             {%- if not ns_chk.can_sort -%}{%- break -%}{%- endif -%}
214d210
<                     {%- break -%}
```

**1. `visible_text`'s last branch.** altar-1's `{%- else -%}{{- content }}` renders a null content as
Jinja renders Python's `None`: the text `None`. GLM-5.3's renders nothing. A *missing* `content` is
Jinja's undefined and renders nothing in both. Every place that goes through `visible_text` is
affected: user and system turns (`<|user|>None`), assistant turns (`<think></think>None`, or
`<think>R</think>None<tool_call>…` for the OpenAI tool-call shape `content: null`), tool results
(`<tool_response>None</tool_response>`, sorted or not) and a list-of-outputs entry whose `output` is
null. The port mirrors it (`visible_text`: only undefined is empty, a null is `py_str` = `None`).

**2. No `break` in the sortability check.** A run of tool results is put in the calls' order only
when every result has a distinct id matching a call and every call has a distinct id. GLM-5.3's
template leaves the check at the first failure; altar-1's goes on over every result, every
list-of-outputs entry and every call. `can_sort` only ever goes from true to false and the checks
write nothing, so the text is the same and whether the template raises is the same; what can differ
is *which* error a malformed block raises first (a check of a later result reaches a bad content
before the render does). Evidence: `tools/check_template_breaks.py` renders 50,000 random tool-result
blocks with altar-1's template and with GLM-5.3's template given altar-1's line 65 (so that only the
breaks differ): 22,653 same text, 26,862 raise with the same message, 485 raise with different
messages, 0 with different text, 0 where one raises and the other does not (run on 2026-10-14). The
port mirrors altar-1 (`render_tool_block` has no early exit); `enccheck` compares errors only as
"both raise", as in glm53, so the error-message difference is not tested.

The cases that pin this (`altar_delta` in `tools/gen_encoding_cases.py`, each with tools and again
without tools and with `clear_thinking`, 44 cases):

| Cases | What | GLM-5.3's template |
|---|---|---|
| `null: …` (11 × 2) | null content in a user turn, a system turn first and mid-conversation, an assistant turn without calls, with reasoning, with tool calls; a tool result in a sorted block, an unsorted block, first in the conversation; a null list-of-outputs `output`; every content of an agent turn null | all 22 render differently |
| `missing: …`, `empty: …` (3 × 2) | content absent (undefined) for user, assistant and tool turns; `""` with tool calls | the same: nothing renders |
| `no break: …` (8 × 2) | a first result without id or with an unknown id followed by more results, by list-of-outputs entries, by a result that raises (a number), by a tool reference (raising without tools); a call without id followed by repeated call ids; repeated call ids followed by more calls; a sortable block for contrast | the same text, or both raise (5 cases) |

Across the whole render suite, 1,138 of the 3,107 cases render differently under GLM-5.3's template
(22 of the delta cases, 3 of glm53's targeted cases that already had a null, 1,113 of the 3,000
random conversations, whose generator draws null contents for every role), and none of the 68
that raise. glm53's `glmfront` (its GLM-5.3 port) run on the altar-1 cases gives exactly those 1,138
mismatches, so the suite tells the two templates apart and `altarfront` follows altar-1's.

## Null contents in the server

A null content reaching altar-1's template writes `None` into the prompt, and OpenAI clients send
`content: null` on every tool-call turn (`altarfront`'s own answers have it when a turn is only tool
calls). The server therefore maps a missing or null `content` to `""` before rendering. vLLM's
chat_utils does the same in effect (a None content becomes an empty part list, rendered as `""`, so a
vLLM-served altar-1 never shows the template a null; read from its source, not run: vLLM is not
installed on the box). With GLM-5.3's template null and `""` render alike, so for every request
`altarfront` sends the prompt `glmfront` sends. `render()` itself stays byte-exact to the template,
`None` included; only the server's mapping keeps nulls away from it. Pinned by the unit test
`server::tests::null_content_is_empty` (null, missing and `""` give one prompt; `render` of a null
user content gives `<|user|>None`) and the smoke check "null, missing and empty contents give the
same prompt" (163 prompt tokens each against the mock; a `None` would add tokens).

## Tests and how to rerun them

All CPU, in `frontend/`, run on 2026-10-14 with tokenizers 0.22.2, transformers 5.15.0, jinja2
3.1.6. `MODEL` is a directory holding altar-1's chat_template.jinja, tokenizer.json,
tokenizer_config.json (and README.md, which the tokeniser corpus reads), downloaded at the revision
above; `GLM53_TEMPLATE` is GLM-5.3's chat_template.jinja (the glm53 checkpoint's).

```
nice -n 19 cargo build --release -j4 && nice -n 19 cargo test --release -j4      # 17 unit tests
python3 tools/gen_tokenizer_cases.py $MODEL $CASES                                 # tok_{vocab,corpus,fuzz,sweep}.jsonl
for s in vocab corpus fuzz sweep; do target/release/altarfront tokcheck $CASES/tok_$s.jsonl --tokenizer $MODEL/tokenizer.json; done
python3 tools/gen_encoding_cases.py $MODEL $CASES/enc_cases.jsonl --glm53-template $GLM53_TEMPLATE
target/release/altarfront enccheck $CASES/enc_cases.jsonl --tokenizer $MODEL/tokenizer.json
python3 tools/check_template_breaks.py $MODEL $GLM53_TEMPLATE 50000
target/release/altarfront mock-engine --socket $S &
target/release/altarfront serve --listen 127.0.0.1:$PORT --engine-socket $S --api-key-file $KEY 2> $LOG &
python3 tools/smoke_test.py http://127.0.0.1:$PORT $KEY $LOG
```

| Suite | Cases | Result |
|---|---|---|
| unit tests (glm53's 16, plus `null_content_is_empty`) | 17 | 17 pass |
| tokcheck `vocab` (every entry alone and in 4 contexts; the 893 `ignore_merges` entries in 12 more) | 779,296 | 0 encode, 0 decode mismatches |
| tokcheck `corpus` (glm53's corpus; the repo-doc part is this configuration's docs) | 953 (5.0 MB) | 0 / 0 |
| tokcheck `fuzz` | 60,000 | 0 / 0 |
| tokcheck `sweep` (every assigned code point, every 97th unassigned, 5 contexts) | 1,476,195 | 0 / 0 |
| enccheck render: glm53's targeted cases, the 44 altar-1 delta cases, 3,000 random conversations (the same ones as glm53's suite: same generator and seed, and the added cases draw no random numbers) | 3,107 (68 raise in both; 869,863 tokens; 1,138 differ under GLM-5.3's template) | 0 mismatches: same text, same ids |
| enccheck parse (unchanged from glm53) | 3,000 | 0 mismatches |
| negative control: glm53's `glmfront enccheck` on the same file | 3,107 render | 1,138 mismatches, the `glm53_differs` cases |
| check_template_breaks.py | 50,000 blocks | 0 text differences, 0 raise-or-not differences |
| smoke_test.py against `altarfront mock-engine` (glm53's 29 checks, plus the null-content one) | 30 checks | 0 failures |

The tokeniser suites are glm53's run again on altar-1's tokenizer.json; being the same file, they
could not have come out otherwise, and are kept as the configuration's own record.

## What is not matched

- Everything glm53's page lists under "What is not matched" holds here unchanged (error messages
  not compared, pyjson's simplified `repr`, Jinja `==` between number spellings, the `'function' in
  tool` test on strings and lists, ASCII-only digits in argument coercion, the transcribed vLLM parse
  reference, thinking always on).
- The server's null → `""` mapping departs from what transformers' `apply_chat_template` renders for
  a null content (`None`), by design (above). A client that wants `None` in the prompt cannot get it
  through the server; `render()` gives it.
- The template diff was taken against the GLM-5.3 template the glm53 port uses (the NVIDIA
  snapshot above); zai-org's own repository was not fetched for this page.
- No altar-1 engine speaks the protocol yet; HELLO checks only the version, vocabulary size and eos
  id, which an altar-1 and a GLM-5.3 engine share, so the separate socket is what keeps them apart.
