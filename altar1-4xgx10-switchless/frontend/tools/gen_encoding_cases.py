#!/usr/bin/env python3
"""Expected prompts (from altar-1's chat_template.jinja) and parses (vLLM's GLM rules) for the Rust
port, encoding.rs. Needs transformers 5.x and tokenizers; CPU only:

    gen_encoding_cases.py MODEL_DIR OUT_FILE [N_RANDOM] [--glm53-template PATH]

MODEL_DIR holds altar-1's chat_template.jinja, tokenizer.json and tokenizer_config.json (AikidoSec/altar-1
at 5d591cd98958e4e7e429517887e80ca3bb7ce2c4). With --glm53-template (GLM-5.3's chat_template.jinja), every
render case is rendered with that template too, and the ones whose outcome differs (text, or raising or
not) are marked "glm53_differs": the cases that tell the two templates apart.

Writes one case per line, checked with `altarfront enccheck OUT_FILE`:
  {"kind": "render", "name", "messages", "tools"?, "kwargs", "add_generation_prompt", "text" and "ids" | "error",
   "glm53_differs"?}
      rendered by transformers' apply_chat_template (its sandboxed Jinja, trim_blocks, lstrip_blocks,
      its tojson), ids by the `tokenizers` library with add_special_tokens=False
  {"kind": "parse", "name", "text", "thinking", "tools"?, "result": {"reasoning", "content", "tool_calls": [[name, args]]}}
      the tool-call arguments as vLLM's GLM-4.7 parser gives them (vllm/parser/glm47_moe.py, the schema
      coercion in vllm/tool_parsers/utils.py; transcribed below, Apache-2.0, Copyright contributors to the
      vLLM project)
The render suite: targeted cases for every branch of the template (effort, clear_thinking, tools with and
without the function wrapper, defer_loading and strict, content parts and media, reasoning from the field or
from <think> in the content, tool calls with every argument shape, tool results reordered by id and every
reason not to reorder, lists of outputs, tool references, mid-conversation system turns, the errors the
template raises), the cases altar-1's template renders differently from GLM-5.3's (null contents) and
the ones that run its break-less sortability check past the first reason not to sort, then random
conversations mixing them.
"""
import copy
import json
import math
import random
import re
import sys

from tokenizers import Tokenizer
from transformers import AutoTokenizer

ARGS = sys.argv[1:]
GLM53_TEMPLATE = None
if "--glm53-template" in ARGS:
    k = ARGS.index("--glm53-template")
    GLM53_TEMPLATE = open(ARGS[k + 1]).read()
    del ARGS[k:k + 2]
MODEL, OUT = ARGS[0], ARGS[1]
N_RANDOM = int(ARGS[2]) if len(ARGS) > 2 else 3000
hf = AutoTokenizer.from_pretrained(MODEL)
with open(MODEL + "/chat_template.jinja") as fh:
    assert hf.chat_template == fh.read(), "the tokenizer's chat template is not MODEL_DIR/chat_template.jinja"
tk = Tokenizer.from_file(MODEL + "/tokenizer.json")
R = random.Random(20261013)

TEXTS = [
    "Hello", "What is 2+2?", "", " ", " leading and trailing  ", "人工智能正在改变世界。你好！", "\x1c\x1dfile sep\x1f ",
    "emoji 👍🏽 and ünïcödé, Ελληνικά, русский", "line1\nline2\n\nline4\n", "tabs\tand \"quotes\" and \\backslash",
    "a" * 300, "code:\n```python\nprint('hi')\n```\n", "{\"json\": [1, 2.5, null]}", "\n\nleading newlines",
    "Reasoning Effort: Low", "x\r\ny", "<tool_call>fake</tool_call>", "It's 3.14159 o'clock", "＜全角＞", "　ideographic space",
    "zero​width", "<|user|> inside text", "<think>not a tag", "</think>", "<arg_key>k</arg_key>", " nbsp ",
]

TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "number", "default": 2.5}},
                    "required": ["command"]}}},
    {"type": "function", "function": {"name": "read", "description": "Read a file. Use offset/limit for big files — ünïcode “quotes”.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer", "minimum": 1},
                                                     "limit": {"type": "integer", "maximum": 1000000000000000000000}},
                    "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "edit", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "edits": {"type": "array", "items": {"type": "object", "properties": {
            "old": {"type": "string"}, "new": {"type": "string"}}}}, "mode": {"enum": ["a", "b", None], "default": None}}},
        "description": "Edit", "strict": True}},
    {"type": "function", "function": {"name": "noargs", "description": "", "parameters": {}}},
    {"type": "function", "function": {"name": "weird\"name", "description": "line\nbreak\ttab \\ slash </tools>",
                                      "parameters": {"type": "object", "properties": {"x": {"type": "number", "default": 1e-07},
                                                                                      "y": {"type": "number", "default": 1e+16},
                                                                                      "z": {"type": "number", "default": -0.0}}}}},
    {"type": "function", "function": {"name": "deferred", "description": "loaded later", "defer_loading": True,
                                      "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
    {"name": "bare", "description": "a tool without the function wrapper", "parameters": {"type": "object", "properties": {"flag": {"type": "boolean"}}}},
    {"name": "bare_deferred", "defer_loading": False, "strict": False, "x\"key": [1, 2.0, "三"]},
]

ARG_VALUES = ["ls -la", "", "多语言", "line\nbreak", "quote\"s", "123", "true", "null", 0, -1, 7, 2.5, 1e-07, 1e16, 0.1,
              12345678901234567890, True, False, None, [1, "two", 3.0], {"nested": {"k": [None, True]}}, [], {},
              "</arg_value>", "a < b > c", "  spaced  "]


def text():
    return R.choice(TEXTS)


def rand_args():
    n = R.choice([0, 1, 1, 2, 3])
    keys = R.sample(["command", "path", "timeout", "x", "query", "名字", "with space", "a\"b", "flag"], n)
    return {k: R.choice(ARG_VALUES) for k in keys}


def rand_content():
    r = R.random()
    if r < 0.6:
        return text()
    if r < 0.7:
        return None
    parts = []
    for _ in range(R.choice([1, 2, 3])):
        q = R.random()
        if q < 0.5:
            parts.append({"type": "text", "text": text()})
        elif q < 0.6:
            parts.append(text())
        elif q < 0.75:
            t = R.choice(["image", "image_url", "video", "video_url", "audio", "audio_url", "input_audio"])
            parts.append({"type": t, t: {"url": "x"}})
        elif q < 0.85:
            parts.append({"type": "text", "text": R.choice([None, 5, 2.5, True])})
        elif q < 0.9:
            parts.append({"type": "text"})
        else:
            parts.append({"type": "file", "file": "x"})
    return parts


def rand_tool_call(i, ids=True):
    name = R.choice(["bash", "read", "edit", "noargs", "bare", "名前"])
    tc = {"type": "function", "function": {"name": name, "arguments": rand_args()}}
    if ids:
        tc["id"] = f"call_{i}"
    if R.random() < 0.1:
        tc = {"id": tc.get("id"), "name": name, "arguments": tc["function"]["arguments"]}  # no function wrapper
    return tc


def rand_assistant():
    m = {"role": "assistant"}
    r = R.random()
    c = rand_content() if R.random() < 0.8 else text()
    if r < 0.4:
        m["reasoning_content"] = R.choice(["Let me think.", "", "多步推理\n\n1. a\n2. b", "  spaced  "])
    elif r < 0.55 and isinstance(c, str):
        c = R.choice(["<think>R1</think>", "pre<think>R2</think>post", "R3</think>", "a</think>b</think>c", "<think>x<think>y</think> z"]) + c
    elif r < 0.6:
        m["reasoning_content"] = R.choice([None, 5])
    m["content"] = c
    return m


def rand_tool_block(calls):
    """Tool results for `calls`, in some order, sometimes with every reason not to reorder."""
    msgs = []
    order = list(range(len(calls)))
    R.shuffle(order)
    for k in order:
        tc = calls[k]
        m = {"role": "tool", "content": rand_tool_content()}
        idk = R.choice(["tool_call_id"] * 6 + ["id", None])
        if idk and tc.get("id"):
            m[idk] = tc["id"]
        msgs.append(m)
    r = R.random()
    if r < 0.08 and msgs:
        msgs[0]["tool_call_id"] = "call_unknown"
    elif r < 0.14 and len(msgs) > 1:
        msgs[1]["tool_call_id"] = msgs[0].get("tool_call_id", "dup")
        msgs[0]["tool_call_id"] = msgs[1]["tool_call_id"]
    elif r < 0.2:
        msgs.append({"role": "tool", "content": "extra"})
    return msgs


def rand_tool_content():
    r = R.random()
    if r < 0.7:
        return text()
    if r < 0.8:
        return [{"type": "text", "text": text()}, {"type": "image_url", "image_url": {"url": "x"}}]
    if r < 0.85:
        return None
    if r < 0.86:
        return 42  # the template raises on a number
    return [text(), {"type": "text", "text": "part"}]


def rand_conversation():
    msgs = []
    if R.random() < 0.6:
        msgs.append({"role": R.choice(["system"] * 4 + ["developer"]), "content": rand_content()})
    n_turns = R.choice([1, 1, 2, 3, 4])
    cid = 0
    for t in range(n_turns):
        msgs.append({"role": "user", "content": rand_content()})
        if R.random() < 0.1:
            msgs.append({"role": "system", "content": text()})
        for step in range(R.choice([0, 1, 1, 2])):
            a = rand_assistant()
            if R.random() < 0.6:
                ids = R.random() < 0.9
                calls = [rand_tool_call(cid + i, ids) for i in range(R.choice([1, 1, 2, 3]))]
                cid += len(calls)
                if R.random() < 0.05 and len(calls) > 1 and ids:
                    calls[1]["id"] = calls[0]["id"]
                a["tool_calls"] = calls
                msgs.append(a)
                msgs.extend(rand_tool_block(calls))
            else:
                if R.random() < 0.1:
                    a["tool_calls"] = []
                msgs.append(a)
                break
    if R.random() < 0.15 and msgs[-1]["role"] == "user":
        msgs.append(rand_assistant())
    return msgs


def rand_tools():
    r = R.random()
    if r < 0.35:
        return None
    if r < 0.4:
        return []
    return R.sample(TOOLS, R.choice([1, 2, 3, len(TOOLS)]))


def rand_kwargs():
    kw = {}
    if R.random() < 0.6:
        kw["reasoning_effort"] = R.choice(["low", "high", "max", "medium", "Low", None, 5, "xhigh"])
    if R.random() < 0.5:
        kw["clear_thinking"] = R.choice([True, False, None, "yes", 0, 1])
    return kw


def render_case(name, messages, tools=None, kwargs=None, agp=True):
    kwargs = kwargs or {}
    case = {"kind": "render", "name": name, "messages": messages, "kwargs": kwargs, "add_generation_prompt": agp}
    if tools is not None:
        case["tools"] = tools
    try:
        t = hf.apply_chat_template(copy.deepcopy(messages), tools=copy.deepcopy(tools), add_generation_prompt=agp,
                                   tokenize=False, **kwargs)
        case["text"] = t
        case["ids"] = tk.encode(t, add_special_tokens=False).ids
    except Exception as e:
        case["error"] = f"{type(e).__name__}: {e}"
    if GLM53_TEMPLATE is not None:
        try:
            g = hf.apply_chat_template(copy.deepcopy(messages), tools=copy.deepcopy(tools), add_generation_prompt=agp,
                                       tokenize=False, chat_template=GLM53_TEMPLATE, **kwargs)
        except Exception:
            g = None
        if g != case.get("text"):
            case["glm53_differs"] = True
    return case


def targeted():
    out = []
    u = {"role": "user", "content": "hi"}
    out.append(render_case("minimal", [u]))
    out.append(render_case("minimal no generation prompt", [u], agp=False))
    for e in ["low", "high", "max", "medium", "LOW", None, 3, ""]:
        out.append(render_case(f"effort {e!r}", [u], kwargs={"reasoning_effort": e}))
    out.append(render_case("tools", [u], tools=TOOLS))
    out.append(render_case("tools empty", [u], tools=[]))
    out.append(render_case("only deferred tools", [u], tools=[TOOLS[5]]))
    out.append(render_case("no messages", [], tools=TOOLS[:1]))
    conv = [{"role": "system", "content": "sys"}, u,
            {"role": "assistant", "content": "  Let me check.  ", "reasoning_content": "Need tool.",
             "tool_calls": [{"id": "a", "type": "function", "function": {"name": "read", "arguments": {"path": "x", "offset": 2}}},
                            {"id": "b", "type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}]},
            {"role": "tool", "tool_call_id": "b", "content": "ls out"},
            {"role": "tool", "tool_call_id": "a", "content": "file"},
            {"role": "assistant", "content": "<think>inline</think> Done."},
            {"role": "user", "content": "Thanks"}]
    for ct in [None, True, False]:
        kw = {} if ct is None else {"clear_thinking": ct}
        out.append(render_case(f"agent turn, clear_thinking {ct}", conv, tools=TOOLS[:2], kwargs=kw))
        out.append(render_case(f"agent turn ending in a tool result, clear_thinking {ct}", conv[:5], tools=TOOLS[:2], kwargs=kw))
    # every reason not to reorder
    base = conv[:3]
    for name, block in [
        ("in call order", [{"role": "tool", "tool_call_id": "a", "content": "1"}, {"role": "tool", "tool_call_id": "b", "content": "2"}]),
        ("one without id", [{"role": "tool", "tool_call_id": "b", "content": "2"}, {"role": "tool", "content": "1"}]),
        ("by id field", [{"role": "tool", "id": "b", "content": "2"}, {"role": "tool", "id": "a", "content": "1"}]),
        ("unknown id", [{"role": "tool", "tool_call_id": "b", "content": "2"}, {"role": "tool", "tool_call_id": "z", "content": "1"}]),
        ("duplicate id", [{"role": "tool", "tool_call_id": "b", "content": "2"}, {"role": "tool", "tool_call_id": "b", "content": "1"}]),
        ("fewer results", [{"role": "tool", "tool_call_id": "b", "content": "2"}]),
        ("int ids", [{"role": "tool", "tool_call_id": 7, "content": "2"}]),
        ("list of outputs", [{"role": "tool", "content": [{"id": "b", "output": "2"}, {"id": "a", "output": [{"type": "text", "text": "1"}]}]}]),
        ("list of outputs, no ids", [{"role": "tool", "content": [{"output": "2"}, {"output": None}]}]),
        ("list of outputs, mixed", [{"role": "tool", "content": [{"tool_call_id": "b", "output": "2"}]}, {"role": "tool", "tool_call_id": "a", "content": "1"}]),
        ("tool reference", [{"role": "tool", "tool_call_id": "b", "content": [{"type": "tool_reference", "name": "bash"}, {"type": "tool_reference", "name": "read"}]},
                            {"role": "tool", "tool_call_id": "a", "content": "1"}]),
        ("tool reference in outputs", [{"role": "tool", "content": [{"id": "b", "output": [{"type": "tool_reference", "name": "deferred"}]}, {"id": "a", "output": "1"}]}]),
        ("tool reference to nothing", [{"role": "tool", "content": [{"type": "tool_reference", "name": "missing"}]}]),
    ]:
        out.append(render_case(f"results {name}", base + block, tools=TOOLS))
        out.append(render_case(f"results {name}, no tools", base + block))
    dup_calls = copy.deepcopy(base)
    dup_calls[2]["tool_calls"][1]["id"] = "a"
    out.append(render_case("calls with one id twice", dup_calls + [{"role": "tool", "tool_call_id": "a", "content": "1"}], tools=TOOLS))
    out.append(render_case("tool first", [{"role": "tool", "content": "orphan"}, u]))
    out.append(render_case("tool after user", [u, {"role": "tool", "tool_call_id": "a", "content": "x"}]))
    out.append(render_case("assistant with empty tool_calls", [u, {"role": "assistant", "content": "x", "tool_calls": []}, u]))
    out.append(render_case("assistant null content", [u, {"role": "assistant", "content": None, "tool_calls": base[2]["tool_calls"]}]))
    out.append(render_case("arguments as a JSON string (raises)", [u, {"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"x\"}"}}]}]))
    out.append(render_case("name not a string (raises)", [u, {"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": 5, "arguments": {}}}]}]))
    out.append(render_case("content is a number", [{"role": "user", "content": 2.5}, {"role": "assistant", "content": 7}]))
    out.append(render_case("content is a dict", [{"role": "user", "content": {"a": "b"}}]))
    out.append(render_case("unknown role", [{"role": "robot", "content": "beep"}, u]))
    out.append(render_case("reasoning not a string", [u, {"role": "assistant", "content": "a</think>b", "reasoning_content": None}]))
    out.append(render_case("whitespace content", [u, {"role": "assistant", "content": " \n\t\x1c　"}]))
    for a in ["<think>R</think>", "R</think>", "</think>", "x</think>y</think>z", "<think>a<think>b</think>c"]:
        out.append(render_case(f"think in content {a!r}", [u, {"role": "assistant", "content": a}, u]))
    return out + altar_delta(conv)


def altar_delta(conv):
    """altar-1's template against GLM-5.3's. (1) visible_text's last branch is `else` (GLM-5.3: `elif content
    is not none`): a null content renders as "None" in every place visible_text reaches; a missing content
    (undefined) still renders nothing. (2) The sortability check of a tool-result block has no `break`s:
    every result and call is looked at after the first reason not to sort; the text and whether it raises
    stay the same (only which error a malformed block raises first can differ, and errors are compared
    only as raising), so these cases pin that the port keeps both."""
    out = []
    u = {"role": "user", "content": "hi"}
    calls = conv[2]["tool_calls"]  # ids a (read), b (bash)
    asst = {"role": "assistant", "content": None, "tool_calls": calls}
    null = [
        ("null: user content", [{"role": "user", "content": None}]),
        ("null: system content", [{"role": "system", "content": None}, u]),
        ("null: system content mid-conversation", [u, {"role": "assistant", "content": "ok"}, {"role": "system", "content": None}, u]),
        ("null: assistant content, no tool calls", [u, {"role": "assistant", "content": None}, u]),
        ("null: assistant content with reasoning", [u, {"role": "assistant", "content": None, "reasoning_content": "R"}, u]),
        ("null: assistant content with tool calls (the OpenAI shape)", [u, asst]),
        ("null: tool result content, sorted block", [u, asst, {"role": "tool", "tool_call_id": "b", "content": None},
                                                       {"role": "tool", "tool_call_id": "a", "content": "file"}]),
        ("null: tool result content, unsorted block", [u, asst, {"role": "tool", "content": None}]),
        ("null: tool result first in the conversation", [{"role": "tool", "content": None}, u]),
        ("null: list-of-outputs output", [u, asst, {"role": "tool", "content": [{"id": "b", "output": None}, {"id": "a", "output": "1"}]}]),
        ("null: every content of an agent turn", [{"role": "system", "content": None}, {"role": "user", "content": None}, asst,
                                                  {"role": "tool", "tool_call_id": "a", "content": None},
                                                  {"role": "tool", "tool_call_id": "b", "content": None}, {"role": "user", "content": None}]),
        ("missing: user content (undefined: nothing in both)", [{"role": "user"}]),
        ("missing: assistant and tool content (undefined: nothing in both)",
         [u, {"role": "assistant", "tool_calls": calls}, {"role": "tool", "tool_call_id": "a"}, {"role": "tool", "tool_call_id": "b"}]),
        ("empty: assistant content '' with tool calls", [u, {"role": "assistant", "content": "", "tool_calls": calls}]),
    ]
    for name, msgs in null:
        out.append(render_case(name, msgs, tools=TOOLS))
        out.append(render_case(name + ", no tools, clear_thinking", msgs, kwargs={"clear_thinking": True}))
    # (2) every place GLM-5.3's template breaks out of the check, followed by more that the check looks at
    asst = dict(asst, content="c")
    bad = {"role": "tool", "tool_call_id": "b", "content": 42}  # raises when the check or the render reaches it
    br = [
        ("no break: unsortable result first, then results", [{"role": "tool", "content": "noid"},
                                                             {"role": "tool", "tool_call_id": "b", "content": "2"},
                                                             {"role": "tool", "tool_call_id": "a", "content": "1"}], calls),
        ("no break: unsortable result first, then a raising one", [{"role": "tool", "content": "noid"}, bad], calls),
        ("no break: unsortable list-of-outputs entry first, then entries", [{"role": "tool", "content": [
            {"id": "z", "output": "?"}, {"id": "b", "output": "2"}, {"output": "noid"}, {"id": "a", "output": "1"}]}], calls),
        ("no break: unsortable entry, then a raising message", [{"role": "tool", "content": [{"id": "z", "output": "?"}]}, bad], calls),
        ("no break: duplicate result ids, then a tool reference", [{"role": "tool", "tool_call_id": "a", "content": "1"},
                                                                  {"role": "tool", "tool_call_id": "a", "content": "1b"},
                                                                  {"role": "tool", "tool_call_id": "b", "content": [{"type": "tool_reference", "name": "bash"}]}], calls),
        ("no break: a call without id, then duplicate call ids", [{"role": "tool", "tool_call_id": "b", "content": "2"},
                                                                 {"role": "tool", "tool_call_id": "a", "content": "1"}],
         [{"type": "function", "function": {"name": "read", "arguments": {}}}, calls[1], calls[1], calls[0]]),
        ("no break: duplicate call ids, then more calls", [{"role": "tool", "tool_call_id": "a", "content": "1"},
                                                          {"role": "tool", "tool_call_id": "b", "content": "2"}],
         [calls[0], calls[0], calls[1], dict(calls[1], id="")]),
        ("no break: the sortable block for contrast", [{"role": "tool", "tool_call_id": "b", "content": "2"},
                                                       {"role": "tool", "tool_call_id": "a", "content": "1"}], calls),
    ]
    for name, block, cs in br:
        msgs = [u, dict(asst, tool_calls=cs)] + block
        out.append(render_case(name, msgs, tools=TOOLS))
        out.append(render_case(name + ", no tools", msgs))
    return out


# ---------------------------------------------------------------- vLLM's parse rules (transcribed)

_ARG_RE = re.compile(r"<arg_key>(?P<key>.*?)</arg_key>\s*<arg_value>(?P<value>.*?)</arg_value>", re.DOTALL)

_TYPE_ALIASES = {"str": "string", "text": "string", "varchar": "string", "char": "string", "enum": "string", "int": "integer",
                 "int32": "integer", "int64": "integer", "uint": "integer", "uint32": "integer", "uint64": "integer", "long": "integer",
                 "short": "integer", "unsigned": "integer", "float": "number", "float32": "number", "float64": "number",
                 "double": "number", "bool": "boolean", "dict": "object", "arr": "array", "list": "array", "sequence": "array"}


def _is_json_finite(obj):
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, list):
        return all(_is_json_finite(x) for x in obj)
    if isinstance(obj, dict):
        return all(_is_json_finite(x) for x in obj.values())
    return True


def extract_types_from_schema(schema):
    if schema is None or not isinstance(schema, dict):
        return ["string"]
    types = set()
    if "type" in schema:
        tv = schema["type"]
        if isinstance(tv, str):
            types.add(tv)
        elif isinstance(tv, list):
            types.update(t for t in tv if isinstance(t, str))
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        for v in schema["enum"]:
            if v is None:
                types.add("null")
            elif isinstance(v, bool):
                types.add("boolean")
            elif isinstance(v, int):
                types.add("integer")
            elif isinstance(v, float):
                types.add("number")
            elif isinstance(v, str):
                types.add("string")
            elif isinstance(v, list):
                types.add("array")
            elif isinstance(v, dict):
                types.add("object")
    for f in ("anyOf", "oneOf", "allOf"):
        if f in schema and isinstance(schema[f], list):
            for c in schema[f]:
                types.update(extract_types_from_schema(c))
    return list(types) if types else ["string"]


def coerce_to_schema_type(value, schema_type):
    if isinstance(schema_type, str):
        schema_type = [schema_type]
    norm = {_TYPE_ALIASES.get(k, k) for t in schema_type for k in [t.strip().lower()]}
    for c in ["null", "integer", "number", "boolean", "object", "array", "string"]:
        if c not in norm:
            continue
        if c == "null":
            if value.lower() == "null":
                return None
            continue
        if c == "string":
            return value
        if c == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
        if c == "number":
            try:
                val = float(value)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(val):
                continue
            return val if val != int(val) else int(val)
        if c == "boolean":
            lv = value.lower().strip()
            if lv in ("true", "1"):
                return True
            if lv in ("false", "0"):
                return False
            continue
        if c in ("object", "array"):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if _is_json_finite(parsed):
                return parsed
            continue
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
    if not _is_json_finite(parsed):
        return value
    return parsed


def tool_properties(tools, name):
    for t in tools or []:
        f = t.get("function", t)
        if f.get("name") == name:
            return (f.get("parameters") or {}).get("properties", {})
    return None


def parse_call(inner, tools):
    p = inner.find("<arg_key>")
    name, raw = (inner[:p], inner[p:]) if p >= 0 else (inner, "")
    name = name.strip()
    props = tool_properties(tools, name)
    if not name or props is None:
        return None
    params = {}
    for m in _ARG_RE.finditer(raw):
        params[m.group("key").strip()] = m.group("value")
    if props:
        for k, v in params.items():
            prop = props.get(k)
            if isinstance(prop, dict):
                params[k] = coerce_to_schema_type(v, extract_types_from_schema(prop))
    return name, json.dumps(params, ensure_ascii=False)


def ref_parse(text, thinking, tools):
    tools = tools or None
    reasoning, content, calls = [], [], []
    state = "R" if thinking else "C"
    i = 0
    while i <= len(text):
        if state in "RC":
            ms = ["</think>", "<think>"] + (["<tool_call>"] if tools else [])
            hits = [(text.find(m, i), m) for m in ms if text.find(m, i) >= 0]
            if not hits:
                (reasoning if state == "R" else content).append(text[i:])
                break
            p, m = min(hits)
            (reasoning if state == "R" else content).append(text[i:p])
            i = p + len(m)
            state = {"</think>": "C", "<think>": "R", "<tool_call>": "T"}[m]
        else:
            p = text.find("</tool_call>", i)
            if p < 0:
                content.append("<tool_call>" + text[i:])
                break
            call = parse_call(text[i:p], tools)
            if call:
                calls.append(list(call))
            else:
                content.append("<tool_call>" + text[i:p] + "</tool_call>")
            i = p + 12
            state = "C"
    return {"reasoning": "".join(reasoning), "content": "".join(content), "tool_calls": calls}


VALUES = ["a.txt", "", "  spaced  ", "多语言", "line\nbreak", "2", "007", "-3", "+4", "1_000", "2.0", "2.5", "1e5", "1e20", "inf",
          "nan", "true", "False", "1", "0", "null", "NULL", "[1, 2]", "{\"a\": [1, null]}", "[1e999]", "{bad", "\"quoted\"", "<b>", " 12 "]
SCHEMAS = [{"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"}, {"type": ["string", "null"]},
           {"type": ["integer", "null"]}, {"type": "array"}, {"type": "object"}, {"enum": ["a", 1, 2.5, None]}, {"anyOf": [{"type": "integer"}, {"type": "string"}]},
           {"type": "int"}, {"type": "bool"}, {"type": "whatever"}, {}, "not a dict"]
PARSE_TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
    f"p{i}": s for i, s in enumerate(SCHEMAS)}}}},
    {"type": "function", "function": {"name": "nop", "parameters": {"type": "object"}}},
    {"name": "bare", "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}}}]


def rand_call():
    name = R.choice(["f", "f", "f", " f\n", "nop", "bare", "unknown", ""])
    s = "<tool_call>" + name
    for _ in range(R.choice([0, 1, 2, 3, 4])):
        k = R.choice([f"p{i}" for i in range(len(SCHEMAS))] + ["n", "free", " p1 "])
        s += R.choice(["", "\n"]) + f"<arg_key>{k}</arg_key>" + R.choice(["", "\n", " \t"]) + f"<arg_value>{R.choice(VALUES)}</arg_value>"
    if R.random() < 0.05:
        s += "<arg_key>dangling</arg_key>"
    return s + R.choice(["</tool_call>"] * 9 + ["\n</tool_call>"])


def parse_cases(n):
    out = []
    for i in range(n):
        parts = []
        thinking = R.random() < 0.9
        if thinking:
            parts.append(R.choice(["Let me think.", "", "multi\nline 思考", "<think>again", "a <tool_call>f</tool_call> in reasoning"]))
            if R.random() < 0.9:
                parts.append("</think>")
        parts.append(R.choice(["", "Answer.", "\n\nSure: ", "stray </think> and <think>more", "x < y > z"]))
        for _ in range(R.choice([0, 0, 1, 1, 2, 3])):
            parts.append(rand_call())
            if R.random() < 0.3:
                parts.append(R.choice(["\n", " text between ", ""]))
        if R.random() < 0.05:
            parts.append("<tool_call>f<arg_key>p1</arg_key><arg_value>3")  # cut off
        tools = R.choice([PARSE_TOOLS, PARSE_TOOLS, None, []])
        text = "".join(parts)
        c = {"kind": "parse", "name": f"parse {i}", "text": text, "thinking": thinking, "result": ref_parse(text, thinking, tools)}
        if tools is not None:
            c["tools"] = tools
        out.append(c)
    return out


if __name__ == "__main__":
    cases = targeted()
    for i in range(N_RANDOM):
        cases.append(render_case(f"random {i}", rand_conversation(), tools=rand_tools(), kwargs=rand_kwargs(), agp=R.random() < 0.85))
    cases += parse_cases(3000)
    with open(OUT, "w") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_r = sum(c["kind"] == "render" for c in cases)
    n_e = sum("error" in c for c in cases)
    n_d = sum(bool(c.get("glm53_differs")) for c in cases)
    n_dt = sum(bool(c.get("glm53_differs")) and c["name"].startswith(("null", "missing", "empty", "no break")) for c in cases)
    print(f"{len(cases)} cases ({n_r} render, {n_e} of them raise; {len(cases) - n_r} parse) -> {OUT}")
    if GLM53_TEMPLATE is not None:
        print(f"{n_d} render cases differ under GLM-5.3's template ({n_dt} of the altar-1 delta cases, {n_d - n_dt} others)")
