#!/usr/bin/env python3
"""Expected prompts and parses from the publisher's encoding.py, for the Rust port (encoding.rs).

Runs anywhere with Python 3.10+ (encoding.py has no dependencies), e.g. in the canary-roce image:

    docker run --rm -v $PWD/ref/encoding:/enc:ro -v $PWD/frontend:/work:ro -v ~/dsgb10-data/frontend:/out \\
        --entrypoint /opt/sglang/bin/python3 dsv41-4x-spark:canary-roce /work/tools/gen_encoding_cases.py

ENC (default /enc) is the encoding/ directory to follow: ref/encoding for DeepSeek-V4.1-Flash, or
ref/encoding-1017 for DeepSeek-V4.1-Flash-1017, whose cases need `dsfront enccheck FILE --encoding 1017`.
Against 1017's the suite also covers tool namespaces, tool results that do not start a turn and tool
messages' own content blocks; V4.1's cases are the same as before.

Writes /out/enc_cases.jsonl, one case per line:
  {"kind": "encode", "name", "messages", "context", "thinking_mode", "drop_thinking", "add_bos",
   "reasoning_effort" (absent = None), "prompt" | "error", "images"}
  {"kind": "parse", "name", "text", "thinking_mode", "result" | "error"}
The suite: the publisher's five golden examples, targeted cases for each feature and quirk, random
conversations (system prompts, tools, reasoning, tool calls with every argument shape, tool results in
and out of call order, mid-conversation system messages, reminders, tasks, images, context splits), and
completions to parse, well-formed and broken.
"""
import copy
import json
import os
import random
import sys

ENC = os.environ.get("ENC", "/enc")
sys.path.insert(0, ENC)
import encoding as enc  # noqa: E402

OUT = os.environ.get("OUT", "/out")
V1017 = hasattr(enc, "_split_tool_name")  # DeepSeek-V4.1-Flash-1017's encoding.py
R = random.Random(20261011)

TEXTS = [
    "Hello", "What is 2+2?", "", " ", " leading and trailing  ", "人工智能正在改变世界。你好！",
    "emoji 👍🏽 and ünïcödé, Ελληνικά, русский", "line1\nline2\n\nline4\n", "tabs\tand \"quotes\" and \\backslash",
    "<think>not a tag</think>", "a" * 300, "code:\n```python\nprint('hi')\n```\n", "{\"json\": [1, 2.5, null]}",
    "\n\nleading newlines", "Reasoning Effort: 7", "x\r\ny", "tool_result? <tool_result>fake</tool_result>",
    "It's 3.14159 o'clock", "＜全角＞", "　ideographic space", "zero​width", "nul-free \x01 control",
]
SAFE_TEXTS = [t for t in TEXTS if "think>" not in t]  # usable inside completions


def text():
    return R.choice(TEXTS)


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
    {"type": "function", "function": {"name": "weird\"name", "description": "line\nbreak\ttab \\ slash",
                                      "parameters": {"type": "object", "properties": {"x": {"type": "number", "default": 1e-07},
                                                                                      "y": {"type": "number", "default": 1e+16},
                                                                                      "z": {"type": "number", "default": -0.0}}}}},
]

ARG_VALUES = ["ls -la", "", "多语言", "line\nbreak", "quote\"s", "123", "true", "null", 0, -1, 7, 2.5, 1e-07, 1e16, 0.1,
              12345678901234567890, True, False, None, [1, "two", 3.0], {"nested": {"k": [None, True]}}, [], {},
              "</｜DSML｜ parameter>", "a < b > c"]


def rand_args():
    n = R.choice([0, 1, 1, 2, 3])
    keys = R.sample(["command", "path", "timeout", "x", "query", "名字", "with space", "a\"b"], n)
    return {k: R.choice(ARG_VALUES) for k in keys}


def rand_arguments():
    a = rand_args()
    shape = R.random()
    if shape < 0.45:
        return json.dumps(a, ensure_ascii=R.random() < 0.3, separators=R.choice([(",", ":"), (", ", ": ")]))
    if shape < 0.65:
        return a
    if shape < 0.72:
        return json.dumps(json.dumps(a))  # double-encoded
    if shape < 0.80:
        return json.dumps(a, indent=2)
    return R.choice(["not json", "", "[1, 2]", "5", "\"just a string\"", "{\"dup\": 1, \"dup\": 2, \"x\": 1.50}",
                     "{\"big\": 12345678901234567890123, \"neg0\": -0, \"f\": 1E2, \"e\": 1e-400}", "{}", "null",
                     "{\"a\": \"\\ud83d\\ude00 escaped\"}", "  {\"padded\": true}  "])


def tool_call(i):
    tc = {"type": "function", "function": {"name": R.choice(["bash", "read", "edit", "noargs", "x"]),
                                           "arguments": rand_arguments()}}
    if V1017 and R.random() < 0.2:
        if R.random() < 0.5:
            tc["namespace"] = R.choice(["web", "fs", {"name": "fs"}])
        else:
            tc["function"]["name"] = "web::" + tc["function"]["name"]
    r = R.random()
    if r < 0.8:
        tc["id"] = f"call_{i}"
    elif r < 0.9:
        tc["function"]["id"] = f"call_{i}"
    return tc


def user_msg():
    r = R.random()
    if r < 0.7:
        return {"role": "user", "content": text()}
    if r < 0.85:
        return {"role": "user", "content": [{"type": "text", "text": text()} for _ in range(R.randint(1, 3))]}
    if r < 0.92:
        return {"role": "user", "content": [{"type": "text", "text": text()},
                                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
    if r < 0.96:
        return {"role": "user"}
    return {"role": "user", "content": text(), "name": "someone"}


def assistant_msg(thinking, calls):
    m = {"role": "assistant"}
    r = R.random()
    m["content"] = text() if r < 0.7 else (None if r < 0.85 else "")
    if thinking or R.random() < 0.2:
        m["reasoning_content"] = R.choice([text(), None, ""])
    if calls:
        m["tool_calls"] = calls
    return m


def tool_msgs(calls):
    out = []
    for i, tc in enumerate(calls):
        tid = tc.get("id") or tc["function"].get("id", "")
        r = R.random()
        if r < 0.7:
            c = text()
        elif r < 0.8:
            c = [{"type": "text", "text": text()}, {"type": "text", "text": text()}]
        elif r < 0.85:
            c = [{"type": "text", "text": text()}, {"type": "audio"}]
        elif r < 0.9:
            c = None
        else:
            c = ""
        m = {"role": "tool", "tool_call_id": tid, "content": c}
        if R.random() < 0.05:
            del m["tool_call_id"]
        out.append(m)
    if R.random() < 0.5:
        R.shuffle(out)
    return out


def namespaced(tools):
    """1017 only: some tools get a namespace, given on the tool or in the name."""
    out = []
    for t in tools:
        t = copy.deepcopy(t)
        r = R.random()
        if r < 0.25:
            t["namespace"] = R.choice(NAMESPACES)
        elif r < 0.35:
            t["function"]["name"] = "ns::" + t["function"]["name"]
        out.append(t)
    return out


def conversation():
    thinking = R.random() < 0.6
    msgs = []
    if R.random() < 0.7:
        s = {"role": "system", "content": R.choice([text(), "You are a helpful assistant.", [{"type": "text", "text": "parts"}], None])}
        msgs.append(s)
    tools = None
    if R.random() < 0.5:
        tools = R.sample(TOOLS, R.randint(1, len(TOOLS)))
    turns = R.randint(1, 5)
    ncall = 0
    for t in range(turns):
        msgs.append(user_msg())
        if R.random() < 0.15:
            msgs.append(user_msg())  # consecutive user messages merge
        if t == turns - 1 and R.random() < 0.6:
            break
        k = R.choice([0, 0, 1, 1, 2, 3]) if tools or R.random() < 0.2 else 0
        calls = [tool_call(ncall + i) for i in range(k)]
        ncall += k
        msgs.append(assistant_msg(thinking, calls))
        if calls:
            msgs += tool_msgs(calls)
            if R.random() < 0.3:
                msgs.append(user_msg())  # a user message right after tool results merges into them
            if R.random() < 0.5 and t < turns - 1:
                k2 = R.choice([1, 2])
                calls2 = [tool_call(ncall + i) for i in range(k2)]
                ncall += k2
                msgs.append(assistant_msg(thinking, calls2))
                msgs += tool_msgs(calls2)
        if R.random() < 0.1:
            msgs.append({"role": "system", "content": text()})
        if R.random() < 0.05:
            msgs.append({"role": "latest_reminder", "content": text()})
    r = R.random()
    if r < 0.1:
        msgs.append(assistant_msg(thinking, []))  # prefill / continuation
        if R.random() < 0.5:
            msgs[-1]["wo_eos"] = True
    elif r < 0.15:
        msgs.append({"role": "latest_reminder", "content": text()})
    elif r < 0.2:
        for m in reversed(msgs):
            if m["role"] == "user":
                m["task"] = R.choice(list(enc.DS_TASK_SP_TOKENS))
                break
    if tools:
        msgs[0]["tools"] = namespaced(tools) if V1017 and R.random() < 0.5 else tools
    if R.random() < 0.05 and msgs[0]["role"] == "system":
        msgs[0]["response_format"] = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    return msgs, thinking


def effort():
    r = R.random()
    if r < 0.3:
        return "absent"
    if r < 0.4:
        return None
    if r < 0.95:
        return R.choice([1, 25, 50, 75, 100, 33, "low", "high", "xhigh", "max"])
    return R.choice([0, 101, -5, "medium", "none", 50.0, True, "", [50]])


cases = []


def add_encode(name, messages, thinking_mode, context=None, drop_thinking=True, add_bos=True, reasoning_effort="absent"):
    case = {"kind": "encode", "name": name, "messages": messages, "context": context, "thinking_mode": thinking_mode,
            "drop_thinking": drop_thinking, "add_bos": add_bos}
    kwargs = {}
    if reasoning_effort != "absent":
        case["reasoning_effort"] = reasoning_effort
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        prompt, media = enc.encode_messages(copy.deepcopy(messages), thinking_mode=thinking_mode, context=copy.deepcopy(context),
                                            drop_thinking=drop_thinking, add_default_bos_token=add_bos,
                                            return_multi_modal_data=True, **kwargs)
        case["prompt"] = prompt
        case["images"] = media["images"]
    except Exception as e:  # the port must fail too
        case["error"] = f"{type(e).__name__}: {e}"
    cases.append(case)
    return case


def add_parse(name, text, thinking_mode):
    case = {"kind": "parse", "name": name, "text": text, "thinking_mode": thinking_mode}
    try:
        case["result"] = enc.parse_message_from_completion_text(text, thinking_mode)
    except Exception as e:
        case["error"] = f"{type(e).__name__}: {e}"
    cases.append(case)


def golden():
    d = f"{ENC}/tests"
    for i in range(1, 6):
        c = enc.load_cases(f"{d}/test_input_{i}.json")[0]
        mode = c.get("thinking_mode") or "chat"
        eff = c.get("reasoning_effort")
        got = add_encode(f"golden_{i}", c["messages"], mode, context=c.get("context"),
                         reasoning_effort="absent" if eff is None else eff)
        want = open(f"{d}/test_output_{i}.txt").read()
        assert got.get("prompt") == want, f"golden {i}: encoding.py disagrees with its own test output"


def targeted():
    u = {"role": "user", "content": "q"}
    a = {"role": "assistant", "content": "a", "reasoning_content": "r"}
    s = {"role": "system", "content": "sys"}
    for mode in ["chat", "thinking"]:
        for drop in [True, False]:
            for bos in [True, False]:
                add_encode(f"basic_{mode}_{drop}_{bos}", [s, u, a, u], mode, drop_thinking=drop, add_bos=bos)
        add_encode(f"empty_{mode}", [], mode)
        add_encode(f"empty_bad_effort_{mode}", [], mode, reasoning_effort=0)
        add_encode(f"only_system_{mode}", [s], mode)
        add_encode(f"mid_system_{mode}", [s, u, a, {"role": "system", "content": "now"}], mode)
        add_encode(f"mid_system_first_user_{mode}", [u, {"role": "system", "content": "x"}, a, u], mode)
        add_encode(f"assistant_last_{mode}", [u, a], mode)
        add_encode(f"wo_eos_{mode}", [u, dict(a, wo_eos=True)], mode)
        add_encode(f"reminder_{mode}", [s, u, {"role": "latest_reminder", "content": "rem"}], mode)
        add_encode(f"reminder_none_{mode}", [u, {"role": "latest_reminder"}], mode)
        add_encode(f"reminder_then_assistant_{mode}", [u, {"role": "latest_reminder", "content": "r"}, a], mode)
        for task in list(enc.DS_TASK_SP_TOKENS) + ["bogus"]:
            add_encode(f"task_{task}_{mode}", [s, dict(u, task=task)], mode)
            add_encode(f"task_{task}_then_assistant_{mode}", [dict(u, task=task), a, u], mode)
        add_encode(f"unknown_role_{mode}", [u, {"role": "narrator", "content": "x"}], mode)
        add_encode(f"unknown_role_early_{mode}", [{"role": "narrator", "content": "x"}, u, a, u], mode)
        add_encode(f"no_role_{mode}", [{"content": "x"}], mode)
        add_encode(f"user_null_content_{mode}", [{"role": "user", "content": None}], mode)
        add_encode(f"user_int_content_{mode}", [{"role": "user", "content": 5}], mode)
        add_encode(f"system_int_content_{mode}", [{"role": "system", "content": 5}, u], mode)
        add_encode(f"assistant_float_content_{mode}", [u, {"role": "assistant", "content": 2.50}], mode)
        add_encode(f"user_empty_blocks_{mode}", [{"role": "user", "content": []}], mode)
        add_encode(f"user_blocks_{mode}", [{"role": "user", "content_blocks": [{"type": "text", "text": "x"}, {"type": "thing"}]}], mode)
        add_encode(f"placeholder_in_text_{mode}", [{"role": "user", "content": "x" + enc.IMAGE_PLACEHOLDER}], mode)
        add_encode(f"placeholder_in_block_{mode}", [{"role": "user", "content": [{"type": "text", "text": enc.IMAGE_PLACEHOLDER}]}], mode)
        add_encode(f"image_anthropic_{mode}", [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "AA"}}, {"type": "text", "text": "what?"}]}], mode)
        add_encode(f"image_bad_{mode}", [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}], mode)
        add_encode(f"image_str_url_{mode}", [{"role": "user", "content": [{"type": "image_url", "image_url": "http://x/y.png"}]}], mode)
        add_encode(f"image_in_tool_{mode}", [u, {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "{}"}}]},
                                             {"role": "tool", "tool_call_id": "1", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}], mode)
        add_encode(f"tools_on_user_{mode}", [dict(u, tools=TOOLS[:2]), a, u], mode)
        add_encode(f"tools_on_system_{mode}", [dict(s, tools=TOOLS), u, a, u], mode)
        add_encode(f"tools_empty_{mode}", [dict(s, tools=[]), u, a, u], mode)
        add_encode(f"tools_bad_{mode}", [dict(s, tools=[{"type": "function"}]), u], mode)
        add_encode(f"response_format_{mode}", [dict(s, response_format={"type": "json_object"}), u], mode)
        add_encode(f"tool_call_no_args_{mode}", [u, {"role": "assistant", "tool_calls": [{"function": {"name": "f"}}]}], mode)
        add_encode(f"tool_call_int_name_{mode}", [u, {"role": "assistant", "tool_calls": [{"function": {"name": 7, "arguments": {}}}]}], mode)
        add_encode(f"tool_role_unmerged_ctx_{mode}", [u], mode, context=[u, {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "{}"}}, {"id": "b", "function": {"name": "g", "arguments": "{}"}}]}])
        # tool results answered in reverse order, split across context and messages
        call_a = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{\"x\": 1}"}},
            {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{\"y\": 2}"}},
            {"id": "c", "type": "function", "function": {"name": "h", "arguments": "{}"}}]}
        res = [{"role": "tool", "tool_call_id": i, "content": f"result {i}"} for i in "cab"]
        add_encode(f"sort_results_{mode}", [u, call_a] + res + [u], mode)
        add_encode(f"sort_results_ctx_{mode}", res + [u], mode, context=[u, call_a])
        add_encode(f"sort_results_ctx2_{mode}", res[1:] + [u], mode, context=[u, call_a, res[0]])
        add_encode(f"sort_unknown_ids_{mode}", [u, call_a, dict(res[0], tool_call_id="zzz"), res[1], dict(res[2], tool_call_id=None)], mode)
        add_encode(f"dup_ids_{mode}", [u, dict(call_a, tool_calls=[dict(t, id="a") for t in call_a["tool_calls"]]), res[0], res[1]], mode)
        add_encode(f"tool_then_user_merge_{mode}", [u, call_a] + res + [{"role": "user", "content": "and then"}], mode)
        add_encode(f"tool_after_task_user_{mode}", [dict(u, task="query"), {"role": "tool", "tool_call_id": "x", "content": "r"}], mode)
        for eff in [None, 1, 50, 100, "low", "high", "xhigh", "max", 0, 101, "medium", 50.0, True, False, "75", [1]]:
            add_encode(f"effort_{eff!r}_{mode}", [s, u], mode, reasoning_effort=eff)
            add_encode(f"effort_nosys_{eff!r}_{mode}", [u], mode, reasoning_effort=eff)
        add_encode(f"context_{mode}", [u], mode, context=[s, u, a])
        add_encode(f"context_drop_{mode}", [a, u], mode, context=[s, u])
        add_encode(f"context_only_assistant_{mode}", [a], mode, context=[s, u])
        add_encode(f"context_odd_roles_{mode}", [{"role": "narrator", "content": "x"}, u], mode,
                   context=[u, {"role": "narrator", "content": "y"}, {"role": "narrator", "content": "z"}, {"role": "narrator", "content": "w"}])
        add_encode(f"context_neg_{mode}", [{"role": "narrator"}, u], mode,
                   context=[u, {"role": "narrator"}, {"role": "narrator"}, {"role": "narrator"}])
        add_encode(f"context_empty_list_{mode}", [u], mode, context=[])
        add_encode(f"drop_with_tool_calls_{mode}", [u, call_a] + res + [a, u], mode)
        add_encode(f"args_shapes_{mode}", [u, {"role": "assistant", "tool_calls": [
            {"id": str(i), "function": {"name": "f", "arguments": v}} for i, v in enumerate(
                ["{\"a\": 1}", {"a": [1, 2.0, {"b": None}]}, "\"{\\\"x\\\": 1}\"", "nope", 7, None, [1], "{\"a\": NaN}",
                 "{\"a\": Infinity}", "{\"s\": \"\\u00e9\\n\"}", "{\"a\":1}  ", "{\"a\": 1} x"])]}], mode)


NAMESPACES = ["web", {"name": "fs"}, {"name": "fs", "description": "File tools."}, {"name": "x", "description": ""}]


def targeted_1017():
    """Tool namespaces, tool results that do not start a turn, and tool messages' content blocks."""
    u = {"role": "user", "content": "q"}
    s = {"role": "system", "content": "sys"}
    call = lambda name, i="a", **kw: dict({"id": i, "type": "function", "function": dict({"name": name, "arguments": "{\"x\": 1}"}, **kw.pop("f", {}))}, **kw)
    for mode in ["chat", "thinking"]:
        for ns in NAMESPACES + [None, "", 5, "a::b", {"description": "no name"}, ["::"], ["x"]]:
            tool = {"type": "function", "namespace": ns, "function": {"name": "search", "description": "Search", "parameters": {}}}
            add_encode(f"ns_tool_{ns!r}_{mode}", [dict(s, tools=[tool]), u], mode)
        for name in ["web::search", "::x", "x::", "a::b::c", "plain", 7]:
            add_encode(f"ns_tool_name_{name!r}_{mode}", [dict(s, tools=[{"type": "function", "function": {"name": name}}]), u], mode)
        add_encode(f"ns_tool_desc_none_{mode}", [dict(s, tools=[{"type": "function", "namespace": {"name": "n", "description": "D"}, "function": {"name": "f"}}]), u], mode)
        add_encode(f"ns_tool_desc_int_{mode}", [dict(s, tools=[{"type": "function", "namespace": {"name": "n", "description": "D"}, "function": {"name": "f", "description": 3}}]), u], mode)
        add_encode(f"ns_tool_on_function_{mode}", [dict(s, tools=[{"type": "function", "function": {"name": "f", "namespace": "n", "parameters": {}}}]), u], mode)
        add_encode(f"ns_tool_not_dict_{mode}", [dict(s, tools=[{"type": "function", "function": "f"}]), u], mode)
        for i, c in enumerate([call("web::search"), call("search", namespace="web"), call("search", f={"namespace": "web"}),
                               call("search", namespace={"name": "web"}), call("web::search", namespace="web"),
                               call("web::search", namespace="other"), call("a::b::c"), call("search", namespace=""),
                               call("search", namespace="", f={"namespace": "fn"}), call(7), call("s", namespace={"x": 1})]):
            add_encode(f"ns_call_{i}_{mode}", [u, {"role": "assistant", "content": "", "tool_calls": [c]}, u], mode)
        # tool results alone: the same turn, with and without tools declared
        loop = [u, {"role": "assistant", "reasoning_content": "r1", "content": "", "tool_calls": [call("f")]},
                {"role": "tool", "tool_call_id": "a", "content": "res"},
                {"role": "assistant", "reasoning_content": "r2", "content": "", "tool_calls": [call("g", "b")]},
                {"role": "tool", "tool_call_id": "b", "content": "res2"}]
        for drop in [True, False]:
            add_encode(f"turn_tool_loop_{drop}_{mode}", loop, mode, drop_thinking=drop)
            add_encode(f"turn_tool_loop_then_user_{drop}_{mode}", loop + [{"role": "user", "content": "next"}], mode, drop_thinking=drop)
            add_encode(f"turn_tool_loop_ctx_{drop}_{mode}", loop[3:], mode, context=loop[:3], drop_thinking=drop)
            add_encode(f"turn_tool_loop_tools_{drop}_{mode}", [dict(s, tools=TOOLS[:1])] + loop, mode, drop_thinking=drop)
        add_encode(f"turn_mid_system_{mode}", loop + [{"role": "system", "content": "now"}], mode)
        add_encode(f"turn_blocks_text_and_result_{mode}", loop[:2] + [{"role": "user", "content_blocks": [
            {"type": "tool_result", "tool_use_id": "a", "content": "r"}, {"type": "text", "text": "and"}]}], mode)
        add_encode(f"turn_blocks_not_dict_{mode}", [u, {"role": "user", "content_blocks": ["x"]}], mode)
        # tool messages' own content blocks
        for i, m in enumerate([{"content_blocks": [{"type": "text", "text": "cb"}]},
                               {"content": "str", "content_blocks": [{"type": "text", "text": "cb"}]},
                               {"content": None, "content_blocks": [{"type": "text", "text": "cb"}, {"type": "audio"}]},
                               {"content": [{"type": "text", "text": "list"}, {"type": "image_url", "image_url": {"url": "t.png"}}]},
                               {"content_blocks": []}]):
            add_encode(f"tool_blocks_{i}_{mode}", loop[:2] + [dict({"role": "tool", "tool_call_id": "a"}, **m)], mode)
        add_encode(f"tool_placeholder_{mode}", loop[:2] + [{"role": "tool", "tool_call_id": "a", "content": "x" + enc.IMAGE_PLACEHOLDER}], mode)
        add_encode(f"user_list_then_user_{mode}", [{"role": "user", "content": [{"type": "text", "text": "a"}]}, u], mode)
        # images in tool results, in call order
        two = {"role": "assistant", "content": "", "tool_calls": [call("f"), call("g", "b")]}
        res = [{"role": "tool", "tool_call_id": i, "content": [{"type": "text", "text": f"r{i}"}, {"type": "image_url", "image_url": {"url": f"{i}.png"}}]} for i in "ba"]
        add_encode(f"tool_images_order_{mode}", [u, two] + res, mode)
        add_encode(f"tool_images_order_ctx_{mode}", res + [u], mode, context=[u, two])


def random_cases(n):
    for i in range(n):
        msgs, thinking = conversation()
        mode = "thinking" if (thinking if R.random() < 0.8 else not thinking) else "chat"
        context = None
        if R.random() < 0.15 and len(msgs) > 2:
            k = R.randint(1, len(msgs) - 1)
            context, msgs = msgs[:k], msgs[k:]
        add_encode(f"random_{i}", msgs, mode, context=context, drop_thinking=R.random() < 0.85,
                   add_bos=R.random() < 0.9, reasoning_effort=effort())


def completion(thinking):
    parts = []
    if thinking:
        parts.append(R.choice(SAFE_TEXTS) + enc.thinking_end_token)
    parts.append(R.choice(SAFE_TEXTS))
    calls = []
    for i in range(R.choice([0, 0, 1, 2, 3])):
        args = rand_args()
        calls.append(enc.tool_call_template.format(dsml_token=enc.dsml_token, tool_call_tag_name=enc.tool_call_tag_name,
                                                   name=R.choice(["bash", "read", "名", "a b"] + (["web::search", "::x", "a::b::c"] if V1017 else [])),
                                                   arguments=enc.encode_arguments_to_dsml({"arguments": args})))
    if calls:
        parts.append("\n\n" + enc.tool_calls_template.format(dsml_token=enc.dsml_token, tool_calls="\n".join(calls),
                                                              tc_block_name=enc.tool_calls_block_name))
    parts.append(enc.eos_token)
    return "".join(parts)


BREAKS = [
    lambda t: t[:-len(enc.eos_token)],                       # no EOS
    lambda t: t + "junk",                                     # text after EOS
    lambda t: t.replace("｜DSML｜ ", "｜DSML｜"),               # V4 (unspaced) tags
    lambda t: t.replace("</think>", ""),                      # no end of thinking
    lambda t: t.replace(">\n<｜DSML｜ invoke", "><｜DSML｜ invoke", 1),
    lambda t: t.replace("string=\"true\"", "string=\"maybe\"", 1),
    lambda t: t.replace("</｜DSML｜ calls>", "</｜DSML｜ calls>\n", 1),
    lambda t: t.replace("</｜DSML｜ invoke>", "", 1),
    lambda t: t.replace("<｜DSML｜ parameter", "<｜DSML｜ parameter name=\"dup\" string=\"true\">1</｜DSML｜ parameter>\n<｜DSML｜ parameter", 1).replace(
        "<｜DSML｜ parameter name=\"dup\"", "<｜DSML｜ parameter name=\"dup\" string=\"true\">1</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"dup\"", 1),
    lambda t: t.replace("\n\n<｜DSML｜ calls", "\n\n<｜DSML｜ calls" + enc.eos_token, 1),
    lambda t: t[: t.find("<｜DSML｜ invoke") + 5] if "<｜DSML｜ invoke" in t else t,   # cut inside a call
    lambda t: t[: t.find("\n\n<｜DSML｜ calls") + len("\n\n<｜DSML｜ calls")] if "\n\n<｜DSML｜ calls" in t else t,
    lambda t: t.replace(enc.eos_token, "<think>" + enc.eos_token, 1),
]


def parse_cases(n):
    add_parse("readme", "Simple arithmetic.</think>2 + 2 = 4.<｜end▁of▁sentence｜>", "thinking")
    add_parse("empty_chat", enc.eos_token, "chat")
    add_parse("empty_thinking", "</think>" + enc.eos_token, "thinking")
    add_parse("only_calls_block", "x\n\n<｜DSML｜ calls", "chat")
    add_parse("calls_then_nothing", "x\n\n<｜DSML｜ calls>\n</｜DSML｜ calls>", "chat")
    add_parse("value_with_lt", "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"a\" string=\"true\">x</｜DSML｜ invoke>y<</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>" + enc.eos_token, "chat")
    add_parse("name_with_newline", "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"a\">\nb\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>" + enc.eos_token, "chat")
    add_parse("name_trailing_nl", "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke  name=\"a\">\n\n</｜DSML｜ invoke>\n</｜DSML｜ calls>" + enc.eos_token, "chat")
    add_parse("param_value_nl", "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"a\">\n<｜DSML｜ parameter name=\"p\" string=\"false\">[1,\n2]<\n/｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>" + enc.eos_token, "chat")
    for i in range(n):
        thinking = R.random() < 0.6
        t = completion(thinking)
        if R.random() < 0.3:
            t = R.choice(BREAKS)(t)
        add_parse(f"random_{i}", t, "thinking" if thinking else "chat")


if __name__ == "__main__":
    golden()
    targeted()
    if V1017:
        targeted_1017()
    random_cases(int(os.environ.get("N_ENCODE", "6000")))
    parse_cases(int(os.environ.get("N_PARSE", "4000")))
    path = os.path.join(OUT, "enc_cases.jsonl")
    with open(path, "w") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    enc_n = sum(c["kind"] == "encode" for c in cases)
    enc_err = sum(c["kind"] == "encode" and "error" in c for c in cases)
    par_n = sum(c["kind"] == "parse" for c in cases)
    par_err = sum(c["kind"] == "parse" and "error" in c for c in cases)
    print(f"{enc_n} encode cases ({enc_err} raise), {par_n} parse cases ({par_err} raise) -> {path}")
