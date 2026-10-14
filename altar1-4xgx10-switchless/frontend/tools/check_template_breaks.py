#!/usr/bin/env python3
"""Do the `break`s altar-1's chat template lacks change what it renders? (CPU only; transformers 5.x.)

    check_template_breaks.py MODEL_DIR GLM53_TEMPLATE [N]

MODEL_DIR holds altar-1's chat_template.jinja and tokenizer files; GLM53_TEMPLATE is GLM-5.3's
chat_template.jinja. altar-1's template differs from GLM-5.3's in visible_text's last branch (`else`
for `elif content is not none`) and in four `break`s missing from the tool-result block's sortability
check. This renders N random tool-result blocks (ids present, missing, empty, numeric, repeated, unknown;
results as strings, null, numbers, booleans, mappings, parts, lists of outputs, tool references; with
and without tools) with altar-1's template and with a hybrid, GLM-5.3's template with only the
visible_text line changed to altar-1's, so that what differs is the breaks alone, and counts the
outcomes: same text, both raising with the same message, both raising with different messages, and
(never seen) different text or one raising and the other not.
"""
import copy
import random
import sys

from transformers import AutoTokenizer

MODEL, GLM = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
hf = AutoTokenizer.from_pretrained(MODEL)
altar = open(MODEL + "/chat_template.jinja").read()
glm = open(GLM).read()
hybrid = glm.replace("{%- elif content is not none -%}", "{%- else -%}", 1)
assert hybrid != glm and hybrid.count("{%- break -%}") == altar.count("{%- break -%}") + 4
R = random.Random(7)
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
IDS = ["a", "b", "c", "", None, 7, "a"]


def render(template, msgs, tools):
    try:
        return "T", hf.apply_chat_template(copy.deepcopy(msgs), tools=copy.deepcopy(tools), chat_template=template,
                                          tokenize=False, add_generation_prompt=True)
    except Exception as e:
        return "E", f"{type(e).__name__}: {e}"


def rid():
    return R.choice(IDS)


def content():
    return R.choice(["x", "", None, 5, True, {"k": 1}, [], ["s"], [{"type": "text", "text": "t"}],
                     [{"type": "tool_reference", "name": "f"}], [{"output": "o", "id": rid()}, {"output": None, "tool_call_id": rid()}],
                     [{"output": [{"type": "tool_reference", "name": "f"}], "id": rid()}], [{"output": 3}], "s"])


counts = {}
for i in range(N):
    calls = []
    for j in range(R.choice([0, 1, 2, 3, 4])):
        tc = {"type": "function", "function": {"name": "f", "arguments": {}}}
        k = R.random()
        if k < 0.8:
            tc["id"] = rid()
        elif k < 0.9:
            tc["tool_call_id"] = rid()
        calls.append(tc)
    msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a", "tool_calls": calls}]
    for j in range(R.choice([1, 2, 3, 4, 5])):
        m = {"role": "tool", "content": content()}
        k = R.random()
        if k < 0.6:
            m["tool_call_id"] = rid()
        elif k < 0.75:
            m["id"] = rid()
        msgs.append(m)
    tools = TOOLS if R.random() < 0.7 else None
    a, h = render(altar, msgs, tools), render(hybrid, msgs, tools)
    if a == h:
        kind = "same text" if a[0] == "T" else "both raise, same message"
    elif a[0] == h[0] == "E":
        kind = "both raise, different messages"
    elif a[0] == h[0] == "T":
        kind = "DIFFERENT TEXT"
    else:
        kind = "ONE RAISES, THE OTHER NOT"
    counts[kind] = counts.get(kind, 0) + 1
print(f"{N} random tool-result blocks, altar-1's template against GLM-5.3's with altar-1's visible_text:")
for k in ["same text", "both raise, same message", "both raise, different messages", "DIFFERENT TEXT", "ONE RAISES, THE OTHER NOT"]:
    print(f"  {k}: {counts.get(k, 0)}")
sys.exit(1 if counts.get("DIFFERENT TEXT") or counts.get("ONE RAISES, THE OTHER NOT") else 0)
