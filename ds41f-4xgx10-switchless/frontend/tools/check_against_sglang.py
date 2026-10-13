#!/usr/bin/env python3
"""Render requests the way the kit's SGLang does and compare with the front-end's prompts.

Input: the files `dsfront serve --dump-dir DIR` writes ({"request", "prompt", "ids"} per request).
Runs in the canary-roce image, which has the SGLang the kit serves with:

    docker run --rm -v DIR:/dumps:ro -v $PWD/frontend:/work:ro \\
        -v /data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8:/models:ro \\
        --entrypoint /opt/sglang/bin/python3 dsv41-4x-spark:canary-roce /work/tools/check_against_sglang.py

This repeats SGLang's DSV4.1 path (serving_chat._convert_to_internal_request and the dsv41 branch of
_apply_jinja_template) with the kit's patches (adapter/encoding_compat.py: publisher effort tiers,
enable_thinking alias, default effort 75), then compares prompt text, and the front-end's ids with
the tokenizer's encoding of the text (they differ only where a session resent the model's own ids).
"""
import copy
import glob
import json
import os
import sys

from sglang.srt.entrypoints.openai import chat_encoding, encoding_dsv41
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import normalize_assistant_tool_call_arguments
from transformers import AutoTokenizer

KIT_EFFORT = {"low": 50, "high": 75, "xhigh": 75, "max": 100}
KIT_DEFAULT_EFFORT = 75
encoding_dsv41.REASONING_EFFORT_MAPPINGS.clear()
encoding_dsv41.REASONING_EFFORT_MAPPINGS.update(KIT_EFFORT)


def kit_ctk(ctk):
    """adapter/encoding_compat.normalize_chat_template_kwargs"""
    out = dict(ctk or {})
    if "thinking" in out:
        out["thinking"] = bool(out["thinking"])
        out["enable_thinking"] = out["thinking"]
    elif "enable_thinking" in out:
        out["thinking"] = bool(out["enable_thinking"])
        out["enable_thinking"] = out["thinking"]
    return out


def sglang_prompt(body):
    req = ChatCompletionRequest(**body)
    req.chat_template_kwargs = kit_ctk(req.chat_template_kwargs)
    effort = req.chat_template_kwargs.pop("reasoning_effort", None)
    if effort is not None:
        req.reasoning_effort = effort
    thinking = req.chat_template_kwargs.get("thinking", False)
    messages = [m.model_dump() for m in req.messages]
    for m in messages:
        normalize_assistant_tool_call_arguments(m, strict=True)
    messages = copy.deepcopy(messages)
    for m in messages:
        if m.get("content") is None:
            m["content"] = ""
    if messages[0]["role"] != "system" and req.tools:
        messages.insert(0, {"role": "system", "content": ""})
    if req.tools:
        messages[0]["tools"] = [chat_encoding.dsv41_tool_payload(t) for t in req.tools]
    resolved = chat_encoding.resolve_dsv41_reasoning_effort(req.reasoning_effort)
    prompt, _ = encoding_dsv41.encode_messages(messages, thinking_mode="thinking" if thinking else "chat",
                                               reasoning_effort=KIT_DEFAULT_EFFORT if resolved is None else resolved,
                                               return_multi_modal_data=True)
    return prompt


def main():
    tok = AutoTokenizer.from_pretrained("/models")
    files = sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "/dumps", "*.json")))
    same_text = same_ids = 0
    for f in files:
        d = json.load(open(f))
        try:
            want = sglang_prompt(d["request"])
        except Exception as e:
            print(f"{os.path.basename(f)}: SGLang path raised {type(e).__name__}: {e}")
            continue
        if want == d["prompt"]:
            same_text += 1
        else:
            k = next((i for i, (a, b) in enumerate(zip(want, d["prompt"])) if a != b), min(len(want), len(d["prompt"])))
            print(f"{os.path.basename(f)}: prompt differs at char {k}:\n  sglang {want[max(0, k - 40):k + 40]!r}\n  ours   {d['prompt'][max(0, k - 40):k + 40]!r}")
        ids = tok.encode(want)
        if ids == d["ids"]:
            same_ids += 1
        else:
            k = next((i for i, (a, b) in enumerate(zip(ids, d["ids"])) if a != b), min(len(ids), len(d["ids"])))
            print(f"{os.path.basename(f)}: ids differ from the canonical encoding at token {k} of {len(ids)} "
                  f"(session reuse resends generated ids)")
    print(f"{len(files)} requests: prompt text identical to SGLang's in {same_text}, ids identical to its tokenisation in {same_ids}")


if __name__ == "__main__":
    main()
