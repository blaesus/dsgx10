"""The token files (made with GLM-5.3's template, glm53 reference § 4) under altar-1's own chat_template.jinja and
tokenizer.json: re-render every NAME.messages.json, encode, and compare with NAME.txt and NAME.tokens.json; also
through transformers' AutoTokenizer.apply_chat_template.

  python3 -I check_tokens.py --ckpt CKPT [--names short prose python agentic]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tokens import render  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--names", nargs="+", default=["short", "prose", "python", "agentic"])
    a = ap.parse_args()
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokens")
    tok = Tokenizer.from_file(os.path.join(a.ckpt, "tokenizer.json"))
    with open(os.path.join(a.ckpt, "chat_template.jinja")) as f:
        tmpl = f.read()
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(a.ckpt)
    bad = 0
    for n in a.names:
        m = json.load(open(os.path.join(here, n + ".messages.json")))
        text = render(tmpl, m["messages"], m["tools"])
        ids = tok.encode(text, add_special_tokens=False).ids
        old_text = open(os.path.join(here, n + ".txt"), encoding="utf-8").read()
        old = json.load(open(os.path.join(here, n + ".tokens.json")))["tokens"]
        t2 = hf.apply_chat_template(m["messages"], tools=m["tools"], tokenize=False, add_generation_prompt=True)
        i2 = hf(t2, add_special_tokens=False)["input_ids"]
        ok = text == old_text and ids == old and t2 == text and i2 == ids
        bad += not ok
        print(f"{n:8s} {len(ids):6d} tokens: altar-1 template text == file {text == old_text}, ids == file {ids == old}; "
              f"transformers text == ours {t2 == text}, ids == ours {i2 == ids}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
