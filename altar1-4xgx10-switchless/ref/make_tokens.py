"""Build the reference's token sequences: GLM-5.3 conversations rendered with the checkpoint's own
chat_template.jinja (jinja2, transformers' settings) and encoded with its tokenizer.json (`tokenizers`),
as docs/model.md § 7 specifies.

  python3 make_tokens.py --ckpt CKPT --ds-transcript ../../ds41f-4xgx10-switchless/golden/long.messages.json \
      --out-dir tokens [--check-transformers]

Writes, per sequence NAME: NAME.messages.json (the conversation), NAME.txt (the rendered text),
NAME.tokens.json ({"name", "tokens", "sha256_tokens", ...}), and SHA256SUMS over all of them.
Sequences:
  short     ~300 tokens: a passage of Jerome K. Jerome, "Three Men in a Boat" (1889, public domain)
  prose     ~4.9k: the opening of Charles Darwin, "The Voyage of the Beagle", chapter I (public domain)
  python    ~5k: this configuration's tools/*.py, as a code-review request
  agentic   the DeepSeek configuration's golden agentic transcript (4 tools, 3 tasks, 11 tool calls),
            re-encoded as a GLM conversation; machine paths and box names replaced
Every conversation ends with the generation prompt (`<|assistant|><think>`), as a served request does.
"""
import argparse
import hashlib
import json
import os
import re

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.dirname(HERE)

# The DeepSeek transcript read a design document from a worktree path and names boxes by their
# host names; neither belongs in a public token file. The host names are not in this repository: pass
# them with --rename NAME=REPLACEMENT (applied in the order given, so give longer names first). The
# committed token files were made with the four boxes' names mapped to "rank 0" ... "rank 3".
SCRUB = [
    (re.compile(r"/home/[a-z]+/projects/dsgb10/\.claude/worktrees/agent-[0-9a-f]+"), "/workspace/dsgb10"),
    (re.compile(r"/home/[a-z]+"), "~"),
]


def scrub(s):
    for pat, rep in SCRUB:
        s = pat.sub(rep, s)
    return s


def scrub_obj(o):
    if isinstance(o, str):
        return scrub(o)
    if isinstance(o, list):
        return [scrub_obj(x) for x in o]
    if isinstance(o, dict):
        return {k: scrub_obj(v) for k, v in o.items()}
    return o


def render(template_src, messages, tools=None, **kw):
    """transformers' chat-template environment (utils/chat_template_utils.py): sandboxed,
    trim_blocks, lstrip_blocks, loopcontrols, a json.dumps tojson."""
    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template_src).render(messages=messages, tools=tools, add_generation_prompt=True, **kw)


def glm_agentic(ds):
    """The DeepSeek transcript (OpenAI message form, JSON-string arguments) as a GLM conversation:
    arguments parsed to mappings (the template iterates them), everything else kept."""
    msgs = []
    for m in ds["messages"]:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                f = c.get("function", c)
                args = f["arguments"]
                calls.append({"id": c.get("id"), "type": "function",
                              "function": {"name": f["name"], "arguments": json.loads(args) if isinstance(args, str) else args}})
            m["tool_calls"] = calls
        msgs.append(m)
    return scrub_obj(msgs), scrub_obj(ds["tools"])


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def build(args):
    seqs = {}
    short = read(os.path.join(HERE, "text", "short.txt"))
    seqs["short"] = ([{"role": "user", "content":
                       "Here is a passage from Jerome K. Jerome's novel *Three Men in a Boat* (1889). "
                       "What is the narrator's complaint, and how does the passage make it funny?\n\n" + short}], None)
    prose = read(os.path.join(HERE, "text", "prose.txt"))
    seqs["prose"] = ([{"role": "user", "content":
                       "Below is the opening of chapter I of Charles Darwin's *The Voyage of the Beagle*. "
                       "Summarise the route of the voyage so far and list the natural-history observations "
                       "Darwin makes, in order.\n\n" + prose}], None)
    files = sorted(f for f in os.listdir(os.path.join(CONF, "tools")) if f.endswith(".py"))
    code = "".join(f"### tools/{f}\n```python\n{read(os.path.join(CONF, 'tools', f))}```\n\n" for f in files)
    seqs["python"] = ([{"role": "system", "content": "You are a careful senior Python reviewer."},
                       {"role": "user", "content":
                        "Review these three scripts from our repository: point out bugs, unclear names and "
                        "anything that would break on a different machine.\n\n" + code}], None)
    with open(args.ds_transcript) as f:
        msgs, tools = glm_agentic(json.load(f))
    seqs["agentic"] = (msgs, tools)
    if args.diagnostic:
        # the transcript's first file (json/__init__.py, its second tool result), first 8,000 characters:
        # as a plain document after the template's prefix, and quoted in a single user turn
        doc = [m for m in msgs if m["role"] == "tool"][1]["content"][:8000]
        seqs["doc"] = (None, doc)
        seqs["docchat"] = ([{"role": "user", "content": "Here is a file from the Python standard library.\n\n" + doc}], None)
    return seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint dir with tokenizer.json and chat_template.jinja")
    ap.add_argument("--ds-transcript", required=True, help="ds41f golden/long.messages.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rename", action="append", default=[], metavar="NAME=REPLACEMENT",
                    help="replace a host name in the transcript (repeatable)")
    ap.add_argument("--diagnostic", action="store_true",
                    help="also doc (json/__init__.py as a plain document) and docchat (the same in a user turn)")
    ap.add_argument("--check-transformers", action="store_true",
                    help="also render and encode with transformers' AutoTokenizer.apply_chat_template and compare")
    args = ap.parse_args()
    for r in args.rename:
        name, rep = r.split("=", 1)
        SCRUB.append((re.compile(re.escape(name)), rep))
    os.makedirs(args.out_dir, exist_ok=True)
    tok = Tokenizer.from_file(os.path.join(args.ckpt, "tokenizer.json"))
    tmpl = read(os.path.join(args.ckpt, "chat_template.jinja"))
    hf = None
    if args.check_transformers:
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(args.ckpt)
    sums = {}
    for name, (msgs, tools) in build(args).items():
        text = "[gMASK]<sop>" + tools if msgs is None else render(tmpl, msgs, tools)
        ids = tok.encode(text, add_special_tokens=False).ids
        assert tok.decode(ids, skip_special_tokens=False) == text, f"{name}: decode does not round-trip"
        if hf is not None and msgs is not None:
            t2 = hf.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
            i2 = hf(t2, add_special_tokens=False)["input_ids"]
            print(f"{name}: transformers text {'==' if t2 == text else '!='} ours, ids {'==' if i2 == ids else '!='} ours")
        sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        rec = {"name": name, "n": len(ids), "sha256_tokens": sha, "tokens": ids}
        for suffix, data in ((".messages.json", json.dumps({"tools": tools if msgs is not None else None, "messages": msgs}, ensure_ascii=False, indent=1)),
                             (".txt", text), (".tokens.json", json.dumps(rec))):
            p = os.path.join(args.out_dir, name + suffix)
            with open(p, "w", encoding="utf-8") as f:
                f.write(data)
            sums[name + suffix] = hashlib.sha256(data.encode("utf-8")).hexdigest()
        print(f"{name}: {len(ids)} tokens, sha256(json ids) {sha[:16]}")
    with open(os.path.join(args.out_dir, "SHA256SUMS"), "w") as f:
        f.writelines(f"{h}  {n}\n" for n, h in sorted(sums.items()))


if __name__ == "__main__":
    main()
