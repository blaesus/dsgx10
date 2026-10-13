"""Token sequences for the golden runs, encoded with the publisher's encoding.py.

  make_tokens.py chat     <ckpt> <prompt> <completion> <out.json>
      one user turn in chat mode (thinking off), completion text re-tokenised and appended
  make_tokens.py sglang   <ckpt> <golden_sglang.json> <key> <out.json>
      the same, but the completion is the exact token sequence SGLang generated, rebuilt from the
      token bytes in its logprobs (re-tokenising generated text can change the ids)
  make_tokens.py messages <ckpt> <messages.json> <out.json>
      a whole conversation: {"thinking_mode", "reasoning_effort"?, "tools"?, "messages",
      "completion"?}; tools attach to the first message as in encoding.load_cases

The output is {"tokens", "n_prompt", "text"}: text is the encoded prompt string.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "encoding"))
from encoding import encode_messages  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def bytes_to_unicode():
    """GPT-2 byte-level BPE alphabet: byte -> printable unicode character."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def ids_from_logprob_bytes(tok, content):
    """Token ids of a generated sequence from SGLang's per-token bytes."""
    b2u = bytes_to_unicode()
    ids = []
    for ent in content:
        raw = bytes(ent["bytes"])
        cand = "".join(b2u[b] for b in raw)
        i = tok.convert_tokens_to_ids(cand)
        if i is None or i == tok.unk_token_id:
            i = tok.convert_tokens_to_ids(ent["token"])  # added/special tokens are stored literally
        if i is None or i == tok.unk_token_id or tok.decode([i]) != raw.decode("utf-8", "replace"):
            raise ValueError(f"cannot map generated token {ent['token']!r} back to an id")
        ids.append(i)
    return ids


def main():
    mode, ckpt = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(ckpt)
    completion_ids = []
    if mode == "chat":
        prompt, completion, out = sys.argv[3], sys.argv[4], sys.argv[5]
        text = encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat")
        if completion:
            completion_ids = tok.encode(completion, add_special_tokens=False)
    elif mode == "sglang":
        src, key, out = sys.argv[3], sys.argv[4], sys.argv[5]
        rec = json.load(open(src))[key]
        text = encode_messages([{"role": "user", "content": rec["prompt"]}], thinking_mode="chat")
        completion_ids = ids_from_logprob_bytes(tok, rec["logprobs"]["content"])
        retok = tok.encode(rec["content"], add_special_tokens=False)
        same = sum(1 for a, b in zip(completion_ids, retok) if a == b)
        print(f"sglang ids: {len(completion_ids)}, re-tokenised text: {len(retok)}, "
              f"identical prefix positions: {same}", flush=True)
    elif mode == "messages":
        src, out = sys.argv[3], sys.argv[4]
        case = json.load(open(src))
        messages = case["messages"]
        if case.get("tools"):
            messages[0]["tools"] = case["tools"]
        text = encode_messages(messages, thinking_mode=case.get("thinking_mode", "chat"),
                               reasoning_effort=case.get("reasoning_effort"))
        if case.get("completion"):
            completion_ids = tok.encode(case["completion"], add_special_tokens=False)
    else:
        raise SystemExit(__doc__)
    ids = tok.encode(text, add_special_tokens=False)
    n_prompt = len(ids)
    ids += completion_ids
    json.dump({"tokens": ids, "n_prompt": n_prompt, "text": text}, open(out, "w"))
    print(n_prompt, len(ids), repr(text[:120]), flush=True)


if __name__ == "__main__":
    main()
