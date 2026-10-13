"""A plain document as a token file: BOS + the text, no chat template, cut to N tokens. For perplexity
checks on text the model has surely seen (stdlib source, licence texts).

  golden/run_ref.sh py /work/golden/doc_tokens.py /usr/lib/python3.12/textwrap.py /work/golden/doc-textwrap.tokens.json 2048
"""
import json
import sys

from transformers import AutoTokenizer

src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
tok = AutoTokenizer.from_pretrained("/models")
text = open(src).read()
ids = ([tok.bos_token_id] + tok.encode(text, add_special_tokens=False))[:n]
json.dump({"tokens": ids, "n_prompt": len(ids), "text": f"BOS + the first {len(ids) - 1} tokens of {src}"},
          open(out, "w"))
print(src, len(ids), repr(tok.decode(ids[:12])))
