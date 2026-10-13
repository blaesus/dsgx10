"""long.tokens.json minus the json task (tokens 566..12902): the same textwrap/heapq/DESIGN.md content
~12.3k tokens earlier, to tell content from context length. Optionally only its first N tokens.

  python3 golden/cut_long.py golden/long.tokens.json golden/long-cut.tokens.json           (16,152 tokens)
  python3 golden/cut_long.py golden/long.tokens.json golden/long-cut5120.tokens.json 5120  (collapse ~4,527)
"""
import json
import sys

src, out = sys.argv[1], sys.argv[2]
t = json.load(open(src))["tokens"]
cut = t[:566] + t[12903:]
if len(sys.argv) > 3:
    cut = cut[: int(sys.argv[3])]
json.dump({"tokens": cut, "n_prompt": len(cut), "text": f"long.tokens.json with tokens 566..12902 removed, "
           f"first {len(cut)}", "shift": 12903 - 566}, open(out, "w"))
print(len(t), "->", len(cut))
