"""One markdown row per run of the collapse experiments: top-1 accuracy in the 256-token windows after
the onset, and mean next-token logprob before the onset (0..4,479) and over the first 5,119 positions.

  golden/run_ref.sh py /work/golden/collapse_table.py /out/orig-base /out/off-l01 ...
"""
import json
import sys

import numpy as np

print("| run | what changed | 4,608+256 | 4,864+256 | logprob 0–4,479 | logprob 0–5,118 |")
print("|---|---|---|---|---|---|")
for r in sys.argv[1:]:
    toks = np.load(f"{r}/tokens.npy")
    lp = np.load(f"{r}/next_logprob.npy")[:5119]
    hit = np.load(f"{r}/logits_topi.npy")[:5119, 0] == toks[1:5120]
    m = json.load(open(f"{r}/manifest.json"))
    what = ", ".join(f"{k}={m[k]}" for k in ("ablate", "engram_off", "engram_off_layers", "overrides")
                     if m.get(k))
    print(f"| {r.split('/')[-1]} | {what or '-'} | {hit[4608:4864].mean():.2f} | {hit[4864:5119].mean():.2f} | "
          f"{lp[:4480].mean():.2f} | {lp.mean():.2f} |")
