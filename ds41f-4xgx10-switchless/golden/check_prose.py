"""Compare a golden run's next-token predictions with SGLang's greedy completion and top-5 logprobs.

  golden/run_ref.sh py /work/golden/check_prose.py /out/prose /work/bench/golden_sglang.json prose

For each completion step: does our argmax equal the token SGLang generated, how far apart are the
two top candidates (ours and SGLang's), and how close are our log-probabilities of SGLang's top-5
tokens to SGLang's own. Needs the run's full logits at the completion positions (--full-logits all).
"""
import json
import sys

import numpy as np

run_dir, sg_file, key = sys.argv[1], sys.argv[2], sys.argv[3]
man = json.load(open(f"{run_dir}/manifest.json"))
toks = np.load(f"{run_dir}/tokens.npy")
full = np.load(f"{run_dir}/logits_full.npy")  # [n_full, V], the last n_full positions
L, n_prompt = man["L"], man["n_prompt"]
first = L - full.shape[0]
rec = json.load(open(sg_file))[key]["logprobs"]["content"]
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/models")
except Exception:  # noqa: BLE001
    tok = None

rows, diffs, misses, per_step = [], [], [], []
for k, ent in enumerate(rec):
    p = n_prompt - 1 + k  # position whose logits predict completion token k
    if p + 1 >= L or p < first:
        break
    lg = full[p - first].astype(np.float64)
    lp = lg - (lg.max() + np.log(np.exp(lg - lg.max()).sum()))
    order = np.argsort(-lp)
    ours, nxt = int(order[0]), int(toks[p + 1])
    sg_top = ent["top_logprobs"]
    sg_margin = sg_top[0]["logprob"] - sg_top[1]["logprob"] if len(sg_top) > 1 else float("inf")
    our_margin = lp[order[0]] - lp[order[1]]
    step_d = []
    if tok is not None:
        for t in sg_top:
            ids = tok.encode(bytes(t["bytes"]).decode("utf-8", "replace"), add_special_tokens=False)
            if len(ids) == 1:
                step_d.append(lp[ids[0]] - t["logprob"])
        diffs.extend(abs(x) for x in step_d)
    per_step.append((k, float(lp[nxt]), float(ent["logprob"]), [round(float(x), 3) for x in step_d]))
    if ours != nxt:
        misses.append({"step": k, "pos": p, "sglang_token": nxt, "our_argmax": ours,
                       "our_lp_of_sglang_token": float(lp[nxt]), "our_top_lp": float(lp[ours]),
                       "our_margin": float(our_margin), "sglang_margin": float(sg_margin),
                       "sglang_token_text": ent["token"],
                       "our_text": tok.decode([ours]) if tok is not None else None})
    rows.append((k, ours == nxt, float(our_margin), float(sg_margin)))

n = len(rows)
agree = sum(r[1] for r in rows)
print(f"steps {n}: argmax == SGLang token at {agree}/{n}")
if diffs:
    d = np.array(diffs)
    print(f"|our logprob - SGLang logprob| over SGLang's top-5 tokens: mean {d.mean():.4f}, "
          f"median {np.median(d):.4f}, p95 {np.percentile(d, 95):.4f}, max {d.max():.4f} ({len(d)} pairs)")
ours_lp = np.array([r[1] for r in per_step])
sg_lp = np.array([r[2] for r in per_step])
print(f"mean logprob of SGLang's tokens: ours {ours_lp.mean():.4f}, SGLang's own {sg_lp.mean():.4f}")
for k, a, b, d in per_step[:12]:
    print(f"step {k:2d}: ours {a:8.4f} sglang {b:8.4f}  top5 deltas {d}")
for m in misses:
    print("miss:", json.dumps(m, ensure_ascii=False))
