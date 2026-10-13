"""Compare two golden runs' next-token distributions (logits_full) position by position.

  cmp_runs.py <run_a> <run_b>   (a = reference)
"""
import json
import sys

import numpy as np

a_dir, b_dir = sys.argv[1], sys.argv[2]
ma = json.load(open(f"{a_dir}/manifest.json"))
A = np.load(f"{a_dir}/logits_full.npy").astype(np.float64)
B = np.load(f"{b_dir}/logits_full.npy").astype(np.float64)
n = min(len(A), len(B))
A, B = A[-n:], B[-n:]
L, n_prompt = ma["L"], ma["n_prompt"]
first = L - n


def lsm(x):
    m = x.max(-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))


la, lb = lsm(A), lsm(B)
for label, lo in (("all positions", first), ("completion positions", max(first, n_prompt - 1))):
    agree, d5, kl, clear, near = 0, [], [], [], 0
    rows = list(range(lo, L - 1))
    for p in rows:
        a, b = la[p - first], lb[p - first]
        oa, ob = np.argsort(-a)[:5], np.argsort(-b)[:2]
        d5.extend(np.abs(b[oa] - a[oa]))
        pa = np.exp(a)
        kl.append(float((pa * (a - b)).sum()))
        if oa[0] == ob[0]:
            agree += 1
        else:
            ma_, mb_ = a[oa[0]] - a[ob[0]], b[ob[0]] - b[oa[0]]
            if ma_ > 1.0 and mb_ > 1.0:
                clear.append((p, int(oa[0]), int(ob[0]), round(float(ma_), 3), round(float(mb_), 3)))
            else:
                near += 1
    d5 = np.array(d5)
    print(f"{label}: {len(rows)} rows, argmax agree {agree}/{len(rows)} (near-tie flips {near}, "
          f"clear flips {len(clear)}); |d logprob| over a's top-5: mean {d5.mean():.4f} median "
          f"{np.median(d5):.4f} p95 {np.percentile(d5, 95):.4f} max {d5.max():.4f}; "
          f"KL(a||b) mean {np.mean(kl):.5f} max {np.max(kl):.4f}")
    for c in clear:
        print("  clear flip (pos, a_top, b_top, a_margin, b_margin):", c)
