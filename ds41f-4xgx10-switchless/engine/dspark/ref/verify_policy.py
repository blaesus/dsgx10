#!/usr/bin/env python3
# Verify-length policies on the DSpark reference run (prose): per step, the confidence head's survival
# S_j = prod_{i<=j} sigmoid(conf_i) and the number of leading drafts the golden tokens accept. Step cost:
# C(rows) linear between the runner's 19.4 ms (1 row) and 37.3 ms (6 rows), plus the draft pass D.
# usage: verify_policy.py REF_DIR [DRAFT_MS]   (e.g. /data/dsgb10/dspark/prose 3.2)
import array
import ast
import math
import sys

d = sys.argv[1]
D = float(sys.argv[2]) if len(sys.argv) > 2 else 3.2


def load(name, code):
    raw = open(f'{d}/{name}.npy', 'rb').read()
    hl = int.from_bytes(raw[8:10], 'little')
    hdr = ast.literal_eval(raw[10:10 + hl].decode())
    return array.array(code, raw[10 + hl:]), hdr['shape']


conf, sh = load('confidence', 'f')
acc, _ = load('accepted', 'i')
S = sh[0]
surv = []
for s in range(S):
    p, row = 1.0, []
    for j in range(5):
        p *= 1 / (1 + math.exp(-conf[s * 5 + j]))
        row.append(p)
    surv.append(row)


def C(rows):
    return 19.4 + (37.3 - 19.4) * (rows - 1) / 5


def run(choose, draft_cost):
    tok = ms = 0.0
    for s in range(S):
        k = choose(s)  # drafts verified
        tok += 1 + min(acc[s], k)
        ms += C(1 + k) + draft_cost
    return tok / ms * 1000


print(f'no drafting: {run(lambda s: 0, 0):.1f} tok/s')
for k in range(1, 6):
    print(f'fixed {k} drafts: {run(lambda s, k=k: k, D):.1f} tok/s')
for tau in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
    r = run(lambda s, t=tau: sum(1 for x in surv[s] if x >= t), D)
    ks = [sum(1 for x in surv[s] if x >= tau) for s in range(S)]
    print(f'survival >= {tau:.2f}: {r:.1f} tok/s (mean drafts verified {sum(ks) / S:.2f})')
oracle = run(lambda s: acc[s], D)
print(f'oracle (verify exactly the accepted drafts): {oracle:.1f} tok/s')
