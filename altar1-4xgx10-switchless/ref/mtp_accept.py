"""MTP draft acceptance by tokens (draft at row t == the main argmax at t+1) for the runs all-f32, all and mtp-nvfp4
under GOLDEN_ROOT, and each run's drafts against all-f32's (A1d). REF and RUNS name other runs of the same tokens
(A1f: REF all-f32-abs, RUNS all-f32-abs,all-f32).

  python3 -I mtp_accept.py GOLDEN_ROOT [REF [RUNS]]
"""
import json
import os
import sys

import numpy as np

G = os.path.expanduser(sys.argv[1])


def load(run):
    d = os.path.join(G, run)
    man = json.load(open(os.path.join(d, "manifest.json")))
    am = np.load(os.path.join(d, "final/argmax.npy"))
    mtp = np.load(os.path.join(d, "mtp/argmax.npy"))
    rows = np.load(os.path.join(d, "mtp/rows_main.npy")) if os.path.exists(os.path.join(d, "mtp/rows_main.npy")) else None
    tok = np.load(os.path.join(d, "tokens.npy"))
    return man, am, mtp, rows, tok


REF = sys.argv[2] if len(sys.argv) > 2 else "all-f32"
RUNS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["all-f32", "all", "mtp-nvfp4"]
ref = load(REF)
for run in RUNS:
    man, am, mtp, rows, tok = load(run)
    print(run, "mtp", mtp.shape, mtp.dtype, "rows_main", None if rows is None else rows.shape)
    hit = n = 0
    moff = 0
    for s in man["sequences"]:
        o, k = s["off"], s["n"]
        # MTP rows: n - 1 a sequence (summarize.py's layout); row t drafts token t+2;
        # accepted if it equals the main argmax at t+1 (which predicts t+2)
        d = mtp[moff:moff + k - 2]
        acc = d == am[o + 1:o + k - 1]
        same = (mtp[moff:moff + k - 1] == ref[2][moff:moff + k - 1]).mean()
        moff += k - 1
        print(f"  {s['name']:8s} accept {acc.mean():.4f}  draft == {REF}'s draft {same:.4f}")
        hit += int(acc.sum()); n += acc.size
    print(f"  by tokens {hit / n:.4f} ({hit}/{n})")
