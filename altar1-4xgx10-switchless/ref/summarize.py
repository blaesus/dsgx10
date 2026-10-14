"""Summarise a glmref.py output directory: teacher-forced top-1 and log-probability per sequence, greedy
text at a few places, expert-usage skew per MoE layer (overall and per sequence), the MTP layer's argmax
acceptance, index-set statistics, and per-layer timing.

  python3 summarize.py OUT [--tokenizer CKPT/tokenizer.json] [--json OUT/summary.json]
"""
import argparse
import glob
import json
import os

import numpy as np


def cover(mass, fracs=(0.5, 0.9, 0.99)):
    """How many experts, largest first, carry each fraction of the mass."""
    m = np.sort(mass)[::-1]
    c = np.cumsum(m) / m.sum()
    return [int(np.searchsorted(c, f) + 1) for f in fracs]


def skew(idx, w, ne=256):
    mass = np.bincount(idx.ravel(), weights=w.ravel(), minlength=ne)
    p = mass / mass.sum()
    nz = p[p > 0]
    return {"cover50_90_99": cover(mass), "top_quarter_share": float(np.sort(p)[::-1][:ne // 4].sum()),
            "unused": int((mass == 0).sum()), "eff_experts": float(np.exp(-(nz * np.log(nz)).sum()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.out, "manifest.json")))
    toks = np.load(os.path.join(a.out, "tokens.npy"))
    tk = None
    if a.tokenizer:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(a.tokenizer)
    res = {"sequences": {}, "layers": {}, "mtp": {}}

    fin = os.path.join(a.out, "final")
    if os.path.exists(os.path.join(fin, "argmax.npy")):
        am = np.load(os.path.join(fin, "argmax.npy"))
        nlp = np.load(os.path.join(fin, "next_logprob.npy"))
        print("== teacher-forced next-token prediction (main model)")
        for s in man["sequences"]:
            o, n = s["off"], s["n"]
            hit = am[o:o + n - 1] == toks[o + 1:o + n]
            lp = nlp[o:o + n - 1]
            r = {"n": n, "top1": float(hit.mean()), "mean_logprob": float(lp.mean()),
                 "ppl": float(np.exp(-lp.mean()))}
            # windows of 1024 positions
            r["top1_by_1024"] = [round(float(hit[i:i + 1024].mean()), 3) for i in range(0, n - 1, 1024)]
            res["sequences"][s["name"]] = r
            print(f"{s['name']:8s} n={n:6d} top1 {r['top1']:.3f}  mean logprob {r['mean_logprob']:.3f}  ppl {r['ppl']:.2f}")
            if tk is not None and s["name"] == "short":
                st = o + n // 2
                print("   greedy (argmax) vs actual from position", n // 2)
                print("   actual :", repr(tk.decode(toks[st + 1:st + 41].tolist())))
                print("   argmax :", repr(tk.decode(am[st:st + 40].tolist())))

    mt = os.path.join(a.out, "mtp")
    if os.path.exists(os.path.join(mt, "argmax.npy")):
        dam = np.load(os.path.join(mt, "argmax.npy"))
        rows = np.load(os.path.join(mt, "rows_main.npy"))
        am = np.load(os.path.join(fin, "argmax.npy"))
        print("== MTP draft (teacher-forced): row t drafts token t+2")
        moff = 0
        for s in man["sequences"]:
            o, n = s["off"], s["n"]
            m = n - 1
            d = dam[moff:moff + m - 1]                       # rows t = 0..n-3
            tgt = toks[o + 2:o + n]                          # token t+2
            main_next = am[o + 1:o + n - 1]                  # main argmax at row t+1 (its guess for t+2)
            r = {"vs_token": float((d == tgt).mean()), "vs_main_argmax": float((d == main_next).mean()),
                 "vs_token_when_main_right": float((d == tgt)[main_next == tgt].mean())}
            res["mtp"][s["name"]] = r
            print(f"{s['name']:8s} draft == token t+2: {r['vs_token']:.3f}   draft == main argmax at t+1: "
                  f"{r['vs_main_argmax']:.3f}   (== token | main right: {r['vs_token_when_main_right']:.3f})")
            moff += m

    print("== expert usage by routing mass (experts carrying 50/90/99 %; top-64 share; effective number)")
    tot = []
    for d in sorted(glob.glob(os.path.join(a.out, "layers", "L*"))):
        l = int(os.path.basename(d)[1:])
        t = json.load(open(os.path.join(d, "timing.json")))
        r = {"layer_s": t["layer_s"], "attention_s": t["attention_s"]}
        if os.path.exists(os.path.join(d, "route_idx.npy")):
            idx = np.load(os.path.join(d, "route_idx.npy"))
            w = np.load(os.path.join(d, "route_w.npy"))
            r["all"] = skew(idx, w)
            moff = 0
            for s in man["sequences"]:
                o, n = (s["off"], s["n"]) if l != 78 else (moff, s["n"] - 1)
                r[s["name"]] = skew(idx[o:o + n], w[o:o + n])
                moff += s["n"] - 1
            if l != 78 and len(man["sequences"]) > 1 and "agentic" in [s["name"] for s in man["sequences"]]:
                # transfer: the 64 experts with most mass on the other sequences, their share of agentic's mass
                ag = [s for s in man["sequences"] if s["name"] == "agentic"][0]
                sel = np.zeros(len(idx), bool)
                sel[ag["off"]:ag["off"] + ag["n"]] = True
                m_ag = np.bincount(idx[sel].ravel(), weights=w[sel].ravel(), minlength=256)
                m_ot = np.bincount(idx[~sel].ravel(), weights=w[~sel].ravel(), minlength=256)
                hot = np.argsort(m_ot)[::-1][:64]
                r["agentic_share_of_others_top64"] = float(m_ag[hot].sum() / m_ag.sum())
                r["agentic_share_of_own_top64"] = float(np.sort(m_ag)[::-1][:64].sum() / m_ag.sum())
            mg = np.load(os.path.join(d, "route_margin.npy"))
            r["route_margin_lt_1e-3"] = float((mg < 1e-3).mean())
            tot.append(r["all"]["top_quarter_share"])
            per = "  ".join(f"{s['name']} {r[s['name']]['cover50_90_99']}" for s in man["sequences"])
            print(f"L{l:02d} all {r['all']['cover50_90_99']} q {r['all']['top_quarter_share']:.3f} "
                  f"eff {r['all']['eff_experts']:.0f} unused {r['all']['unused']:3d} | {per}")
        for f in glob.glob(os.path.join(d, "topk_margin_*.npy")):
            mg = np.load(f)
            r.setdefault("index_ties", {})[os.path.basename(f)[12:-4]] = float((mg == 0).mean())
        res["layers"][l] = r
    tr = [(v["agentic_share_of_own_top64"], v["agentic_share_of_others_top64"]) for v in res["layers"].values()
          if "agentic_share_of_others_top64" in v]
    if tr:
        tr = np.array(tr)
        print(f"agentic mass on its own top-64 experts: mean {tr[:, 0].mean():.3f}; on the top-64 of the other "
              f"sequences: mean {tr[:, 1].mean():.3f} (min {tr[:, 1].min():.3f})")
    if tot:
        print(f"top-quarter share of routing mass: mean {np.mean(tot):.3f}, min {np.min(tot):.3f}, max {np.max(tot):.3f}"
              f" (uniform 0.25; quantisation.md § 3.4's hybrid threshold 0.40)")
    ls = [v["layer_s"] for k, v in res["layers"].items() if k != 78]
    if ls:
        print(f"timing: {len(ls)} layers, {sum(ls):.0f} s, median {np.median(ls):.1f} s a layer, "
              f"MTP {res['layers'].get(78, {}).get('layer_s', float('nan')):.1f} s")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
