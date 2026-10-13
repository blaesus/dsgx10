"""Read engram_probe.py's npz: where the two Engram layers add unusually much to the residual, which
n-grams and table rows do it, and whether those rows look damaged (NaN codes, extreme scales, sizes
against a random sample of the same sub-table).

  golden/run_ref.sh py /work/golden/engram_report.py /out/probe5120.npz [--window 4490-4545] [--top 12]
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--window", default="4490-4545")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--tokenizer", default="/models")
    a = ap.parse_args()
    d = np.load(a.npz)
    toks = d["tokens"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    def ngram(p, n=4):
        return repr(tok.decode(toks[max(0, p - n + 1): p + 1].tolist()))[:40]

    w0, w1 = (int(v) for v in a.window.split("-"))
    for layer in (1, 14):
        k = f"L{layer:02d}."
        dn, vn, gate, rn = d[k + "delta_norm"], d[k + "value_norm"], d[k + "gate"], d[k + "resid_norm"]
        rel = dn / rn.mean(1)
        share = d[k + "row_share"]  # [L, 24]
        rows = d["hashes"][:, (1, 14).index(layer)]
        med = np.median(dn)
        print(f"== layer {layer}: |delta| median {med:.3g} p90 {np.percentile(dn, 90):.3g} "
              f"p99 {np.percentile(dn, 99):.3g} max {dn.max():.3g} at {dn.argmax()}; |delta|/|h| median "
              f"{np.median(rel):.3g} max {rel.max():.3g} at {rel.argmax()}; gate mean {gate.mean():.3f}; "
              f"reconstruction err max {d[k + 'recon_err'].max():.3g}")
        # the rows: damage codes and sizes against the per-sub-table sample
        sdn = d[k + "sample_deq_norm"]  # [24, n]
        lo, hi = np.percentile(sdn, 0.1, axis=1), np.percentile(sdn, 99.9, axis=1)
        rdn = d[k + "row_deq_norm"]  # [L, 24]
        print(f"   hashed rows: NaN e4m3 codes {int(d[k + 'row_nan_w'].sum())}, 0xFF scales "
              f"{int(d[k + 'row_scale_ff'].sum())}, scale exponents {d[k + 'row_exp_min'].min()}..{d[k + 'row_exp_max'].max()} "
              f"(sample {d[k + 'sample_exp_min'].min()}..{d[k + 'sample_exp_max'].max()}, sample NaN codes "
              f"{int(d[k + 'sample_nan_w'].sum())}, 0xFF {int(d[k + 'sample_scale_ff'].sum())}); row norms outside the "
              f"sample's 0.1-99.9% band: {int((rdn < lo).sum())} low, {int((rdn > hi).sum())} high of {rdn.size}; "
              f"all-zero rows {int((rdn == 0).sum())} (sample {int((sdn == 0).sum())} of {sdn.size})")
        print(f"   sample row norm by order (2,3,4-gram): "
              + ", ".join(f"{np.median(sdn[o * 8:(o + 1) * 8]):.3g}" for o in range(3))
              + "; hashed: " + ", ".join(f"{np.median(rdn[:, o * 8:(o + 1) * 8]):.3g}" for o in range(3)))
        print(f"   top {a.top} positions by |delta|:")
        for p in np.argsort(-dn)[: a.top]:
            j = share[p].argmax()
            print(f"   {p:6d} {ngram(p):42s} |d| {dn[p]:7.3g} ({dn[p] / med:5.1f}x) |v| {vn[p]:7.3g} "
                  f"gate {np.round(gate[p], 2)} top row col {j} id {rows[p, j]} share {share[p, j]:.3g} "
                  f"(|row| {rdn[p, j]:.3g}, sample p99.9 {hi[j]:.3g})")
        print(f"   window {w0}-{w1}: |delta|/median per position")
        line = []
        for p in range(w0, min(w1, len(dn))):
            line.append(f"{p}:{dn[p] / med:.1f}")
        for i in range(0, len(line), 11):
            print("     " + " ".join(line[i:i + 11]))


if __name__ == "__main__":
    main()
