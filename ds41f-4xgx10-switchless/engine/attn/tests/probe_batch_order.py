"""Does the reference's own result depend on the batch size? (debug aid for --state-prefill caches)

Recomputes the compressor's latent and the index keys of a long case's prefill from the recorded
projections, once in one batch (as model.py's prefill) and once in 300-row slices (the shape of the
small cases, where our kernels match bit for bit), and counts the entries that differ.
  python3 probe_batch_order.py --case /out/long32k --layers 2,20
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ref", "torchref"))
sys.path.insert(1, os.path.join(ROOT, "ref", "inference"))
import kernel as K  # noqa: E402
import model as M  # noqa: E402
from ref_run import CkptIndex, load_module  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False


def load(case, name, dtype):
    a = np.load(os.path.join(case, name + ".npy"))
    t = torch.from_numpy(a.view(np.int16) if dtype == torch.bfloat16 else a).cuda()
    return t.view(torch.bfloat16) if dtype == torch.bfloat16 else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/models")
    ap.add_argument("--case", required=True)
    ap.add_argument("--layers", default="2,20")
    ap.add_argument("--slice", type=int, default=300)
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    cfg = json.load(open(os.path.join(ROOT, "ref", "inference", "config.json")))
    args = M.ModelArgs(**cfg)
    args.max_batch_size = 1
    args.max_seq_len = 65536
    M.world_size, M.rank, M.default_dtype = 1, 0, torch.float8_e4m3fn
    M.shared_attn = M.SharedAttentionRuntime()
    ck = CkptIndex(a.ckpt)
    for L in [int(v) for v in a.layers.split(",")]:
        with torch.device("cuda"):
            m = M.Attention(L, args)
        load_module(m, ck, f"layers.{L}.attn.")
        c, ix, ratio = m.compressor, m.indexer, m.compress_ratio
        with torch.inference_mode():
            if ratio == 1:
                pooled = load(a.case, f"L{L}_c0_c_proj", torch.bfloat16).unsqueeze(0)
            else:
                kv = load(a.case, f"L{L}_c0_c_kv", torch.float32).unsqueeze(0)
                sc = load(a.case, f"L{L}_c0_c_score", torch.float32).unsqueeze(0)
                kv, sc = kv.unflatten(1, (-1, ratio)), sc.unflatten(1, (-1, ratio))
                pooled = (kv * sc.softmax(dim=2)).sum(dim=2).to(torch.bfloat16)

            rd = m.rope_head_dim

            def chain(p, j0):
                lat = c.norm(p)
                k = ix.k_norm(ix.wk(lat))
                freqs = m.freqs_cis[j0 * ratio: (j0 + p.size(1)) * ratio: ratio]
                ent = lat.clone()
                M.apply_rotary_emb(ent[..., -rd:], freqs)
                K.fp4_act_quant(ent, 16, True, scale_dtype=torch.float8_e4m3fn)
                return lat, k, ent

            n = pooled.size(1)
            b = chain(pooled, 0)
            parts = [chain(pooled[:, i: i + a.slice], i) for i in range(0, n, a.slice)]
            s = [torch.cat([p[i] for p in parts], 1) for i in range(3)]
            for name, x, y in zip(("latent", "index key (pre-RoPE/quant)", "ckv entry"), b, s):
                rows = (x != y).any(-1)[0].nonzero().flatten().tolist()
                print(f"layer {L}: {n} entries; batch vs {a.slice}-row slices: {name} rows differing {len(rows)}"
                      f" {rows[:8]}", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
