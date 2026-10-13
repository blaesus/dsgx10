"""Per-position anatomy of the two Engram layers on one token sequence: the hash rows (raw e4m3 bytes,
ue8m0 scale bytes, dequantised size), each row's share of the value, key and value sizes, the gate per
hc copy, and what the layer adds to the residual. Runs blocks 0..13 so that layer 14's gate sees its
real input; nothing after layer 14.

  golden/run_ref.sh py /work/golden/engram_probe.py --tokens /work/golden/x.tokens.json --upto 5000 \
      --out /out/engram_probe.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_argv, sys.argv = sys.argv, sys.argv[:1]
sys.path.insert(0, "/work/ref/torchref")
import ref_run as R  # noqa: E402

sys.argv = _argv
M, K = R.M, R.K


def deq_rows(w, s):
    """e4m3 row bytes [n, 256] and ue8m0 scale bytes [n, 8] -> fp32 [n, 256], as MMapEngramEmbed does."""
    w = torch.from_numpy(np.ascontiguousarray(w)).cuda().view(torch.float8_e4m3fn)
    s = torch.from_numpy(np.ascontiguousarray(s)).cuda().view(torch.float8_e8m0fnu)
    return (w.float().unflatten(-1, (-1, 32)) * s.float().unsqueeze(-1)).flatten(-2)


def byte_stats(w, s):
    """Raw-byte facts per row: e4m3fn NaN codes (S.1111.111), ue8m0 NaN (0xFF), the scale exponent
    range, and the dequantised size."""
    deq = deq_rows(w, s)
    e = s.astype(np.int32) - 127
    return {
        "nan_w": ((w & 0x7F) == 0x7F).sum(1), "scale_ff": (s == 0xFF).sum(1),
        "exp_min": e.min(1), "exp_max": e.max(1),
        "deq_max": deq.abs().amax(1).cpu().numpy(), "deq_norm": deq.norm(dim=1).cpu().numpy(),
    }


def row_stats(emb, idx):
    """byte_stats of the rows `idx` [L, 24] of one table, shaped like idx."""
    flat = idx.reshape(-1)
    st = byte_stats(np.asarray(emb.w[flat]), np.asarray(emb.s[flat]))
    return {k: v.reshape(idx.shape) for k, v in st.items()}


def table_sample(emb, primes, n=4096, seed=0):
    """byte_stats of `n` contiguous rows at a random place in each of the 24 sub-tables (one per
    n-gram order and head), the reference for what a typical row looks like: [24, n] each."""
    rng = np.random.default_rng(seed)
    offs = np.cumsum([0, *primes[:-1]])
    out = []
    for off, p in zip(offs, primes):
        a = int(off + rng.integers(0, p - n))
        out.append(byte_stats(np.asarray(emb.w[a:a + n]), np.asarray(emb.s[a:a + n])))
    return {k: np.stack([o[k] for o in out]) for k in out[0]}


def anatomy(eng, h, hash_ids):
    """The Engram forward of model.py taken apart; returns per-position arrays and the new h."""
    rows = eng.embed(hash_ids)  # [1, L, 24, 256] bf16
    kv = eng.wkv(rows.flatten(-2))
    key, value = kv.split([eng.hc_mult * eng.dim, eng.dim], dim=-1)
    key = key.float().unflatten(-1, (eng.hc_mult, eng.dim))
    weight = eng.q_weight.float() * eng.k_weight.float()
    hf = h.float()
    rstd = torch.rsqrt(hf.square().mean(-1) + eng.eps) * torch.rsqrt(key.square().mean(-1) + eng.eps)
    dot = (hf * weight * key).sum(-1) * rstd * eng.dim ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(eng.clamp_value).sqrt(), dot))
    out = eng(h, hash_ids, None)  # the real thing, to check the reconstruction
    delta = (out.float() - hf)[0]
    # each row's share of the value: wkv is linear in the rows (up to activation rounding)
    wv = K.dequant_fp8_weight(eng.wkv.weight, eng.wkv.scale, 32)[eng.hc_mult * eng.dim:]  # [dim, 24*256]
    r = rows[0].float()  # [L, 24, 256]
    share = torch.stack([(r[:, j] @ wv[:, j * 256:(j + 1) * 256].t()).norm(dim=-1) for j in range(r.size(1))], 1)
    vf = value[0].float()
    res = {
        "value_norm": vf.norm(dim=-1), "key_norm": key[0].norm(dim=-1), "dot": dot[0], "gate": gate[0],
        "resid_norm": hf[0].norm(dim=-1), "delta_norm": delta.flatten(1).norm(dim=-1),
        "recon_err": (delta - gate[0].unsqueeze(-1) * vf.unsqueeze(1)).flatten(1).norm(dim=-1),
        "row_share": share,
    }
    return {k: v.cpu().numpy() for k, v in res.items()}, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--upto", type=int, default=0, help="only the first N tokens (0 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    toks = json.load(open(a.tokens))["tokens"]
    toks = toks[: a.upto] if a.upto else toks
    L = len(toks)
    args = M.ModelArgs(**json.load(open(os.path.join(R.INF, "config.json"))))
    args.max_batch_size, args.max_seq_len = 1, ((L + 255) // 256) * 256 + 256
    M.world_size, M.rank, M.default_dtype = 1, 0, torch.float8_e4m3fn
    M.ParallelEngramEmbedding = R.NoEngramTable
    R.REC["positions"] = None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/models")
    ck = R.CkptIndex("/models")
    layout = R.EngramLayout.from_args(args)
    ids = torch.tensor([toks], dtype=torch.long)
    hashes = R.NgramHashState(args, layout, tok)(ids, 0, None)  # [1, L, 2, 24]
    h = torch.nn.functional.embedding(ids, ck.get("embed.weight"))
    h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, args.hc_mult)
    out = {"tokens": np.array(toks, dtype=np.int32), "hashes": hashes[0].cpu().numpy()}
    t0 = time.time()
    for i in range(max(args.engram_layer_ids) + 1):
        with torch.device("cuda"):
            blk = M.Block(i, args, layout)
        R.load_module(blk, ck, f"layers.{i}.")
        if blk.engram is not None:
            emb = R.MMapEngramEmbed(ck, i)
            blk.engram.embed = emb
            hid = hashes[:, :, blk.engram.layer_hash_index, :]
            parts, h = anatomy(blk.engram, h, hid)
            for k, v in parts.items():
                out[f"L{i:02d}.{k}"] = v
            for k, v in row_stats(emb, hid[0].cpu().numpy()).items():
                out[f"L{i:02d}.row_{k}"] = v
            primes = [p for per_ngram in layout.primes[blk.engram.layer_hash_index] for p in per_ngram]
            for k, v in table_sample(emb, primes).items():
                out[f"L{i:02d}.sample_{k}"] = v
            print(f"layer {i}: engram anatomy done ({time.time()-t0:.0f}s), max recon err "
                  f"{parts['recon_err'].max():.3g}", flush=True)
        if i < max(args.engram_layer_ids):
            h, pre_mix = blk(h, 0, pre_mix, None)
        del blk
        torch.cuda.empty_cache()
        print(f"layer {i} done {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(a.out, **out)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
