#!/usr/bin/env python3
"""DSpark reference data: the publisher's model.py draft (Transformer.forward_spec, unchanged) driven over a
golden run's saved main hiddens, with torchref's kernel stand-ins, at world size 1 on one GB10.

model.py has no speculative loop (nothing calls forward_spec); its __main__ shows the protocol, which this
follows. A prefill call (start_pos 0) seeds each block's sliding-window KV with main_kv of the prompt
positions; then one call per decoded position i, forward_spec(token at i+1, main_hidden[i], i), writes
main_kv of position i into the window and drafts a block at positions i+1..i+5 (that token, then four
noise tokens). Greedy: temperature 0 (model.py samples the drafts at the model's temperature otherwise).

The blocks run one at a time over every step: a block's state (its window KV) depends only on main_x, so
only one block's weights are resident. Every step's intermediates are saved (fp32 copies of bf16 values).

  OUT=/data/dsgb10 golden/run_ref.sh py /work/engine/dspark/ref/dspark_ref.py \
      --golden /out/golden/prose --out-dir /out/dspark/prose
"""
import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/work/ref/torchref")
import ref_run as R  # noqa: E402  (puts torchref's kernel.py ahead of the publisher's)

M = R.M
CUR = {}  # this call's captures, by name


def rec(key, v):
    CUR.setdefault(key, []).append(v)


def capture(name, take_input=False, keep=False):
    def hook(mod, inp, out):
        v = inp[0] if take_input else out
        rec(name, v.detach().clone())
        if keep:  # the tensor itself: model.py edits it in place afterwards (RoPE, act_quant)
            rec(name + "#obj", v)
    return hook


def instrument(blk):
    a = blk.attn
    hs = [
        blk.attn_norm.register_forward_hook(capture("attn_in")),
        blk.ffn_norm.register_forward_hook(capture("ffn_in")),
        a.wq_a.register_forward_hook(capture("qa")),
        a.q_norm.register_forward_hook(capture("qr")),
        a.wq_b.register_forward_hook(capture("q_pre")),
        a.wkv.register_forward_hook(capture("wkv")),
        a.kv_norm.register_forward_hook(capture("kv", keep=True)),
        a.wo_b.register_forward_hook(capture("woa_out", take_input=True)),
        a.wo_b.register_forward_hook(capture("attn_out")),
        blk.ffn.shared_experts.register_forward_hook(capture("shared_out")),
        blk.ffn.register_forward_hook(capture("ffn_out")),
    ]

    def gate_hook(mod, inp, out):
        rec("route_w", out[0].detach().clone())
        rec("route_idx", out[1].detach().clone())

    hs.append(blk.ffn.gate.register_forward_hook(gate_hook))
    mixes0, post0 = blk.hc_mixes, blk.hc_post

    def mixes(x, fn, sc, base):
        r = mixes0(x, fn, sc, base)
        rec("mix", tuple(t.clone() for t in r))
        return r

    def post(x, res, p, c):
        y = post0(x, res, p, c)
        rec("hc_post", y.clone())
        return y

    blk.hc_mixes, blk.hc_post = mixes, post
    return hs


_sparse_attn = M.sparse_attn


def sparse_attn(q, kv, sink, idx, scale, *rest, **kw):  # keeps the output: the inverse RoPE edits it in place
    o = _sparse_attn(q, kv, sink, idx, scale, *rest, **kw)
    rec("attn_o#obj", o)
    return o


M.sparse_attn = sparse_attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/models")
    ap.add_argument("--golden", required=True, help="a golden run's directory (main_hidden.npy, tokens.npy)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--first", type=int, default=-1, help="first step's position (default: the last prompt one)")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--full", default="0,1,48,95", help="steps whose full logits and Markov bias are saved")
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    t0 = time.time()
    cfg = json.load(open(os.path.join(R.INF, "config.json")))
    man = json.load(open(os.path.join(a.golden, "manifest.json")))
    L, n_prompt = man["L"], man["n_prompt"]
    assert man["positions"] == "all", "needs main_hidden at every position"
    toks = np.load(os.path.join(a.golden, "tokens.npy")).astype(np.int64)
    hid = torch.from_numpy(np.load(os.path.join(a.golden, "main_hidden.npy"))).cuda().to(torch.bfloat16)
    args = M.ModelArgs(**cfg)
    args.max_batch_size = 1
    args.max_seq_len = ((L + 8 + 255) // 256) * 256
    args.temperature = 0.0
    M.world_size, M.rank, M.default_dtype = 1, 0, torch.float8_e4m3fn
    nb, dim = args.dspark_block_size, args.dim
    first = n_prompt - 1 if a.first < 0 else a.first
    steps = list(range(first, L - 1))  # step i drafts from the token at i + 1
    S = len(steps)
    full = [int(s) for s in a.full.split(",") if s and int(s) < S]
    ck = R.CkptIndex(a.ckpt)
    W = R.Writer(a.out_dir)
    tok_dev = torch.from_numpy(toks).cuda()
    W.put("steps", np.array(steps, dtype=np.int32), "each step's position i: main_hidden[i] joins the window, "
          "the block sits at i+1..i+5, [S]", "-")
    W.put("tokens", toks.astype(np.int32), "the golden run's tokens, [L]; step i's block starts with tokens[i+1]", "all")

    # torch's order for the sum of 3 routing weights (the engine router writes it out)
    sums = {"(0+1)+2": 0, "(0+2)+1": 0, "0+(1+2)": 0}
    for _ in range(400):
        w = torch.rand(nb, 3, dtype=torch.float32) * 3
        s = w.sum(dim=-1, keepdim=True)[:, 0]
        sums["(0+1)+2"] += int(((w[:, 0] + w[:, 1]) + w[:, 2] == s).all())
        sums["(0+2)+1"] += int(((w[:, 0] + w[:, 2]) + w[:, 1] == s).all())
        sums["0+(1+2)"] += int((w[:, 0] + (w[:, 1] + w[:, 2]) == s).all())
    print("3-value sum order (of 400 trials):", sums, flush=True)

    # ---- stage 0's main_proj/main_norm and the shared embedding: block inputs and main_x
    def build(stage):
        with torch.device("cuda"):
            blk = M.DSparkBlock(args.n_layers + stage, args)
        R.load_module(blk, ck, f"mtp.{stage}.")
        return blk

    blk = build(0)
    emb = M.ParallelEmbedding(args.vocab_size, dim)
    emb.weight.data.copy_(ck.get("embed.weight"))
    blk.embed = emb
    pre_seed, main_x_seed = blk.forward_embed(hid[None, :first + 1], tok_dev[first + 1:first + 2])
    x_in, main_x = [], []
    for i in steps:
        x, mx = blk.forward_embed(hid[None, i:i + 1], tok_dev[i + 1:i + 2])
        x_in.append(x)
        main_x.append(mx)
    blk.embed = None
    del emb
    W.put("main_x_seed", main_x_seed[0], f"main_norm(main_proj(main_hidden)) of positions 0..{first} as the "
          "prefill call computes them (one call), [first+1, dim]", "0..first")
    W.put("main_x", torch.cat(main_x)[:, 0], "main_x of each step's position, one call per step, [S, dim]", "steps")
    W.put("draft_ids_in", torch.full((S, nb), args.dspark_noise_token_id).index_put_(
        (torch.arange(S), torch.zeros(S, dtype=torch.long)), tok_dev[torch.tensor(steps) + 1]),
        "the block's input ids: the token at i+1, then the noise token, [S, 5]", "steps")

    pre_mix0 = M.make_identity_pre_mix(x_in[0], args.hc_mult)
    pre_mix = [pre_mix0] * S
    meaning = {
        "stream_in": "block input stream, [S, 5, hc, dim]",
        "attn_in": "attention input: hc_pre with the previous pre_mix, then attn_norm, [S, 5, dim]",
        "qa": "wq_a output, [S, 5, 1280]", "qr": "q_norm output, [S, 5, 1280]",
        "q_pre": "wq_b output before RoPE, [S, 5, 64*512]",
        "main_wkv": "wkv(main_x) of the step's position, [S, 512]",
        "main_kv": "the window entry written for the step's position: kv_norm, RoPE, fp8 round trip, [S, 512]",
        "blk_wkv": "wkv of the block rows, [S, 5, 512]",
        "blk_kv": "the block rows' keys: kv_norm, RoPE at i+1..i+5, fp8 round trip, [S, 5, 512]",
        "attn_o": "sparse_attn output after the inverse RoPE (wo_a's input), [S, 5, 64, 512]",
        "woa_out": "wo_a output (wo_b's input), [S, 5, 8192]",
        "attn_out": "wo_b output, [S, 5, dim]",
        "mid": "stream after the attention sublayer's hc_post, [S, 5, hc, dim]",
        "ffn_in": "MoE input: hc_pre with this block's attention pre, then ffn_norm, [S, 5, dim]",
        "route_idx": "chosen experts in top-k order, [S, 5, 3]", "route_w": "their weights (x route_scale), [S, 5, 3]",
        "shared_out": "shared expert output, [S, 5, dim]", "ffn_out": "MoE output incl. shared expert, [S, 5, dim]",
        "out": "block output stream, [S, 5, hc, dim]",
        "attn_pre": "hc mixes of the attention sublayer (pre used by this block's FFN), [S, 5, hc]",
        "attn_post": "[S, 5, hc]", "attn_comb": "[S, 5, hc, hc]",
        "ffn_pre": "hc mixes of the FFN sublayer (pre = the pre_mix handed on), [S, 5, hc]",
        "ffn_post": "[S, 5, hc]", "ffn_comb": "[S, 5, hc, hc]",
    }
    for stage in range(args.n_mtp_layers):
        t1 = time.time()
        if stage > 0:
            blk = build(stage)
        hooks = instrument(blk)
        CUR.clear()
        blk(pre_seed, 0, pre_mix0, main_x_seed)  # seeds the window with positions 0..first
        W.put(f"B{stage}.main_kv_seed", CUR["kv#obj"][0][0], "window entries of the prefill call's positions, "
              "[first+1, 512]", "0..first")
        acc = {k: [] for k in meaning}
        for s, i in enumerate(steps):
            CUR.clear()
            x, pm = blk(x_in[s], i, pre_mix[s], main_x[s])
            acc["stream_in"].append(x_in[s][0])
            for k in ("attn_in", "qa", "qr", "attn_out", "woa_out", "ffn_in", "ffn_out"):
                acc[k].append(CUR[k][0][0])
            acc["shared_out"].append(CUR["shared_out"][0])  # the MoE runs its experts on [rows, dim]
            acc["q_pre"].append(CUR["q_pre"][0][0])
            acc["main_wkv"].append(CUR["wkv"][0][0, 0])
            acc["blk_wkv"].append(CUR["wkv"][1][0])
            acc["main_kv"].append(CUR["kv#obj"][0][0, 0].clone())
            acc["blk_kv"].append(CUR["kv#obj"][1][0].clone())
            acc["attn_o"].append(CUR["attn_o#obj"][0][0].clone())
            acc["route_idx"].append(CUR["route_idx"][0].view(nb, -1))
            acc["route_w"].append(CUR["route_w"][0].view(nb, -1))
            acc["mid"].append(CUR["hc_post"][0][0])
            acc["out"].append(CUR["hc_post"][1][0])
            for j, sub in enumerate(("attn", "ffn")):
                p, q, c = CUR["mix"][j]
                acc[f"{sub}_pre"].append(p[0])
                acc[f"{sub}_post"].append(q[0])
                acc[f"{sub}_comb"].append(c[0])
            x_in[s], pre_mix[s] = x, pm
        for h in hooks:
            h.remove()
        for k, v in acc.items():
            W.put(f"B{stage}.{k}", torch.stack(v), meaning[k], "steps")
        del acc
        if stage < args.n_mtp_layers - 1:
            del blk
            gc.collect()
            torch.cuda.empty_cache()
        print(f"block {stage}: {time.time() - t1:.1f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
              flush=True)

    # ---- the head: hc_pre with the last pre_mix, norm, the shared vocab head, the Markov chain, confidence
    head = M.ParallelHead(args.vocab_size, dim, args.norm_eps, args.hc_eps)
    head.weight.data.copy_(ck.get("head.weight"))
    blk.head = head
    hx, hxn, lg_v, lg_i, lse, mk_e, b_v, b_i, d_v, d_i, ids, conf, f_lg, f_b = ([] for _ in range(14))

    def norm_hook(mod, inp, out):
        rec("hx", inp[0].detach().clone())
        rec("hxn", out.detach().clone())

    def head_hook(mod, inp, out):
        if mod is head:
            rec("logits", out.detach().clone())

    def markov_hook(mod, inp, out):
        rec("bias", out[0].detach().clone())
        rec("membed", out[1].detach().clone())

    hooks = [blk.norm.register_forward_hook(norm_hook), head.register_forward_hook(head_hook),
             blk.markov_head.register_forward_hook(markov_hook)]
    k = a.topk
    for s, i in enumerate(steps):
        CUR.clear()
        out_ids, logits, c = blk.forward_head(x_in[s], pre_mix[s], tok_dev[i + 1:i + 2])
        lg = CUR["logits"][0][0]  # [5, V] before the Markov bias
        bias = torch.cat(CUR["bias"])  # [5, V]
        hx.append(CUR["hx"][0][0])
        hxn.append(CUR["hxn"][0][0])
        v, ix = lg.topk(k, dim=-1)
        lg_v.append(v), lg_i.append(ix), lse.append(torch.logsumexp(lg, -1))
        v, ix = bias.topk(k, dim=-1)
        b_v.append(v), b_i.append(ix)
        mk_e.append(torch.cat(CUR["membed"]))
        v, ix = logits[0].topk(k, dim=-1)
        d_v.append(v), d_i.append(ix)
        ids.append(out_ids[0])
        conf.append(c[0])
        if s in full:
            f_lg.append(lg)
            f_b.append(bias)
    for h in hooks:
        h.remove()
    W.put("head_x", torch.stack(hx), "hc_pre(last block's stream, its pre_mix): the norm's input and the confidence "
          "head's hidden, [S, 5, dim]", "steps")
    W.put("head_xn", torch.stack(hxn), "after the final norm (mtp.2.norm): the vocab head's input, [S, 5, dim]", "steps")
    W.put("logits_topv", torch.stack(lg_v), f"top-{k} fp32 logits before the Markov bias, [S, 5, {k}]", "steps")
    W.put("logits_topi", torch.stack(lg_i), "their ids", "steps")
    W.put("logits_lse", torch.stack(lse), "logsumexp of the unbiased logits, [S, 5]", "steps")
    W.put("markov_embed", torch.stack(mk_e), "Markov embedding of the previous token (the input token, then each "
          "draft), [S, 5, 256]", "steps")
    W.put("bias_topv", torch.stack(b_v), f"top-{k} Markov biases, [S, 5, {k}]", "steps")
    W.put("bias_topi", torch.stack(b_i), "their ids", "steps")
    W.put("draft_topv", torch.stack(d_v), f"top-{k} biased logits (what the draft takes the argmax of), [S, 5, {k}]",
          "steps")
    W.put("draft_topi", torch.stack(d_i), "their ids", "steps")
    W.put("draft_ids", torch.stack(ids), "output_ids: the input token, then the 5 drafts, [S, 6]", "steps")
    W.put("confidence", torch.stack(conf), "confidence head output (fp32, no sigmoid), [S, 5]", "steps")
    if full:
        W.put("full_logits", torch.stack(f_lg), "unbiased logits of the steps in full_steps, [F, 5, V]", "full_steps")
        W.put("full_bias", torch.stack(f_b), "Markov biases of the same, [F, 5, V]", "full_steps")
        W.put("full_steps", np.array(full, dtype=np.int32), "step indices of full_logits / full_bias", "-")

    # ---- acceptance against the golden sequence, and what the confidence head predicts
    d = torch.stack(ids)[:, 1:].cpu().numpy()
    cf = torch.stack(conf).float().cpu().numpy()
    topi = np.load(os.path.join(a.golden, "logits_topi.npy"))[:, 0]
    accepted, cond, cum = [], [[] for _ in range(nb)], [[] for _ in range(nb)]
    for s, i in enumerate(steps):
        n = 0
        while n < nb and i + 2 + n < L and d[s, n] == toks[i + 2 + n]:
            n += 1
        accepted.append(n)
        for j in range(nb):
            if i + 2 + j >= L:
                break
            p = 1 / (1 + math.exp(-cf[s, j]))
            cum[j].append((p, int(n > j)))
            if n >= j:
                cond[j].append((p, int(n > j)))
    ref_ok = [int(d[s, 0] == topi[i + 1]) for s, i in enumerate(steps)]

    def stats(pairs):
        if not pairs:
            return None
        p = np.array([x for x, _ in pairs])
        y = np.array([t for _, t in pairs])
        return {"n": len(pairs), "rate": round(float(y.mean()), 3), "mean_sigmoid": round(float(p.mean()), 3),
                "brier": round(float(((p - y) ** 2).mean()), 4)}

    summary = {
        "accepted_hist": np.bincount(accepted, minlength=nb + 1).tolist(),
        "mean_accepted": round(float(np.mean(accepted)), 3),
        "draft1_vs_ref_argmax": [sum(ref_ok), S],
        "conditional": [stats(c) for c in cond],
        "cumulative": [stats(c) for c in cum],
    }
    W.put("accepted", np.array(accepted, dtype=np.int32), "leading drafts equal to the golden tokens, [S]", "steps")
    W.finish({"golden": a.golden, "L": L, "n_prompt": n_prompt, "first": first, "S": S, "temperature": 0,
              "block_size": nb, "noise_token": args.dspark_noise_token_id, "sum3_order_trials": sums,
              "seconds": round(time.time() - t0), "summary": summary,
              "notes": "fp32 copies of bf16 tensors; S = steps (steps.npy); B<n> = DSpark block n (mtp.n)"})
    print(json.dumps(summary, indent=1), flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
