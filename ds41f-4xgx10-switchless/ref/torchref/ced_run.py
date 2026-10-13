"""CED prefill reference: the tech report's Decoder SWA Bounded Replay (§2.2, §3.2.2) emulated around
the publisher's model.py, which has no CED code (its prefill runs all 40 layers over every prompt
position), and compared position by position with that exact forward.

CED as the report defines it, in model.py's terms (0-indexed layers, L/2 = 20, n_win = 128):
  * the encoder, layers 0-19 (Engram at 1 and 14 included), runs over every prompt position, exactly;
  * the decoder's global KV is layer 20's compressor (ratio 1, the only decoder kv source) and its
    index keys, both projected from layer 20's attention input = the encoder output; they are built
    for every prompt position and are bit-identical to the exact forward's;
  * the decoder, layers 20-39, runs only over the replay segment [s, P), s = max(0, P - R), R = n_win,
    with every decoder layer's sliding window truncated to it: a query at i sees SWA keys in
    [max(s, i-W+1), i]; global attention (layer 20's candidate pool, the Reindex layers' top-k) is
    unchanged and reaches the whole prompt;
  * decode then continues normally: the decoder rings hold the replay's keys for [P-128, P), and
    every later window [t-W+1, t] lies inside [s, t], so nothing is truncated any more.

Emulation: the decoder runs over the whole sequence, but with every query i >= s denied the window
keys before s. Positions >= s then see exactly what CED computes: their SWA is truncated, their
global KV comes from the encoder output (identical in both modes), and every other op is per token;
whatever the decoder computes at positions < s cannot reach them. Teacher-forced decode is the
same prefill extended past P, as in ref_run.py. One layer-streaming pass: the encoder runs once,
each decoder layer is loaded once and run for the exact stream and then for every CED stream.

  python3 ced_run.py --ckpt /models --tokens x.tokens.json [--length N] --out-dir /out/golden/ced/x \
      --stream ced128:P:128 [--stream NAME:P:R[:FLAGS] ...] [--iso NAME]

A stream NAME:P:R emulates a prompt of P tokens (P <= length) with an R-token replay; the tokens after
P are teacher-forced decode. Every stream's tensors cover positions [s, L), s = max(0, P - R); exact.*
covers [min s, L). Two yardsticks for what is noise:
  * --iso NAME adds a copy of stream NAME whose decoder rows before its s are scrambled after layer 20.
    Positions >= s are unchanged in exact arithmetic (the isolation the emulation relies on); what
    moves is MoE GEMM shapes, so iso vs NAME is the numerical noise of the CED stream.
  * FLAGS "exact,splitk": an exact forward (no truncation; P and R only set the range and segments)
    whose decoder GEMMs sum K in two halves (kernel.py's splitk), a numerics-only twin of exact.
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import ref_run as R  # sets up sys.path, our kernel.py, the chunked indexer; main() does not run

M = R.M
from engram import EngramLayout, NgramHashState  # noqa: E402

# ---------------------------------------------------------------------------------------------
# The window truncation: model.py's own prefill window, minus the keys before the replay start.

TRUNC = {"s": None}
_window_topk_idxs = M.get_window_topk_idxs


def truncated_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    idxs = _window_topk_idxs(window_size, bsz, seqlen, start_pos)
    s = TRUNC["s"]
    if not s:
        return idxs
    assert start_pos == 0
    q = torch.arange(seqlen, device=idxs.device).view(1, -1, 1)
    return torch.where((q >= s) & (idxs < s), -1, idxs).int().contiguous()


M.get_window_topk_idxs = truncated_window_topk_idxs


class Stream:
    def __init__(self, name, P, R_, twin=None, trunc=True, mode=""):
        self.name, self.P, self.R = name, P, R_
        self.s = 0 if P is None else max(0, P - R_)
        self.twin = twin
        self.trunc = trunc and P is not None  # False: an exact forward, saved and scored like a CED stream
        self.mode = mode                      # kernel.ACT_MODE in the decoder ("splitk": numerics twin)
        self.h = self.pm = self.topk = self.cand = self.kv = None
        self.mh = []                    # DSpark main_hidden pieces at [s, L)
        self.hdiff, self.kvdiff, self.trace = [], [], []
        self.l20 = {}


def rel_rows(a, b, chunk=2048):
    """Per-row relative L2 difference of a against b ([n, ...] each) -> [n] fp32 on the CPU."""
    out = []
    for c0 in range(0, a.size(0), chunk):
        x, y = a[c0:c0 + chunk].float().flatten(1), b[c0:c0 + chunk].float().flatten(1)
        out.append(((x - y).norm(dim=1) / y.norm(dim=1).clamp_min(1e-30)).cpu())
    return torch.cat(out)


def seg_stats(st_arr, P, s, L):
    """Agreement statistics of one stream against exact over position segments (arrays over [s, L))."""
    am_e, am_c, kl, nlp_e, nlp_c, nxt = (st_arr[k] for k in ("am_e", "am_c", "kl", "nlp_e", "nlp_c", "next"))
    segs = [("replay", s, min(P - 1, L - 1)), ("first", P - 1, P)]
    edges = [P, P + 128, P + 256, P + 512, P + 1024, P + 2048, P + 4096, P + 8192, P + 16384]
    for a, b in zip(edges, edges[1:]):
        if a < L - 1:
            segs.append((f"decode+{a - P}", a, min(b, L - 1)))
    out = {}
    for name, a, b in segs:
        if b <= a or a < s:
            continue
        i0, i1 = a - s, b - s
        d = (nlp_c[i0:i1] - nlp_e[i0:i1]) if i1 <= len(nlp_e) else None
        r = {"positions": [a, b], "n": b - a,
             "top1_agree": int((am_e[i0:i1] == am_c[i0:i1]).sum()),
             "kl_mean": float(kl[i0:i1].mean()), "kl_max": float(kl[i0:i1].max())}
        if d is not None and len(d):
            r.update({"next_lp_exact": float(nlp_e[i0:i1].mean()), "next_lp_ced": float(nlp_c[i0:i1].mean()),
                      "dlp_mean": float(d.mean()), "dlp_absmean": float(np.abs(d).mean()),
                      "dlp_absmax": float(np.abs(d).max()),
                      "argmax_eq_next_exact": int((am_e[i0:i1] == nxt[i0:i1]).sum()),
                      "argmax_eq_next_ced": int((am_c[i0:i1] == nxt[i0:i1]).sum())})
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/models")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--length", type=int, default=0, help="use only the first N tokens")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stream", action="append", default=[], help="NAME:P:R (prompt length, replay length)")
    ap.add_argument("--iso", default="", help="add an isolation-check copy of this stream")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--set", action="append", default=[], help="override a ModelArgs field (experiments)")
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    cfg = json.load(open(os.path.join(R.INF, "config.json")))
    tj = json.load(open(a.tokens))
    toks = tj["tokens"][: a.length] if a.length else tj["tokens"]
    L = len(toks)
    args = M.ModelArgs(**cfg)
    args.max_batch_size = 1
    args.max_seq_len = ((L + 255) // 256) * 256 + 256
    for kv in a.set:
        k, v = kv.split("=", 1)
        assert hasattr(args, k), k
        setattr(args, k, json.loads(v))
    M.world_size, M.rank, M.default_dtype = 1, 0, torch.float8_e4m3fn
    M.ParallelEngramEmbedding = R.NoEngramTable
    DEC0 = args.n_layers // 2
    # CED's decoder global KV is layer 20's: the candidate source, the decoder's only kv source, ratio 1
    assert DEC0 == args.candidate_source_layer and DEC0 in args.kv_source_layers
    assert args.compress_ratios[DEC0] == 1 and not [i for i in args.kv_source_layers if i > DEC0]
    assert all(i < DEC0 for i in args.engram_layer_ids)

    streams = [Stream("exact", None, None)]
    for spec in a.stream:
        name, P, R_, *fl = spec.split(":")
        fl = set(fl[0].split(",")) if fl else set()
        assert name != "exact" and 0 < int(P) <= L and fl <= {"exact", "splitk"}, spec
        streams.append(Stream(name, int(P), int(R_), trunc="exact" not in fl,
                              mode="splitk" if "splitk" in fl else ""))
    if a.iso:
        twin = next(st for st in streams if st.name == a.iso)
        assert twin.s > 0, "nothing to scramble before s = 0"
        streams.append(Stream(a.iso + "_iso", twin.P, twin.R, twin=twin, trunc=twin.trunc, mode=twin.mode))
    ex = streams[0]
    ex.s = min(st.s for st in streams[1:]) if len(streams) > 1 else 0
    prim = streams[1] if len(streams) > 1 else ex
    trace_pos = sorted({p for p in (prim.s, prim.s + 1, prim.s + 2, prim.s + 63, (prim.P or L) - 1, L - 1)
                        if prim.s <= p < L})

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    ck = R.CkptIndex(a.ckpt)
    layout = EngramLayout.from_args(args)
    W = R.Writer(a.out_dir)
    R.REC["positions"] = None
    W.put("tokens", np.array(toks, dtype=np.int32), "token ids of the whole sequence, [L]", "all")

    t0 = time.time()
    ids = torch.tensor([toks], dtype=torch.long)
    hashes = NgramHashState(args, layout, tok)(ids, 0, None)
    embed_w = ck.get("embed.weight")
    h = F.embedding(ids, embed_w)
    del embed_w
    h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, args.hc_mult)
    print(f"L={L} streams " + ", ".join(f"{st.name}(P={st.P}, s={st.s})" for st in streams), flush=True)

    def build(i):
        with torch.device("cuda"):
            blk = M.Block(i, args, layout)
        R.load_module(blk, ck, f"layers.{i}.")
        return blk

    for i in range(DEC0):  # the encoder, shared by every stream
        t1 = time.time()
        torch.cuda.reset_peak_memory_stats()
        blk = build(i)
        if blk.engram is not None:
            blk.engram.embed = R.MMapEngramEmbed(ck, i)
            h = blk.engram(h, hashes[:, :, blk.engram.layer_hash_index, :], None)
        h, pre_mix = blk(h, 0, pre_mix, None)
        del blk
        gc.collect()
        torch.cuda.empty_cache()
        print(f"layer {i:2d} {time.time()-t1:6.1f}s peak {torch.cuda.max_memory_allocated()/2**30:5.1f} GiB",
              flush=True)

    for st in streams:
        st.h, st.pm = h, pre_mix  # Block.forward never writes its inputs
    del h, pre_mix
    layer20 = {}
    base_mode = R.K.ACT_MODE
    for i in range(DEC0, args.n_layers):
        t1 = time.time()
        torch.cuda.reset_peak_memory_stats()
        blk = build(i)
        assert blk.engram is None
        cap = {}
        hook = blk.attn.kv_norm.register_forward_hook(lambda m, inp, out: cap.__setitem__("kv", out))
        for st in streams:
            if i in args.dspark_target_layer_ids:
                st.mh.append(st.h.mean(dim=2)[0, st.s:])
            # what layer 20 and the index layers hand down the stack is per stream: a numerics twin's
            # pool and top-k differ (a CED stream's pool is the exact one's, checked below)
            M.shared_attn.topk_idxs, M.shared_attn.candidates = st.topk, st.cand
            TRUNC["s"] = st.s if st.trunc else None
            R.K.ACT_MODE = st.mode or base_mode
            st.h, st.pm = blk(st.h, 0, st.pm, None)
            TRUNC["s"], R.K.ACT_MODE = None, base_mode
            st.topk, st.cand, st.kv = M.shared_attn.topk_idxs, M.shared_attn.candidates, cap.pop("kv")
            if i == DEC0 and st is not ex and torch.equal(st.cand, ex.cand):
                st.cand = ex.cand  # one copy: [L, L] bool is 0.8 GB at 28k
            if i == DEC0 and not st.mode:
                # the decoder's global KV, index keys, candidate pool and layer-20 top-k come from the
                # encoder output only: check they are bit-identical in every stream
                now = {"compress_kv": M.shared_attn.compress_kv[:, :L], "index_k": M.shared_attn.index_k[:, :L],
                       "candidates": st.cand, "topk": st.topk}
                if st is ex:  # the two caches are written in place by every stream's run
                    layer20 = {k: v.clone() if k in ("compress_kv", "index_k") else v for k, v in now.items()}
                else:
                    st.l20 = {k: bool(torch.equal(v, layer20[k])) for k, v in now.items()}
            if i == DEC0 and st.twin is not None:  # scramble the rows the emulation claims cannot matter
                s = st.s
                st.h = torch.cat([st.h[:, :s].flip(1), st.h[:, s:]], 1)
                st.pm = torch.cat([st.pm[:, :s].flip(1), st.pm[:, s:]], 1)
            if st is not ex:
                st.hdiff.append(rel_rows(st.h[0, st.s:], ex.h[0, st.s:]))
                st.kvdiff.append(rel_rows(st.kv[0, st.s:], ex.kv[0, st.s:]))
            if st is prim:
                st.trace.append(st.h[0, trace_pos].float().cpu())
        for st in streams:
            st.kv = None
        hook.remove()
        if i == DEC0:
            del layer20
        del blk, cap
        gc.collect()
        torch.cuda.empty_cache()
        worst = max((float(st.hdiff[-1].max()) for st in streams[1:]), default=0.0)
        print(f"layer {i:2d} {time.time()-t1:6.1f}s peak {torch.cuda.max_memory_allocated()/2**30:5.1f} GiB "
              f"max rel |h - h_exact| at >= s: {worst:.3e}", flush=True)
    M.shared_attn.topk_idxs = M.shared_attn.candidates = None
    for st in streams:
        st.topk = st.cand = None

    norm = M.RMSNorm(args.dim, args.norm_eps)
    norm.weight.data.copy_(ck.get("norm.weight"))
    head_w = ck.get("head.weight").float()
    for st in streams:
        st.xn = norm(M.Block.hc_pre(None, st.h, st.pm))
        st.h = st.pm = None
    ids_next = torch.tensor(toks[1:] + [0], dtype=torch.long)

    def logits(st, c0, c1):
        lg = F.linear(st.xn[0, c0:c1].float(), head_w)
        return lg, torch.logsumexp(lg, -1)

    # exact stream: top-k, lse, next-token logprob over [min s, L)
    topv, topi, lse, nlp = [], [], [], []
    for c0 in range(ex.s, L, 1024):
        c1 = min(L, c0 + 1024)
        lg, ls = logits(ex, c0, c1)
        v, ix = lg.topk(a.topk, dim=-1)
        topv.append(v.cpu()), topi.append(ix.cpu()), lse.append(ls.cpu())
        nlp.append((lg.gather(1, ids_next[c0:c1, None]).squeeze(1) - ls).cpu())
        del lg
    pspec = f"[{ex.s}, L)"
    W.put("exact.logits_topv", torch.cat(topv), f"exact: top-{a.topk} logits (fp32), [L-{ex.s}, {a.topk}]", pspec)
    W.put("exact.logits_topi", torch.cat(topi), "exact: token ids of logits_topv", pspec)
    W.put("exact.logits_lse", torch.cat(lse), "exact: logsumexp of the full logits", pspec)
    W.put("exact.next_logprob", torch.cat(nlp)[: L - 1 - ex.s], "exact: log-prob of the actual next token",
          f"[{ex.s}, L-1)")

    summary = {}
    for st in streams[1:]:
        acc = {k: [] for k in ("am_e", "am_c", "kl", "nlp_e", "nlp_c", "topv", "topi", "lse", "maxdiff")}
        for c0 in range(st.s, L, 1024):
            c1 = min(L, c0 + 1024)
            lg_c, ls_c = logits(st, c0, c1)
            lg_e, ls_e = logits(ex, c0, c1)
            lp_c, lp_e = lg_c - ls_c[:, None], lg_e - ls_e[:, None]
            nx = ids_next[c0:c1, None]
            acc["kl"].append((lp_e.exp() * (lp_e - lp_c)).sum(-1).cpu())
            acc["nlp_c"].append(lp_c.gather(1, nx).squeeze(1).cpu())
            acc["nlp_e"].append(lp_e.gather(1, nx).squeeze(1).cpu())
            acc["am_e"].append(lg_e.argmax(-1).cpu())
            v, ix = lg_c.topk(a.topk, dim=-1)
            acc["topv"].append(v.cpu()), acc["topi"].append(ix.cpu()), acc["lse"].append(ls_c.cpu())
            acc["am_c"].append(ix[:, 0].cpu())
            if st.twin is not None:
                lg_t, _ = logits(st.twin, c0, c1)
                acc["maxdiff"].append((lg_c - lg_t).abs().amax(-1).cpu())
            del lg_c, lg_e, lp_c, lp_e
        cat = {k: torch.cat(v) for k, v in acc.items() if v}
        n = L - st.s
        pspec = f"[{st.s}, L)"
        W.put(f"{st.name}.logits_topv", cat["topv"], f"{st.name}: top-{a.topk} logits (fp32), [L-s, {a.topk}]", pspec)
        W.put(f"{st.name}.logits_topi", cat["topi"], f"{st.name}: token ids of logits_topv", pspec)
        W.put(f"{st.name}.logits_lse", cat["lse"], f"{st.name}: logsumexp of the full logits", pspec)
        W.put(f"{st.name}.next_logprob", cat["nlp_c"][: n - 1], f"{st.name}: log-prob of the actual next token",
              f"[{st.s}, L-1)")
        W.put(f"{st.name}.kl", cat["kl"], f"{st.name}: KL(exact || {st.name}) of the next-token distribution", pspec)
        W.put(f"{st.name}.hdiff", torch.stack(st.hdiff),
              f"{st.name}: relative L2 distance of each decoder layer's output residual (all hc copies) from "
              f"exact, [{args.n_layers - DEC0} layers {DEC0}.., L-s]", pspec)
        W.put(f"{st.name}.kvdiff", torch.stack(st.kvdiff),
              f"{st.name}: relative L2 distance of each decoder layer's sliding-window KV as the ring stores "
              f"it (after RoPE and the fp8 rounding) from exact, [{args.n_layers - DEC0}, L-s]", pspec)
        mh_c = torch.cat(st.mh, -1)
        mh_e = torch.cat([t[st.s - ex.s:] for t in ex.mh], -1)
        W.put(f"{st.name}.main_hidden_diff", rel_rows(mh_c, mh_e),
              f"{st.name}: relative L2 distance of DSpark's main_hidden (layers 37-39 inputs) from exact", pspec)
        fpos = sorted({p for p in (st.P - 1, st.P) if p < L})
        W.put(f"{st.name}.full_positions", np.array(fpos, dtype=np.int32), f"{st.name}: positions of full_logits", "-")
        W.put(f"{st.name}.full_logits", torch.cat([logits(st, p, p + 1)[0].cpu() for p in fpos]),
              f"{st.name}: full fp32 logits at full_positions (P-1 = the first sampled token)", "full_positions")
        W.put(f"exact.full_logits_{st.name}", torch.cat([logits(ex, p, p + 1)[0].cpu() for p in fpos]),
              f"exact: full fp32 logits at {st.name}.full_positions", f"{st.name}.full_positions")
        if st is prim:
            W.put(f"{st.name}.main_hidden", mh_c[: max(0, min(st.P, L) - st.s)],
                  f"{st.name}: DSpark main_hidden over the replay [s, P) = what seeds DSpark's 128-slot window, "
                  f"[P-s, 3*dim]", f"[{st.s}, P)")
            e = min(L, st.P + 256)
            W.put(f"{st.name}.final_hidden", st.xn[0, st.s:e], f"{st.name}: final norm output (head input), "
                  f"[{e - st.s}, dim]", f"[{st.s}, {e})")
            W.put("trace_positions", np.array(trace_pos, dtype=np.int32), "positions of the trace", "-")
            W.put(f"{st.name}.trace", torch.stack(st.trace),
                  f"{st.name}: residual after each decoder layer at trace_positions, "
                  f"[{args.n_layers - DEC0}, len(trace_positions), hc, dim]", "trace_positions")
        stats = seg_stats({"am_e": cat["am_e"].numpy(), "am_c": cat["am_c"].numpy(), "kl": cat["kl"].numpy(),
                                 "nlp_e": cat["nlp_e"][: n - 1].numpy(), "nlp_c": cat["nlp_c"][: n - 1].numpy(),
                                 "next": ids_next[st.s: L - 1].cpu().numpy()}, min(st.P, L), st.s, L)
        hd = torch.stack(st.hdiff)
        summary[st.name] = {
            "P": st.P, "R": st.R, "s": st.s, "layer20_identical": st.l20,
            "first_token": {"exact": int(cat["am_e"][min(st.P, L) - 1 - st.s]),
                            "ced": int(cat["am_c"][min(st.P, L) - 1 - st.s])},
            "segments": stats,
            "hdiff_last_layer": {"at_s": float(hd[-1, 0]), "at_P-1": float(hd[-1, min(st.P, L) - 1 - st.s]),
                                 "max": float(hd[-1].max())},
            "kvdiff_replay_mean_per_layer": [round(float(r[: max(1, min(st.P, L) - st.s)].mean()), 5)
                                             for r in torch.stack(st.kvdiff)],
        }
        if st.twin is not None:
            summary[st.name]["isolation_max_abs_logit_diff"] = float(cat["maxdiff"].max())
        print(st.name, json.dumps(summary[st.name]["first_token"]), json.dumps(stats), flush=True)

    W.finish({"tokens_file": os.path.basename(a.tokens), "L": L, "length_arg": a.length,
              "streams": [{"name": st.name, "P": st.P, "R": st.R, "s": st.s, "truncated": st.trunc,
                           "decoder_numerics": st.mode or base_mode or "default"} for st in streams],
              "decoder_layers": [DEC0, args.n_layers], "window": args.window_size, "overrides": a.set,
              "seconds": round(time.time() - t0), "summary": summary,
              "notes": "CED emulation (see ced_run.py's docstring): stream tensors cover positions [s, L); "
                       "exact.* covers [min s, L); fp32 copies of bf16 tensors; next_logprob[p] scores "
                       "token p+1; KL is KL(exact || stream) in nats"})
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
