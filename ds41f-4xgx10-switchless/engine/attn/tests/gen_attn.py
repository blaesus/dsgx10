"""Reference data for the attention-side kernels.

Runs the publisher's model.py attention (our pure-torch kernel.py underneath) on real layer weights
and seeded random inputs, for a stack of layers that share state the way the model does (a KV
source followed by its Reuse/Reindex consumers), over one prefill call and a few decode calls.
Every intermediate the engine's kernels produce is recorded, so each kernel can be checked alone.

Inside the canary-roce image (see gen.sh):
  python3 gen_attn.py --layers 2,3 --T 300 --decode 4 --out /out/l2
Tensors go to <out>/<name>.npy (bf16 stored as uint16 bit patterns), listed in <out>/manifest.json.
Names are L<layer>_c<call>_<what>; call 0 is the prefill, 1.. the decode steps.
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ref", "torchref"))
sys.path.insert(1, os.path.join(ROOT, "ref", "inference"))
import kernel as K  # noqa: E402  (torchref stand-in)
import model as M  # noqa: E402
from ref_run import CkptIndex, load_module  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
STATE_RECS = ("_c_kv", "_c_score", "_c_proj", "_latent", "_kv_pre_tail")  # what --state-prefill keeps of call 0
OWN_INDEX_K = False  # --own-index-k (see traced_indexer)


class Rec:
    def __init__(self, out):
        self.out, self.manifest = out, {}

    def __call__(self, name, t):
        t = t.detach()
        if t.dtype == torch.bfloat16:
            a, dt = t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16), "bf16"
        elif t.dtype == torch.float32:
            a, dt = t.contiguous().cpu().numpy(), "f32"
        elif t.dtype in (torch.int32, torch.int64):
            a, dt = t.contiguous().to(torch.int32).cpu().numpy(), "i32"
        elif t.dtype == torch.bool:
            a, dt = t.contiguous().to(torch.uint8).cpu().numpy(), "u8"
        else:
            raise TypeError(f"{name}: {t.dtype}")
        np.save(os.path.join(self.out, name + ".npy"), a)
        self.manifest[name] = {"dtype": dt, "shape": list(a.shape)}

    def save(self, extra):
        with open(os.path.join(self.out, "manifest.json"), "w") as f:
            json.dump({"tensors": self.manifest, **extra}, f, indent=1)


def traced_indexer(ix, x, qr, latent, start_pos, offset, rec, p, state_only=False):
    """Indexer.forward with recording; identical math. state_only: just the key cache (a prefill whose
    scores nobody reads, too big for the reference at long T)."""
    sa = M.shared_attn
    bsz, seqlen, _ = x.size()
    ratio, rd, end_pos = ix.compress_ratio, ix.rope_head_dim, start_pos + seqlen
    if ix.owns_k and latent is not None:
        freqs = (ix.freqs_cis[: seqlen - seqlen % ratio: ratio] if start_pos == 0
                 else ix.freqs_cis[start_pos + 1 - ratio].unsqueeze(0))
        kp = ix.wk(latent)
        rec(p + "ik_proj", kp[0])
        k = ix.k_norm(kp)
        rec(p + "ik_norm", k[0])
        M.apply_rotary_emb(k[..., -rd:], freqs)
        rec(p + "ik_rope", k[0])
        K.fp4_act_quant(k, M.fp4_block_size, True)
        rec(p + "ik", k[0])
        ix.k_cache[:bsz, start_pos // ratio: start_pos // ratio + k.size(1)] = k
        sa.index_k = ix.k_cache
    if ix.owns_k and OWN_INDEX_K:
        # model.py publishes the keys only when a group completed, so a ratio-2 source at an even
        # decode position scores against the last publisher's cache (layer 20's of the previous step)
        sa.index_k = ix.k_cache
    if state_only:
        return None
    q = ix.wq_b(qr).unflatten(-1, (ix.n_local_heads, ix.index_head_dim))
    rec(p + "iq_pre", q[0])
    M.apply_rotary_emb(q[..., -rd:], ix.freqs_cis[start_pos:end_pos])
    rec(p + "iq_rope", q[0])
    K.fp4_act_quant(q, M.fp4_block_size, True)
    rec(p + "iq", q[0])
    index_k = sa.index_k[:bsz, : end_pos // ratio]
    wp = ix.weights_proj(x)
    rec(p + "iw_proj", wp[0])
    weights = wp * (ix.softmax_scale * ix.n_heads ** -0.5)
    rec(p + "iw", weights[0])
    index_score = torch.einsum("bshd,btd->bsht", q, index_k)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    if start_pos == 0:
        compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
        index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
    else:
        compress_lens = end_pos // ratio
    rec(p + "iscore", index_score[0])
    if ix.is_candidate_source:
        sa.candidates = M.select_candidate_blocks(index_score, compress_lens, ix.candidate_topk_blocks,
                                                  ix.candidate_block_size)
        rec(p + "icand", sa.candidates[0])
    elif ix.uses_candidates:
        index_score = index_score.masked_fill(~sa.candidates, -torch.inf)
        rec(p + "iscore_masked", index_score[0])
    topk = min(ix.index_topk, end_pos // ratio)
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    out = torch.where(idxs < compress_lens, idxs + offset, -1).int()
    rec(p + "itopk", torch.where(idxs < compress_lens, idxs, -1).int()[0])
    return out


def traced_compressor(c, x, start_pos, rec, p):
    """Compressor.forward with recording; identical math."""
    bsz, seqlen, _ = x.size()
    ratio, dtype = c.compress_ratio, x.dtype
    if ratio == 1:
        y = c.wkv(x)
        rec(p + "c_proj", y[0])
        y = c.norm(y)
        rec(p + "latent", y[0])
        return y
    x = x.float()
    kv, score = c.wkv(x), c.wgate(x)
    rec(p + "c_kv", kv[0])
    rec(p + "c_score", score[0])
    if start_pos == 0:
        should_compress = seqlen >= ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        if remainder:
            kv, c.kv_state[:bsz, :remainder] = kv.split([cutoff, remainder], dim=1)
            score, c.score_state[:bsz, :remainder] = score.split([cutoff, remainder], dim=1)
        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
    else:
        should_compress = (start_pos + 1) % ratio == 0
        slot = start_pos % ratio
        c.kv_state[:bsz, slot] = kv.squeeze(1)
        c.score_state[:bsz, slot] = score.squeeze(1)
        if should_compress:
            kv = (c.kv_state[:bsz] * c.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
    if not should_compress:
        return None
    rec(p + "c_pooled", kv[0])
    y = c.norm(kv.to(dtype))
    rec(p + "latent", y[0])
    return y


def traced_attention(a, x, start_pos, rec, p, state_only=False):
    """Attention.forward with recording; identical math. state_only: only the caches (see traced_indexer),
    recording the projections the caches are built from (kv_pre_tail: the window's last 256 rows)."""
    sa = M.shared_attn
    bsz, seqlen, _ = x.size()
    freqs_cis = a.freqs_cis[start_pos: start_pos + seqlen]
    rd = a.rope_head_dim
    rec(p + "x", x[0])
    qr = q = None
    if not state_only:
        qa = a.wq_a(x)
        rec(p + "qa", qa[0])
        qr = a.q_norm(qa)
        rec(p + "qr", qr[0])
        q = a.wq_b(qr).unflatten(-1, (a.n_local_heads, a.head_dim))
        rec(p + "q_pre", q[0])
        M.apply_rotary_emb(q[..., -rd:], freqs_cis)
        rec(p + "q", q[0])
    # _window_kv
    win = a.window_size
    kvp = a.wkv(x)
    rec(p + "kv_pre", kvp[0])
    if state_only:
        rec(p + "kv_pre_tail", kvp[0, -256:])
    kv = a.kv_norm(kvp)
    rec(p + "kv_norm", kv[0])
    M.apply_rotary_emb(kv[..., -rd:], freqs_cis)
    rec(p + "kv_rope", kv[0])
    K.act_quant(kv, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    rec(p + "kv", kv[0])
    if start_pos == 0:
        if seqlen <= win:
            a.window_kv_cache[:bsz, :seqlen] = kv
        else:
            cutoff = seqlen % win
            a.window_kv_cache[:bsz, cutoff:win], a.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split(
                [win - cutoff, cutoff], dim=1)
        window_kv = kv
    else:
        a.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
        window_kv = a.window_kv_cache[:bsz]
    topk_idxs = M.get_window_topk_idxs(win, bsz, seqlen, start_pos)
    kv_all = window_kv
    if a.compress_ratio:
        # _compress_kv
        ratio = a.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if a.is_kv_source:
            latent = traced_compressor(a.compressor, x, start_pos, rec, p)
            sa.compress_kv = a.compress_kv_cache
        offset = window_kv.size(1)
        if not a.is_index_source:
            idxs = sa.topk_idxs
        elif compress_len == 0:
            idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        else:
            if a.indexer.freqs_cis is None:
                a.indexer.freqs_cis = a.freqs_cis
            idxs = traced_indexer(a.indexer, x, qr, latent, start_pos, offset, rec, p, state_only)
        sa.topk_idxs = idxs
        if latent is not None:
            freqs = (a.freqs_cis[: seqlen - seqlen % ratio: ratio] if start_pos == 0
                     else a.freqs_cis[start_pos + 1 - ratio].unsqueeze(0))
            M.apply_rotary_emb(latent[..., -rd:], freqs)
            rec(p + "ckv_rope", latent[0])
            K.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
            rec(p + "ckv", latent[0])
            a.compress_kv_cache[:bsz, start_pos // ratio: start_pos // ratio + latent.size(1)] = latent
        if state_only:
            return None
        compress_kv = sa.compress_kv[:bsz, :compress_len]
        kv_all = torch.cat([window_kv, compress_kv], dim=1)
        topk_idxs = torch.cat([topk_idxs, idxs], dim=-1)
    if state_only:
        return None
    rec(p + "attn_idx", topk_idxs[0])
    o = K.sparse_attn(q, kv_all, a.attn_sink, topk_idxs, a.softmax_scale)
    rec(p + "o_raw", o[0])
    M.apply_rotary_emb(o[..., -rd:], freqs_cis, True)
    rec(p + "o", o[0])
    o = o.view(bsz, seqlen, a.n_local_groups, -1)
    wo_a = a.wo_a.weight.view(a.n_local_groups, a.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return a.wo_b(o.flatten(2))


def make_x(ck, layer, n, seed):
    """An attention input with realistic scale: RMS-normalised, times the layer's attn_norm weight.
    A slowly drifting shared direction plus noise, so the scores have structure."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = ck.get(f"layers.{layer}.attn_norm.weight").float()
    base = torch.randn(2, 5120, generator=g, device="cuda", dtype=torch.float32)
    t = torch.linspace(0, 3, n, device="cuda", dtype=torch.float32).unsqueeze(1)
    h = torch.randn(n, 5120, generator=g, device="cuda", dtype=torch.float32)
    h = h + 2 * torch.cos(t) * base[0] + 2 * torch.sin(t) * base[1]
    h = h * torch.rsqrt(h.square().mean(-1, keepdim=True))
    return (h * w).to(torch.bfloat16).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/models")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--decode", type=int, default=3)
    ap.add_argument("--index-topk", type=int, default=0)
    ap.add_argument("--cand-blocks", type=int, default=0)
    ap.add_argument("--no-prefill-rec", action="store_true", help="record only decode calls + caches")
    ap.add_argument("--state-prefill", action="store_true",
                    help="call 0 only builds the caches (long T); records what they are rebuilt from")
    ap.add_argument("--own-index-k", action="store_true",
                    help="every key source scores its own keys (model.py reuses a stale slot at even decode "
                         "positions when several sources run; the engine does not)")
    ap.add_argument("--check", action="store_true", help="also run the untraced forward and compare")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global OWN_INDEX_K
    OWN_INDEX_K = a.own_index_k
    os.makedirs(a.out, exist_ok=True)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    cfg = json.load(open(os.path.join(ROOT, "ref", "inference", "config.json")))
    args = M.ModelArgs(**cfg)
    args.max_batch_size = 1
    args.max_seq_len = ((a.T + a.decode + 255) // 256) * 256 + 256
    if a.index_topk:
        args.index_topk = a.index_topk
    if a.cand_blocks:
        args.candidate_topk_blocks = a.cand_blocks
    M.world_size, M.rank, M.default_dtype = 1, 0, torch.float8_e4m3fn
    layers = [int(v) for v in a.layers.split(",")]
    ck = CkptIndex(a.ckpt)

    def build():
        mods = {}
        for L in layers:
            with torch.device("cuda"):
                m = M.Attention(L, args)
            load_module(m, ck, f"layers.{L}.attn.")
            mods[L] = m
        return mods

    T, D = a.T, a.decode
    xs = {L: make_x(ck, L, T + D, 1000 + L) for L in layers}
    calls = [(0, T)] + [(T + d, 1) for d in range(D)]

    ref_out = {}
    if a.check:
        M.shared_attn = M.SharedAttentionRuntime()
        mods = build()
        with torch.inference_mode():
            for ci, (sp, n) in enumerate(calls):
                for L in layers:
                    ref_out[(ci, L)] = mods[L](xs[L][:, sp: sp + n], sp).clone()
        del mods
        torch.cuda.empty_cache()

    rec = Rec(a.out)
    nop = lambda name, t: None  # noqa: E731
    M.shared_attn = M.SharedAttentionRuntime()
    mods = build()
    with torch.inference_mode():
        for ci, (sp, n) in enumerate(calls):
            for L in layers:
                r = nop if (ci == 0 and a.no_prefill_rec) else rec
                state_only = ci == 0 and a.state_prefill
                if state_only:  # just what the engine's caches are rebuilt from
                    r = lambda name, t: rec(name, t) if name.endswith(STATE_RECS) else None  # noqa: E731
                y = traced_attention(mods[L], xs[L][:, sp: sp + n], sp, r, f"L{L}_c{ci}_", state_only)
                if a.check:
                    same = torch.equal(y, ref_out[(ci, L)])
                    print(f"call {ci} layer {L}: traced == model.py: {same}", flush=True)
                    assert same
            if ci == 0:  # state after the prefill, so decode-only tests can start from it
                for L in layers:
                    m = mods[L]
                    rec(f"L{L}_win_cache", m.window_kv_cache[0])
                    if m.is_kv_source:
                        rec(f"L{L}_ckv_cache", m.compress_kv_cache[0, : T // m.compress_ratio])
                        if m.compressor.compress_ratio > 1:
                            rec(f"L{L}_c_kv_state", m.compressor.kv_state[0])
                            rec(f"L{L}_c_score_state", m.compressor.score_state[0])
                    if m.indexer is not None and m.indexer.owns_k:
                        rec(f"L{L}_ik_cache", m.indexer.k_cache[0, : T // m.compress_ratio])
    for L in layers:
        m = mods[L]
        rec(f"L{L}_attn_sink", m.attn_sink.detach().float())
        rec(f"L{L}_freqs", torch.view_as_real(m.freqs_cis[: T + D]).float())
        # the small weights the attention-side kernels own (bf16 as in the checkpoint)
        rec(f"L{L}_w_q_norm", m.q_norm.weight.detach())
        rec(f"L{L}_w_kv_norm", m.kv_norm.weight.detach())
        if m.compressor is not None:
            c = m.compressor
            rec(f"L{L}_w_c_wkv", c.wkv.weight.detach().to(torch.bfloat16))
            if c.compress_ratio > 1:
                rec(f"L{L}_w_c_wgate", c.wgate.weight.detach().to(torch.bfloat16))
            rec(f"L{L}_w_c_norm", c.norm.weight.detach())
        if m.indexer is not None:
            ix = m.indexer
            rec(f"L{L}_w_i_wproj", ix.weights_proj.weight.detach())
            if ix.owns_k:
                rec(f"L{L}_w_i_wk", ix.wk.weight.detach())
                rec(f"L{L}_w_i_knorm", ix.k_norm.weight.detach())
    rec.save({"layers": layers, "T": T, "decode": D, "calls": calls, "state_prefill": int(a.state_prefill),
              "own_index_k": int(a.own_index_k),
              "index_topk": args.index_topk,
              "cand_blocks": args.candidate_topk_blocks, "cand_block_size": args.candidate_block_size,
              "compress_ratios": [args.compress_ratios[L] for L in layers]})
    print("wrote", len(rec.manifest), "tensors to", a.out, flush=True)


if __name__ == "__main__":
    main()
