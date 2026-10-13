"""Reference data for the DSpark draft attention (model.py DSparkAttention on mtp.<s>.attn weights).

A seeded main stream seeds the stage's window (start_pos 0, T positions); then D decode steps: the main
token at start_pos = T + d writes its window entry and a block of 5 draft rows at start_pos+1..+5
attends to the last 128 main positions plus the whole block (get_dspark_topk_idxs).

Inside the canary-roce image (see gen.sh):
  python3 gen_dspark.py --stages 0,2 --T 200 --decode 3 --out /out/dspark
Names: S<stage>_c<call>_<what> (call 0 = the seeding prefill), S<stage>_<what>; manifest as gen_attn.py.
"""
import argparse, json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ref", "torchref"))
sys.path.insert(1, os.path.join(ROOT, "ref", "inference"))
import kernel as K  # noqa: E402  (torchref stand-in)
import model as M  # noqa: E402
from ref_run import CkptIndex, load_module  # noqa: E402
from gen_attn import Rec  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False


def make_x(ck, norm_key, n, seed):
    """RMS-normalised rows times a norm weight, with a slowly drifting shared direction (as gen_attn)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = ck.get(norm_key).float()
    base = torch.randn(2, 5120, generator=g, device="cuda", dtype=torch.float32)
    t = torch.linspace(0, 3, n, device="cuda", dtype=torch.float32).unsqueeze(1)
    h = torch.randn(n, 5120, generator=g, device="cuda", dtype=torch.float32)
    h = h + 2 * torch.cos(t) * base[0] + 2 * torch.sin(t) * base[1]
    h = h * torch.rsqrt(h.square().mean(-1, keepdim=True))
    return (h * w).to(torch.bfloat16).unsqueeze(0)


def traced_dspark(a, x, start_pos, main_x, rec, p):
    """DSparkAttention.forward with recording; identical math."""
    bsz, seqlen, _ = main_x.size()
    win, rd = a.window_size, a.rope_head_dim
    main_freqs_cis = a.freqs_cis[start_pos: start_pos + seqlen]
    mkv = a.wkv(main_x)
    rec(p + "main_kv_pre", mkv[0])
    main_kv = a.kv_norm(mkv)
    M.apply_rotary_emb(main_kv[..., -rd:], main_freqs_cis)
    K.act_quant(main_kv, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    rec(p + "main_kv", main_kv[0])
    if start_pos == 0:
        if seqlen <= win:
            a.window_kv_cache[:bsz, :seqlen] = main_kv
        else:
            cutoff = seqlen % win
            a.window_kv_cache[:bsz, cutoff:win], a.window_kv_cache[:bsz, :cutoff] = main_kv[:, -win:].split(
                [win - cutoff, cutoff], dim=1)
        return x
    bsz, block_size, _ = x.size()
    freqs_cis = a.freqs_cis[start_pos + seqlen: start_pos + seqlen + block_size]
    qr = a.q_norm(a.wq_a(x))
    q = a.wq_b(qr).unflatten(-1, (a.n_local_heads, a.head_dim))
    rec(p + "q_pre", q[0])
    M.apply_rotary_emb(q[..., -rd:], freqs_cis)
    rec(p + "q", q[0])
    kvp = a.wkv(x)
    rec(p + "kv_pre", kvp[0])
    kv = a.kv_norm(kvp)
    M.apply_rotary_emb(kv[..., -rd:], freqs_cis)
    K.act_quant(kv, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    rec(p + "kv", kv[0])
    topk_idxs = M.get_dspark_topk_idxs(win, bsz, block_size, start_pos)
    a.window_kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)
    kv = torch.cat([a.window_kv_cache[:bsz], kv], dim=1)
    o = K.sparse_attn(q, kv, a.attn_sink, topk_idxs, a.softmax_scale)
    rec(p + "o_raw", o[0])
    M.apply_rotary_emb(o[..., -rd:], freqs_cis, True)
    rec(p + "o", o[0])
    o = o.view(bsz, block_size, a.n_local_groups, -1)
    wo_a = a.wo_a.weight.view(a.n_local_groups, a.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return a.wo_b(o.flatten(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/models")
    ap.add_argument("--stages", default="0")
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--decode", type=int, default=3)
    ap.add_argument("--check", action="store_true", help="also run the untraced forward and compare")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    cfg = json.load(open(os.path.join(ROOT, "ref", "inference", "config.json")))
    args = M.ModelArgs(**cfg)
    args.max_batch_size = 1
    B = args.dspark_block_size
    args.max_seq_len = ((a.T + a.decode + B + 255) // 256) * 256 + 256
    M.world_size, M.rank, M.default_dtype = 1, 0, torch.float8_e4m3fn
    M.shared_attn = M.SharedAttentionRuntime()
    stages = [int(v) for v in a.stages.split(",")]
    ck = CkptIndex(a.ckpt)
    T, D = a.T, a.decode
    calls = [(0, T)] + [(T + d, 1) for d in range(D)]

    def build(s):
        with torch.device("cuda"):
            m = M.DSparkAttention(args.n_layers + s, args)
        load_module(m, ck, f"mtp.{s}.attn.")
        return m

    rec = Rec(a.out)
    nop = lambda name, t: None  # noqa: E731
    for s in stages:
        main_xs = make_x(ck, "mtp.0.main_norm.weight", T + D, 2000 + s)
        xs = make_x(ck, f"mtp.{s}.attn_norm.weight", B * D, 3000 + s)
        ref_out = {}
        if a.check:
            m = build(s)
            with torch.inference_mode():
                for ci, (sp, n) in enumerate(calls):
                    x = xs[:, B * (ci - 1): B * ci] if ci else xs[:, :B]
                    ref_out[ci] = m(x, sp, main_xs[:, sp: sp + n]).clone()
            del m
        m = build(s)
        with torch.inference_mode():
            for ci, (sp, n) in enumerate(calls):
                x = xs[:, B * (ci - 1): B * ci] if ci else xs[:, :B]
                y = traced_dspark(m, x, sp, main_xs[:, sp: sp + n], rec if ci or T <= 4096 else nop,
                                  f"S{s}_c{ci}_")
                if a.check:
                    same = torch.equal(y, ref_out[ci])
                    print(f"stage {s} call {ci}: traced == model.py: {same}", flush=True)
                    assert same
        rec(f"S{s}_attn_sink", m.attn_sink.detach().float())
        rec(f"S{s}_freqs", torch.view_as_real(m.freqs_cis[: T + D + B]).float())
        rec(f"S{s}_w_kv_norm", m.kv_norm.weight.detach())
        del m
        torch.cuda.empty_cache()
        print("stage", s, "done", flush=True)
    rec.save({"dspark": stages, "T": T, "decode": D, "block": B, "calls": calls})
    print("wrote", len(rec.manifest), "tensors to", a.out, flush=True)


if __name__ == "__main__":
    main()
