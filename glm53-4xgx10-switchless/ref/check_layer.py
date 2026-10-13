"""Check glmref.py's decoder layer against transformers 5.15's GlmMoeDsaDecoderLayer (eager attention,
default grouped_mm experts) on random BF16 weights: the checkpoint's dimensions except 16 routed experts,
one sequence long enough that rows past 2048 go sparse.

  python3 check_layer.py --ckpt CKPT [--seq 2400] [--threads 8]

Prints, for the indexer's top-k sets, the attention sublayer and the whole layer: how many values are
bit-equal and the largest differences.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glmref as G  # noqa: E402


class DictCk:
    def __init__(self, sd):
        self.sd = sd

    def get(self, name):
        return self.sd[name]


def cmp(tag, a, b):
    a, b = a.float(), b.float()
    eq = (a == b).float().mean().item()
    d = (a - b).abs()
    rel = (d.norm() / b.norm()).item()
    print(f"{tag:28s} bit-equal {eq:7.4f}  max |diff| {d.max().item():.3e}  rel-l2 {rel:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq", type=int, default=2400)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--layer", type=int, default=6, help="a full-indexer MoE layer")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    from transformers import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M

    with open(os.path.join(args.ckpt, "config.json")) as f:
        raw = json.load(f)
    raw.pop("quantization_config", None)
    ne = 16
    raw["n_routed_experts"] = ne
    cfg = GlmMoeDsaConfig(**raw)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "grouped_mm"  # what the full model picks by default (model.md § 2.7)
    torch.manual_seed(0)
    layer = M.GlmMoeDsaDecoderLayer(cfg, args.layer).to(torch.bfloat16).eval()
    for n, p in layer.named_parameters():  # trained-like scales: N(0, 0.02), norms near 1
        if p.dim() >= 2:
            p.normal_(0, 0.02)
        elif "bias" in n and "e_score" not in n:
            p.normal_(0, 0.1)
        else:
            p.normal_(1, 0.1)
    layer.mlp.experts.float()  # keep 3-D expert params as bf16 below
    layer.mlp.experts.to(torch.bfloat16)
    layer.self_attn.indexer.weights_proj.float()  # _keep_in_fp32_modules
    layer.mlp.gate.e_score_correction_bias.normal_(34.4, 0.1)
    rot = M.GlmMoeDsaRotaryEmbedding(cfg)

    S = args.seq
    h = (torch.randn(1, S, G.D) * 0.5).to(torch.bfloat16)
    pos = torch.arange(S)[None]
    cos, sin = rot(h, pos)
    out_ref, topk_ref = layer(h, attention_mask=None, position_ids=pos, position_embeddings=(cos, sin))
    a_ref, _, _ = layer.self_attn(layer.input_layernorm(h), (cos, sin), None, position_ids=pos)

    # the same weights in checkpoint naming
    p = f"model.layers.{G.MTP}"
    sd = {f"{p}.{k}": v for k, v in layer.state_dict().items() if not k.startswith("mlp.experts.")}
    sd[f"{p}.self_attn.indexer.weights_proj.weight"] = sd[f"{p}.self_attn.indexer.weights_proj.weight"].to(torch.bfloat16)
    gu, dn = layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj
    for e in range(ne):
        sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = gu[e, :G.DFF].contiguous()
        sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = gu[e, G.DFF:].contiguous()
        sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = dn[e].contiguous()
    G.NE = ne
    G.COMBINE = "bf16"  # transformers' rounding order, to compare against it
    W = G.LayerW(DictCk(sd), G.MTP)
    a = argparse.Namespace(attn_block=256, head_group=16, index_block=128)
    gcos, gsin = G.rope_tables(S)
    x = G.rms_norm(h[0], W("input_layernorm.weight"), 1e-5)
    a_mine, tk, mg = G.attention(x, W, True, None, gcos, gsin, a)
    sets_ref = topk_ref[0, G.TOPK:].long().sort(dim=1).values
    same = (sets_ref == tk.long()).all(dim=1)
    print(f"index sets equal on {int(same.sum())}/{len(same)} sparse rows; boundary ties: {int((mg == 0).sum())}")
    cmp("attention output", a_mine, a_ref[0])
    hh = h[0] + a_mine
    xx = G.rms_norm(hh, W("post_attention_layernorm.weight"), 1e-5)
    m, route, calib = G.moe(xx, W, G.MTP, G.ThreadPoolExecutor(2))
    out_mine = hh + m
    cmp("layer output", out_mine, out_ref[0])
    # routing against transformers' router on the same input
    _, w_ref, i_ref = layer.mlp.gate(xx)
    i_ref_s, order = i_ref.sort(dim=1)
    mine_s, order2 = torch.from_numpy(route["route_idx"]).long().sort(dim=1)
    print(f"routing: expert sets equal on {(i_ref_s == mine_s).all(1).float().mean().item():.4f} of rows; weights max |diff| "
          f"{(w_ref.gather(1, order) - torch.from_numpy(route['route_w']).gather(1, order2)).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
