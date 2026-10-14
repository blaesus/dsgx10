#!/usr/bin/env python3
"""Real-tensor data for attn_tests R: one rank's slices of an INT4 layer's q_b, kv_b (W_UV and W_UK^T) and o_proj,
and of a BF16 layer's q_b and o_proj, from altar-1's checkpoint. For each: the engine tile (pack.py), the weight as
compressed-tensors' own `decompress` gives it ("lib", INT4 only: the library's dequantisation of the whole tensor,
then the rank's rows), 16 activation rows, and the oracle y = bf16(f32(x) @ f32(W)^T) (torch, f32 accumulation).
Also checks here that pack.py's dequantisation of the tile equals the library's (int4_check.py does the whole
tensors) and that W_UK^T's BF16 tile is the library's weight transposed.

    python3 -I real_ref.py --ckpt DIR [--index JSON] --out OUT [--layer 10] [--bf16-layer 3] [--rank 1]

Shards are read only; verify them against the manifest's sha256 first."""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import pack  # noqa: E402

bf = torch.bfloat16


def u16(t):
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def tbf(a):
    return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(bf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--bf16-layer", type=int, default=3)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count()))
    from safetensors import safe_open
    from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))["quantization_config"]["config_groups"]["group_0"]["weights"]
    scheme = QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(
        **{k: v for k, v in cfg.items() if k in ("num_bits", "type", "symmetric", "strategy", "group_size")}))
    index = json.load(open(a.index or os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]

    def get(name):
        with safe_open(os.path.join(a.ckpt, index[name]), framework="pt") as f:
            return f.get_tensor(name)

    os.makedirs(a.out, exist_ok=True)
    man, r, g = {}, a.rank, torch.Generator().manual_seed(a.seed)

    def rec(name, arr):
        arr = np.ascontiguousarray(arr)
        np.save(os.path.join(a.out, name + ".npy"), arr)
        man[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}

    def xrows(k):  # activations with a spread of row magnitudes
        return (torch.randn(16, k, generator=g) * (0.5 + torch.rand(16, 1, generator=g))).to(bf)

    def oracle(x, w, groups=1):  # x [16][groups * Kg], w [groups * Ng][Kg]
        xf, wf = x.float().view(16, groups, -1), w.float().view(groups, -1, w.shape[1])
        return torch.einsum("rgk,gnk->rgn", xf, wf).reshape(16, -1).to(bf)

    def int4(proj):
        base = f"model.layers.{a.layer}.self_attn.{proj}."
        sd = {k: get(base + k) for k in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape")}
        lib = PackedQuantizationCompressor.decompress(dict(sd), scheme)["weight"]
        return (sd["weight_packed"].numpy().view(np.uint32), u16(sd["weight_scale"]),
                sd["weight_zero_point"].numpy().view(np.uint32)), lib

    t0 = time.time()
    H = range(16 * r, 16 * r + 16)
    # q_b rows [4096r, +4096)
    ct, lib = int4("q_b_proj")
    w, L = pack.int4_q_b(ct, r), lib[4096 * r:4096 * r + 4096]
    assert np.array_equal(pack.int4_dequant(*pack.int4_untile(w, 4096, 2048)), u16(L))
    x = xrows(2048)
    rec("q_b_w", w); rec("q_b_lib", u16(L)); rec("q_b_x", u16(x)); rec("q_b_y", u16(oracle(x, L)))
    # kv_b: W_UV (INT4 tile) and W_UK^T (BF16: the exact dequantisation, transposed)
    ct, lib = int4("kv_b_proj")
    w = pack.int4_w_uv(ct, r)
    L = torch.cat([lib[448 * h + 192:448 * h + 448] for h in H])
    assert np.array_equal(pack.int4_dequant(*pack.int4_untile(w, 4096, 512)), u16(L))
    x = xrows(16 * 512)
    rec("w_uv_w", w); rec("w_uv_lib", u16(L)); rec("w_uv_x", u16(x)); rec("w_uv_y", u16(oracle(x, L, 16)))
    wuk = pack.int4_w_uk(ct, r)
    Luk = torch.cat([lib[448 * h:448 * h + 192].T for h in H])
    assert np.array_equal(wuk, u16(Luk)), "W_UK^T tile != the library's weight transposed"
    xq = xrows(16 * 256)  # q-like rows [16][16 heads][256]: the GEMV reads each head's first 192
    xn = xq.view(16, 16, 256)[:, :, :192].reshape(16, -1)
    rec("w_uk_w", wuk); rec("w_uk_x", u16(xq)); rec("w_uk_y", u16(oracle(xn, Luk, 16)))
    # o_proj columns [4096r, +4096)
    ct, lib = int4("o_proj")
    w, L = pack.int4_o_proj(ct, r), lib[:, 4096 * r:4096 * r + 4096].contiguous()
    assert np.array_equal(pack.int4_dequant(*pack.int4_untile(w, 6144, 4096)), u16(L))
    x = xrows(4096)
    rec("o_proj_w", w); rec("o_proj_lib", u16(L)); rec("o_proj_x", u16(x)); rec("o_proj_y", u16(oracle(x, L)))
    print(f"[{time.time() - t0:.0f}s] INT4 layer {a.layer}: pack.py's tiles dequantise to the library's weights; W_UK^T = "
          "the library's transposed", flush=True)
    # a BF16 layer's q_b and o_proj
    qb = get(f"model.layers.{a.bf16_layer}.self_attn.q_b_proj.weight")
    w = pack.bf16_rows(u16(qb), r, 4096)
    x = xrows(2048)
    rec("q_b_bf16_w", w); rec("q_b_bf16_x", u16(x)); rec("q_b_bf16_y", u16(oracle(x, tbf(w))))
    op = get(f"model.layers.{a.bf16_layer}.self_attn.o_proj.weight")
    w = pack.bf16_cols(u16(op), r, 4096)
    x = xrows(4096)
    rec("o_proj_bf16_w", w); rec("o_proj_bf16_x", u16(x)); rec("o_proj_bf16_y", u16(oracle(x, tbf(w))))
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump({"tensors": man, "kind": "real", "layer": a.layer, "bf16_layer": a.bf16_layer, "rank": r,
                   "source": f"altar-1 layer {a.layer} (INT4) and {a.bf16_layer} (BF16)"}, f, indent=1)
    print(f"[{time.time() - t0:.0f}s] wrote {len(man)} tensors to {a.out}")


if __name__ == "__main__":
    main()
