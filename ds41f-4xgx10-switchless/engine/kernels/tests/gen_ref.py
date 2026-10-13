"""Reference data for the decode kernels, from real checkpoint weights (rank 0's TP=4 slices of
layer 3) and the pure-torch stand-ins in ref/torchref/kernel.py, which mirror model.py's numerics.

Runs in the canary-roce image (see engine/kernels/README.md for the docker line).
Everything is written as .npy: bf16 as uint16 bits, e4m3/e2m1/ue8m0 as uint8, plus manifest.json.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "ref", "torchref"))
import kernel as K  # noqa: E402
from safetensors import safe_open  # noqa: E402

CKPT = os.environ.get("CKPT", "/models")
OUT = os.environ.get("OUT", "/out")
LAYER = 3
M = 8          # activation rows; tests use prefixes 1..8
TP = 4
dev = "cuda"
torch.set_default_dtype(torch.bfloat16)
torch.backends.cuda.matmul.allow_tf32 = False
g = torch.Generator(device="cpu").manual_seed(1234)
manifest = {}


class Ckpt:
    def __init__(self, root):
        self.root = root
        self.map = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.h = {}

    def slice(self, name, rows=None, cols=None):
        f = self.map[name]
        if f not in self.h:
            self.h[f] = safe_open(os.path.join(self.root, f), framework="pt", device="cpu")
        s = self.h[f].get_slice(name)
        r = slice(*rows) if rows else slice(None)
        if len(s.get_shape()) == 1:
            return s[r]
        c = slice(*cols) if cols else slice(None)
        return s[r, c]


ck = Ckpt(CKPT)


def save(name, t, meaning):
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        a, dt = t.view(torch.uint16).numpy(), "bf16"
    elif t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.int8, torch.uint8):
        a = t.view(torch.uint8).numpy()
        dt = {torch.float8_e4m3fn: "e4m3", torch.float8_e8m0fnu: "ue8m0"}.get(t.dtype, "u8")
    elif t.dtype == torch.float32:
        a, dt = t.numpy(), "f32"
    elif t.dtype in (torch.int32, torch.int64):
        a, dt = t.to(torch.int32).numpy(), "i32"
    else:
        raise TypeError(t.dtype)
    np.save(os.path.join(OUT, name + ".npy"), np.ascontiguousarray(a))
    manifest[name] = {"dtype": dt, "shape": list(a.shape), "meaning": meaning}


def rand_act(rows, k, spread=True):
    """bf16 activations: normal values, optionally with per-32-block magnitudes over 4 decades."""
    x = torch.randn(rows, k, generator=g)
    if spread:
        mag = 10 ** (torch.rand(rows, k // 32, generator=g) * 4 - 2)
        x = x * mag.repeat_interleave(32, 1)
    return x.to(torch.bfloat16)


def quant(x):
    """act_quant's pieces: e4m3 values and the ue8m0 scale byte per 32."""
    xf = x.float().unflatten(-1, (-1, 32))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = K.fast_round_scale(amax, 1.0 / 448.0)
    q = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).flatten(-2)
    e = (torch.log2(s.squeeze(-1)).round().to(torch.int32) + 127).to(torch.uint8)
    return q, e


def fp8_linear(x, w, s):
    a, a_s = K.act_quant(x.to(dev), 32, "ue8m0")
    return K.fp8_gemm(a, a_s, w.to(dev).view(torch.float8_e4m3fn), s.to(dev), block_size=32)


def fp4_linear(x, w, s):
    a, a_s = K.act_quant(x.to(dev), 32, "ue8m0")
    return K.fp4_gemm(a, a_s, w.to(dev), s.to(dev))


# ---- act_quant: wide magnitudes, tiny blocks hitting the 1e-4 clamp, zeros, rounding ties
x = torch.randn(M, 5120, generator=g)
mag = 10 ** (torch.rand(M, 160, generator=g) * 10 - 7)  # 1e-7 .. 1e3
x = x * mag.repeat_interleave(32, 1)
x[0, :32] = 0
x[1, :64] = torch.tensor([448.0 * 1.0625 / 512 * (i % 8 + 1) for i in range(64)])
x = x.to(torch.bfloat16)
q, e = quant(x)
save("aq_x", x, "act_quant input bf16 [8,5120]")
save("aq_q", q, "e4m3 [8,5120]")
save("aq_s", e, "ue8m0 scale byte [8,160]")

# ---- FP8 dense GEMVs, rank 0 slices
P = f"layers.{LAYER}."
dense = {
    # name: (tensor, rows, cols) in element units; scales are sliced by /32
    "wq_a": (P + "attn.wq_a", (0, 1280 // TP), None),
    "wkv": (P + "attn.wkv", (0, 512 // TP), None),
    "wq_b": (P + "attn.wq_b", (0, 32768 // TP), None),
    "wo_b": (P + "attn.wo_b", None, (0, 8192 // TP)),
    "sh_w1": (P + "ffn.shared_experts.w1", (0, 2304 // TP), None),
    "sh_w2": (P + "ffn.shared_experts.w2", None, (0, 2304 // TP)),
    "ix_wq_b": ("layers.2.attn.indexer.wq_b", None, None),
}
for name, (t, rows, cols) in dense.items():
    w = ck.slice(t + ".weight", rows, cols)
    s = ck.slice(t + ".scale", rows and (rows[0] // 32, rows[1] // 32), cols and (cols[0] // 32, cols[1] // 32))
    xin = rand_act(M, w.shape[1])
    y = fp8_linear(xin, w, s)
    save(f"{name}_w", w, f"e4m3 {list(w.shape)}")
    save(f"{name}_s", s, f"ue8m0 {list(s.shape)} per 32x32")
    save(f"{name}_x", xin, "bf16 input")
    save(f"{name}_y", y, "bf16 output (act_quant + fp8_gemm)")
    print(name, tuple(w.shape), flush=True)

# ---- MoE: 12 routed experts (rank 0's 576 slice) + the shared expert, 8 rows routed top-6
EXPERTS = sorted(torch.randperm(384, generator=g)[:12].tolist())
I = 2304 // TP
w1s, s1s, w3s, s3s, w2s, s2s = [], [], [], [], [], []
for e in EXPERTS:
    b = P + f"ffn.experts.{e}."
    w1s.append(ck.slice(b + "w1.weight", (0, I))); s1s.append(ck.slice(b + "w1.scale", (0, I)))
    w3s.append(ck.slice(b + "w3.weight", (0, I))); s3s.append(ck.slice(b + "w3.scale", (0, I)))
    w2s.append(ck.slice(b + "w2.weight", None, (0, I // 2))); s2s.append(ck.slice(b + "w2.scale", None, (0, I // 32)))
xm = (rand_act(M, 5120, spread=False).float() * 0.5).to(torch.bfloat16)
sel = torch.stack([torch.randperm(12, generator=g)[:6] for _ in range(M)])
rw = torch.rand(M, 6, generator=g) + 0.05
rw = (rw / rw.sum(-1, keepdim=True) * 1.5).float()
y = torch.zeros(M, 5120, dtype=torch.float32, device=dev)
for m in range(M):
    for j in sorted(range(6), key=lambda j: EXPERTS[sel[m, j]]):
        i = sel[m, j].item()
        xr = xm[m:m + 1]
        gte = fp4_linear(xr, w1s[i], s1s[i]).float()
        up = fp4_linear(xr, w3s[i], s3s[i]).float()
        up = up.clamp(-10.0, 10.0)
        gte = gte.clamp(max=10.0)
        h = (F.silu(gte) * up) * rw[m, j].to(dev)
        y[m] += fp4_linear(h.to(torch.bfloat16), w2s[i], s2s[i]).float()[0]
sw1 = ck.slice(P + "ffn.shared_experts.w1.weight", (0, I)); ss1 = ck.slice(P + "ffn.shared_experts.w1.scale", (0, I // 32))
sw3 = ck.slice(P + "ffn.shared_experts.w3.weight", (0, I)); ss3 = ck.slice(P + "ffn.shared_experts.w3.scale", (0, I // 32))
sw2 = ck.slice(P + "ffn.shared_experts.w2.weight", None, (0, I)); ss2 = ck.slice(P + "ffn.shared_experts.w2.scale", None, (0, I // 32))
gte = fp8_linear(xm, sw1, ss1).float().clamp(max=10.0)
up = fp8_linear(xm, sw3, ss3).float().clamp(-10.0, 10.0)
ysh = fp8_linear((F.silu(gte) * up).to(torch.bfloat16), sw2, ss2).float()
save("moe_x", xm, "bf16 [8,5120]")
save("moe_ids", torch.tensor(EXPERTS), "global expert ids of the 12 local slots")
save("moe_sel", sel, "i32 [8,6] local slot per (row, pick)")
save("moe_rw", rw, "f32 [8,6] routing weight per (row, pick)")
save("moe_w1", torch.stack(w1s), "u8 [12,576,2560] e2m1 packed, low nibble first")
save("moe_s1", torch.stack(s1s), "ue8m0 [12,576,160]")
save("moe_w3", torch.stack(w3s), "u8 [12,576,2560]")
save("moe_s3", torch.stack(s3s), "ue8m0 [12,576,160]")
save("moe_w2", torch.stack(w2s), "u8 [12,5120,288]")
save("moe_s2", torch.stack(s2s), "ue8m0 [12,5120,18]")
for n, t in (("w1", sw1), ("s1", ss1), ("w3", sw3), ("s3", ss3), ("w2", sw2), ("s2", ss2)):
    save(f"moe_sh{n}", t, "shared expert rank-0 slice")
save("moe_y_routed", y, "f32 [8,5120]: sum over picks (expert-id order) of bf16 expert outputs")
save("moe_y_shared", ysh, "f32 [8,5120]: bf16 shared-expert output")
print("moe", EXPERTS, flush=True)

# ---- wo_a: rank 0 = groups 0 and 1, FP8 dequantised to bf16 as convert.py does
wa = ck.slice(P + "attn.wo_a.weight", (0, 2048)); sa = ck.slice(P + "attn.wo_a.scale", (0, 64))
wbf = K.dequant_fp8_weight(wa.to(dev), sa.to(dev), 32).to(torch.bfloat16)
o = rand_act(M, 8192, spread=False)
yo = torch.einsum("mgd,grd->mgr", o.to(dev).view(M, 2, 4096).float(), wbf.view(2, 1024, 4096).float())
yo_bf = torch.einsum("mgd,grd->mgr", o.to(dev).view(M, 2, 4096), wbf.view(2, 1024, 4096))
save("woa_w", wa, "e4m3 [2048,4096]"); save("woa_s", sa, "ue8m0 [64,128]")
save("woa_x", o, "bf16 [8,8192] = 2 groups x 4096")
save("woa_y", yo.reshape(M, 2048).to(torch.bfloat16), "bf16 [8,2048], fp32 math")
save("woa_y_einsum", yo_bf.reshape(M, 2048), "bf16 [8,2048], torch bf16 einsum (cuBLAS)")

# ---- head: rank 0's vocab slice, fp32 logits
hw = ck.slice("head.weight", (0, 129280 // TP))
xh = rand_act(M, 5120, spread=False)
lg = xh.to(dev).float() @ hw.to(dev).float().t()
save("head_w", hw, "bf16 [32320,5120]"); save("head_x", xh, "bf16 [8,5120]"); save("head_y", lg, "f32 logits [8,32320]")

# ---- RMSNorm (on the GPU, like the reference run)
nw = ck.slice(P + "attn_norm.weight").to(dev)
xn = rand_act(M, 5120).to(dev)
xf = xn.float()
yn = (nw * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-20))).to(torch.bfloat16)
save("norm_w", nw, "bf16 [5120]"); save("norm_x", xn, "bf16 [8,5120]"); save("norm_y", yn, "bf16 [8,5120]")

# ---- router: full gate (top-k needs all 384 scores)
gw = ck.slice(P + "ffn.gate.weight").to(dev); gb = ck.slice(P + "ffn.gate.bias").to(dev)
xr = rand_act(M, 5120, spread=False).to(dev)
sc = F.linear(xr.float(), gw.float())
s = F.softplus(sc).sqrt()
idx = (s + gb).topk(6, dim=-1)[1]
wts = s.gather(1, idx)
wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * 1.5
save("gate_w", gw, "bf16 [384,5120]"); save("gate_b", gb, "f32 [384]"); save("gate_x", xr, "bf16 [8,5120]")
save("gate_logits", sc, "f32 [8,384] x.W^T"); save("gate_idx", idx, "i32 [8,6] topk order")
save("gate_wts", wts, "f32 [8,6]")

json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print("done", len(manifest), "arrays", flush=True)
