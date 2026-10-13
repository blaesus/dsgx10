"""Reference data for the prefill GEMMs (prefill.cu): 256-row chunks through the same rank-0 layer-3
weights gen_ref.py saved (run that first; this script reads them back from OUT) and the torch
stand-ins in ref/torchref/kernel.py. Arrays are named p_*; docker line as for gen_ref.py, with this
script instead.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "ref", "torchref"))
import kernel as K  # noqa: E402

OUT = os.environ.get("OUT", "/out")
M = 256
dev = "cuda"
torch.set_default_dtype(torch.bfloat16)
torch.backends.cuda.matmul.allow_tf32 = False
g = torch.Generator(device="cpu").manual_seed(4321)
manifest = {}
U8 = {"e4m3": torch.float8_e4m3fn, "ue8m0": torch.float8_e8m0fnu, "u8": torch.uint8}


def load(name):
    """A gen_ref.py array back as the torch dtype it was saved from."""
    dt = json.load(open(os.path.join(OUT, "manifest.json")))[name]["dtype"]
    a = torch.from_numpy(np.load(os.path.join(OUT, name + ".npy")))
    if dt == "bf16":
        return a.view(torch.bfloat16)
    return a.view(U8[dt]) if dt in U8 else a


def save(name, t, meaning):
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        a, dt = t.view(torch.uint16).numpy(), "bf16"
    elif t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.uint8):
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
    x = torch.randn(rows, k, generator=g)
    if spread:
        mag = 10 ** (torch.rand(rows, k // 32, generator=g) * 4 - 2)
        x = x * mag.repeat_interleave(32, 1)
    return x.to(torch.bfloat16)


def quant(x):
    xf = x.float().unflatten(-1, (-1, 32))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = K.fast_round_scale(amax, 1.0 / 448.0)
    q = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).flatten(-2)
    e = (torch.log2(s.squeeze(-1)).round().to(torch.int32) + 127).to(torch.uint8)
    return q, e


def fp8_linear(x, w, s):
    a, a_s = K.act_quant(x.to(dev), 32, "ue8m0")
    return K.fp8_gemm(a, a_s, w.to(dev), s.to(dev), block_size=32)


def fp4_linear(x, w, s):
    a, a_s = K.act_quant(x.to(dev), 32, "ue8m0")
    return K.fp4_gemm(a, a_s, w.to(dev), s.to(dev))


# ---- dense FP8 GEMMs; wq_a and wkv share their input (one two-segment launch)
xa = rand_act(M, 5120)
for name in ("wq_a", "wkv", "wq_b", "wo_b", "sh_w2", "ix_wq_b"):
    w, s = load(f"{name}_w"), load(f"{name}_s")
    xin = xa if name in ("wq_a", "wkv") else rand_act(M, w.shape[1])
    save(f"p_{name}_x", xin, f"bf16 [{M},{w.shape[1]}]")
    save(f"p_{name}_y", fp8_linear(xin, w, s), f"bf16 [{M},{w.shape[0]}]")
    print(name, flush=True)

# ---- shared expert: SwiGLU of w1/w3, its act_quant (w2's input), and w2's output
sw1, ss1, sw3, ss3 = load("moe_shw1"), load("moe_shs1"), load("moe_shw3"), load("moe_shs3")
sw2, ss2 = load("moe_shw2"), load("moe_shs2")
xs = (rand_act(M, 5120, spread=False).float() * 0.5).to(torch.bfloat16)
gte = fp8_linear(xs, sw1, ss1).float().clamp(max=10.0)
up = fp8_linear(xs, sw3, ss3).float().clamp(-10.0, 10.0)
h = (F.silu(gte) * up).to(torch.bfloat16)
hq, hs = quant(h)
save("p_sh_x", xs, f"bf16 [{M},5120]")
save("p_sh_hq", hq, f"e4m3 [{M},576]: act_quant(bf16(silu(min(w1 x, 10)) * clamp(w3 x, +-10)))")
save("p_sh_hs", hs, f"ue8m0 [{M},18]")
save("p_sh_y", fp8_linear(h, sw2, ss2), f"bf16 [{M},5120]")
print("shared", flush=True)

# ---- routed experts: gen_ref.py's 12 (local slots in expert-id order), skewed top-6 routing so that
# some experts get several 64-row tiles, some a partial one and some none
w1, s1, w3, s3 = load("moe_w1"), load("moe_s1"), load("moe_w3"), load("moe_s3")
w2, s2 = load("moe_w2"), load("moe_s2")
E = w1.shape[0]
pw = torch.tensor([40.0, 15, 8, 5, 3, 2, 1, 0.5, 0.2, 0.05, 0.01, 1e-6])[torch.randperm(E, generator=g)]
sel = torch.multinomial(pw.float().expand(M, E), 6, replacement=False, generator=g)
rw = torch.rand(M, 6, generator=g) + 0.05
rw = (rw / rw.sum(-1, keepdim=True) * 1.5).float()
xm = (rand_act(M, 5120, spread=False).float() * 0.5).to(torch.bfloat16)
y = torch.zeros(M, 5120, dtype=torch.float32, device=dev)
counts = []
for i in range(E):  # ascending expert id: each row adds its picks' outputs in that order
    ms, js = (sel == i).nonzero(as_tuple=True)
    counts.append(len(ms))
    if len(ms) == 0:
        continue
    X = xm[ms]
    gte = fp4_linear(X, w1[i], s1[i]).float().clamp(max=10.0)
    up = fp4_linear(X, w3[i], s3[i]).float().clamp(-10.0, 10.0)
    hh = (F.silu(gte) * up) * rw[ms, js].to(dev)[:, None]
    y.index_add_(0, ms.to(dev), fp4_linear(hh.to(torch.bfloat16), w2[i], s2[i]).float())
gte = fp8_linear(xm, sw1, ss1).float().clamp(max=10.0)
up = fp8_linear(xm, sw3, ss3).float().clamp(-10.0, 10.0)
ysh = fp8_linear((F.silu(gte) * up).to(torch.bfloat16), sw2, ss2).float()
save("p_moe_x", xm, f"bf16 [{M},5120]")
save("p_moe_sel", sel, f"i32 [{M},6] local slot (= expert-id order) per pick")
save("p_moe_rw", rw, f"f32 [{M},6]")
save("p_moe_y_routed", y, f"f32 [{M},5120]: per row, sum over picks in expert-id order of bf16 outputs")
save("p_moe_y_shared", ysh, f"f32 [{M},5120]: bf16 shared-expert output")
print("moe rows per expert", counts, flush=True)

# ---- wo_a (block-diagonal, FP8 widened to bf16, fp32 math) and its act_quant (wo_b's input)
wa, sa = load("woa_w"), load("woa_s")
wbf = K.dequant_fp8_weight(wa.to(dev), sa.to(dev), 32).to(torch.bfloat16)
o = rand_act(M, 8192, spread=False)
yo = torch.einsum("mgd,grd->mgr", o.to(dev).view(M, 2, 4096).float(), wbf.view(2, 1024, 4096).float())
yo = yo.reshape(M, 2048).to(torch.bfloat16)
q, e = quant(yo)
save("p_woa_x", o, f"bf16 [{M},8192]")
save("p_woa_y", yo, f"bf16 [{M},2048]")
save("p_woa_q", q, f"e4m3 [{M},2048]")
save("p_woa_qs", e, f"ue8m0 [{M},64]")

# ---- router gate, bf16 x bf16 -> fp32 logits
gw = load("gate_w").to(dev)
xr = rand_act(M, 5120, spread=False).to(dev)
save("p_gate_x", xr, f"bf16 [{M},5120]")
save("p_gate_logits", xr.float() @ gw.float().t(), f"f32 [{M},384]")

mf = os.path.join(OUT, "manifest.json")
full = json.load(open(mf))
full.update(manifest)
json.dump(full, open(mf, "w"), indent=1)
print("done", len(manifest), "arrays", flush=True)
