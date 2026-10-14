"""GLM-5.3 layer-streaming CPU reference: the numerics contract (docs/model.md, docs/quantisation.md § 5) as
torch code, run teacher-forced on the NVFP4 checkpoint, one layer's weights at a time.

  python3 glmref.py --ckpt CKPT --out-dir OUT --tokens A.tokens.json [B.tokens.json ...] [--threads 8]

Each tokens file is {"name": ..., "tokens": [...]} (make_tokens.py). All sequences go through each layer
together, so every layer's weights are read once a pass; each sequence attends only to itself, from
position 0. The pass is resumable: after every layer the residual stream is checkpointed under
OUT/state, and a rerun with the same arguments continues from the next layer.

The contract, as implemented here (each line cites the model.md section it follows):
- activations are BF16 between operations; every GEMM accumulates in f32 and rounds its output once
  (§ 2). BF16 weights use torch's bf16 GEMM (f32 accumulation, checked on this CPU).
- NVFP4 experts (§ 5.2): W' = E2M1(code) · E4M3(block scale), exact in bf16 and f32; the per-tensor
  weight_scale_2 multiplies the f32 GEMM output before its one rounding: y = bf16(ws2 · Σ x·W'). The
  FP8 dense MLPs of layers 0-2 (§ 5.3) the same way with their per-tensor weight_scale. No activation
  quantisation (input_scale is ignored): quantisation.md § 5, level 2.
- RMSNorm: f32 normalisation, cast to bf16, then a bf16 weight multiply (§ 2.2); eps 1e-5, 1e-6 for the
  two latent norms.
- RoPE interleaved, bf16 arithmetic with bf16 cos/sin, output de-interleaved (§ 2.4.1).
- Attention in the reference's expanded form, eager numerics: bf16 QK^T, scaled by 1/16, masked,
  f32 softmax rounded to bf16, bf16 PV (§ 2.4). Query p attends to all causal keys when p < 2048, else
  to the indexer's top-2048 (§ 2.5).
- Indexer: q, k bf16 (k through a LayerNorm with bias), RoPE on the first 64; f32 scores, ReLU, f32 head
  weights from an f32 GEMM of fp32(x); top-2048 by torch.topk (ties unspecified, § 2.5). Shared layers
  reuse the last full layer's sets.
- Router in f32, bias for selection only, top-8 (taken sorted), weights renormalised × 2.5 (§ 2.7).
- Sublayer outputs (--combine f32, the default and the contract): o_proj, the dense MLP's down projection,
  each routed expert's down projection (× its f32 weight) and the shared expert's down projection stay f32;
  they are summed in f32 (routed experts in expert-id order, then the shared expert) and rounded once, as
  bf16(f32(h) + sum), when added to the residual. This is the TP=4 engine's order: each rank holds f32 partial
  sums that are all-reduced before any rounding. --combine bf16 is transformers' order instead (each GEMM
  output rounded to bf16; routed experts weighted and summed in f32, rounded; shared expert and residual
  added in bf16).
- Logits are kept in f32 (bf16 hidden × bf16 head, f32 accumulation, no output rounding); transformers'
  bf16 logits are bf16() of them up to summation order.
- MTP (§ 3): e = enorm(embed(token t+1)) (zeroed at position 0), g = hnorm(post-final-norm hidden of row t),
  eh_proj(cat(e, g)), decoder layer 78 with its own indexer and BF16 experts at position t,
  shared_head.norm, the shared lm_head: the draft for token t+2.
"""
import argparse
import json
import os
import shutil
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

BF16, F32 = torch.bfloat16, torch.float32

# Dimensions (docs/model.md § 1); checked against config.json at start.
D, NH, RQ, RKV, DN, DR, DQK, DV = 6144, 64, 2048, 512, 192, 64, 256, 256
NI, DI, TOPK = 32, 128, 2048
NE, K8, DFF = 256, 8, 2048
NL, MTP = 78, 78
VOCAB = 154880
ROUTE_SCALE = 2.5
ATT_SCALE = DQK ** -0.5

ST_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn}

# E2M1 codes: bit 3 the sign, bits 0-2 index the magnitudes (model.md § 5.2).
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=F32)
# One packed byte -> its two values: low nibble = even k, high nibble = odd k.
NIB_LUT = torch.stack([E2M1[torch.arange(256) & 15], E2M1[torch.arange(256) >> 4]], dim=1)  # [256, 2] f32


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------------------------- checkpoint

class Ckpt:
    """Tensors by name with plain preads (no mmap, so nothing stays in RSS), page cache dropped after."""

    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self.shards, self.lock = {}, threading.Lock()

    def _shard(self, fname):
        with self.lock:
            if fname not in self.shards:
                fd = os.open(os.path.join(self.root, fname), os.O_RDONLY)
                n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
                self.shards[fname] = (fd, 8 + n, json.loads(os.pread(fd, n, 8)))
            return self.shards[fname]

    def entry(self, name):
        fd, base, hdr = self._shard(self.map[name])
        return fd, base, hdr[name]

    def get(self, name):
        fd, base, e = self.entry(name)
        a, b = e["data_offsets"]
        buf = bytearray(b - a)
        mv, got = memoryview(buf), 0
        while got < len(buf):
            k = os.preadv(fd, [mv[got:]], base + a + got)
            if k <= 0:
                raise IOError(f"short read of {name}")
            got += k
        os.posix_fadvise(fd, base + a, b - a, os.POSIX_FADV_DONTNEED)
        t = torch.frombuffer(buf, dtype=torch.uint8) if buf else torch.empty(0, dtype=torch.uint8)
        return t.view(ST_DTYPES[e["dtype"]]).reshape(e["shape"])

    def rows(self, name, ids):
        """Rows of a 2-D BF16 tensor (the embedding), one pread each."""
        fd, base, e = self.entry(name)
        assert e["dtype"] == "BF16"
        width = e["shape"][1] * 2
        out = torch.empty(len(ids), e["shape"][1], dtype=BF16)
        ov = out.view(torch.uint8).numpy()
        start = base + e["data_offsets"][0]
        for i, r in enumerate(ids):
            ov[i] = np.frombuffer(os.pread(fd, width, start + int(r) * width), dtype=np.uint8)
        return out

    def slab(self, name, r0, r1):
        """Rows r0..r1-1 of a 2-D BF16 tensor, one pread, page cache dropped after."""
        fd, base, e = self.entry(name)
        assert e["dtype"] == "BF16"
        width = e["shape"][1] * 2
        off = base + e["data_offsets"][0] + r0 * width
        buf = bytearray((r1 - r0) * width)
        mv, got = memoryview(buf), 0
        while got < len(buf):
            k = os.preadv(fd, [mv[got:]], off + got)
            if k <= 0:
                raise IOError(f"short read of {name}")
            got += k
        os.posix_fadvise(fd, off, len(buf), os.POSIX_FADV_DONTNEED)
        return torch.frombuffer(buf, dtype=BF16).view(r1 - r0, e["shape"][1])

    def has(self, name):
        return name in self.map


def nvfp4_raw(ck, p):
    return ck.get(p + ".weight"), ck.get(p + ".weight_scale"), float(ck.get(p + ".weight_scale_2"))


def nvfp4_dequant(q, s, s2):
    """NVFP4 weight as (W' f32 [N, K] = E2M1 · E4M3 block scale, exactly; weight_scale_2)."""
    n, k2 = q.shape
    w = NIB_LUT[q.long()].view(n, k2 * 2 // 16, 16) * s.to(F32).unsqueeze(-1)
    return w.view(n, k2 * 2), s2


def fp8(ck, p):
    """FP8 per-tensor weight as (E4M3 values in f32, weight_scale)."""
    return ck.get(p + ".weight").to(F32), float(ck.get(p + ".weight_scale"))


# ---------------------------------------------------------------------------------------------- the math

def rms_norm(x, w, eps):
    x32 = x.to(F32)
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return w * y.to(BF16)


def linear(x, w):
    """BF16 weight: bf16 GEMM, f32 accumulation, one rounding. An f32 weight (a quantised non-expert tensor,
    dequantised: tracks mode) runs as an f32 GEMM with one rounding."""
    if w.dtype == F32:
        return F.linear(x.to(F32), w).to(BF16)
    return F.linear(x, w)


# Where the sublayer outputs are rounded (--combine). "f32", the contract: the engine at TP=4 holds only
# partial sums of o_proj, of the dense/shared down projections and of every routed expert's down
# projection, all-reduces them in f32, and rounds once when adding to the residual:
# h = bf16(f32(h) + Σ f32 partials). "bf16": transformers' order (each GEMM output rounded to bf16; routed
# experts summed in f32 and rounded; + shared expert in bf16; + residual in bf16).
COMBINE = "f32"


def out_linear(x, w):
    """A sublayer's last GEMM on a BF16 weight: bf16 output (transformers) or f32 (the contract)."""
    if COMBINE == "f32":
        return F.linear(x.to(F32), w.to(F32))
    return linear(x, w)


def out_scaled_linear(x, w32, scale, chunk=4096):
    """The same for a scaled (NVFP4/FP8) weight."""
    if COMBINE != "f32":
        return scaled_linear(x, w32, scale, chunk)
    out = torch.empty(x.shape[0], w32.shape[0], dtype=F32)
    for i in range(0, x.shape[0], chunk):
        out[i:i + chunk] = F.linear(x[i:i + chunk].to(F32), w32) * scale
    return out


def add_residual(h, y):
    """bf16 residual + sublayer output: one rounding of the f32 sum (contract), or a bf16 add."""
    if y.dtype == F32:
        return (h.to(F32) + y).to(BF16)
    return h + y


def scaled_linear(x, w32, scale, chunk=4096):
    """Quantised weight held as exact f32 values W' and a per-tensor scale: bf16(scale · Σ x·W')."""
    out = torch.empty(x.shape[0], w32.shape[0], dtype=BF16)
    for i in range(0, x.shape[0], chunk):
        out[i:i + chunk] = (F.linear(x[i:i + chunk].to(F32), w32) * scale).to(BF16)
    return out


def rope_tables(n):
    inv = 1.0 / (8_000_000.0 ** (torch.arange(0, DR, 2, dtype=F32) / DR))  # [32]
    ang = torch.arange(n, dtype=F32)[:, None] * inv[None, :]                # [n, 32]
    return ang.cos().to(BF16), ang.sin().to(BF16)


def rope(x, cos, sin):
    """Interleaved pairs (x[2i], x[2i+1]) rotated in bf16; the output de-interleaved (evens' results first)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def silu_mul(g, u):
    return F.silu(g) * u


class LayerW:
    """One layer's non-expert weights, read on demand."""

    def __init__(self, ck, l, quant=None, cache=None):
        self.ck, self.p = ck, f"model.layers.{l}"
        self.quant = quant or {}          # tracks mode: weight name -> "fp8b" | "nvfp4"
        self.cache = cache if cache is not None else {}

    def __call__(self, name):
        fmt = self.quant.get(name)
        if fmt is None:
            return self.ck.get(f"{self.p}.{name}")
        key = (self.p, name, fmt)
        if key not in self.cache:
            self.cache[key] = quantise_nonexpert(self.ck.get(f"{self.p}.{name}"), fmt)
        return self.cache[key]


def fp8_block(w):
    """zai's FP8 release, recomputed from the BF16 master: per 128x128 block, scale = amax * f32(1/448)
    (checked equal to the release's weight_scale_inv on 4 tensors, 100 %), codes e4m3(w / scale) (checked
    equal to the release's codes, quantisation.md § 3.0). Returns the dequantised values in f32 (code ×
    scale rounded once to f32)."""
    w = w.to(F32)
    R, C = w.shape
    pr, pc = -R % 128, -C % 128
    wp = F.pad(w, (0, pc, 0, pr))
    blk = wp.reshape((R + pr) // 128, 128, (C + pc) // 128, 128)
    sc = blk.abs().amax(dim=(1, 3), keepdim=True) * torch.tensor(1 / 448, dtype=F32)
    q = (blk / torch.where(sc == 0, torch.ones_like(sc), sc)).to(torch.float8_e4m3fn).to(F32)
    return (q * sc).reshape(R + pr, C + pc)[:R, :C].contiguous()


def quantise_nonexpert(w, fmt):
    if fmt == "fp8b":
        return fp8_block(w)
    if fmt == "nvfp4":
        pk = quant_formats().nvfp4_rtn(w.to(F32))
        wq, s2 = nvfp4_dequant(torch.from_numpy(pk["weight"]),
                               torch.from_numpy(pk["weight_scale"]).view(torch.float8_e4m3fn),
                               float(pk["weight_scale_2"]))
        return wq * s2
    raise ValueError(fmt)


# Non-expert precision presets for tracks mode (--track ne:NAME). "fp8": DESIGN decision 2, zai's FP8 values
# for the attention, indexer and shared-expert linears (the dense MLPs of layers 0-2 stay the checkpoint's
# per-tensor FP8 in every track, the routers, norms, indexer weights_proj, embedding and head as shipped);
# the others put some of them at NVFP4 round-to-nearest (NVIDIA's recipe, quant/formats.py nvfp4_rtn).
NE_FP8 = ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
          "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight", "self_attn.indexer.wq_b.weight",
          "self_attn.indexer.wk.weight", "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
          "mlp.shared_experts.down_proj.weight"]
NE_NVFP4 = {"fp8": [], "o4": ["self_attn.o_proj.weight"],
            "oqk4": ["self_attn.o_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_b_proj.weight"],
            "oqks4": ["self_attn.o_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_b_proj.weight",
                      "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
                      "mlp.shared_experts.down_proj.weight"]}


def ne_preset(name):
    q = {n: "fp8b" for n in NE_FP8}
    q.update({n: "nvfp4" for n in NE_NVFP4[name]})
    return q


def indexer_topk(x, cq, W, cos, sin, block):
    """Top-2048 key sets for the rows at positions >= 2048 (rows below see every causal key).
    Returns (idx int32 [S-2048, 2048] sorted ascending, margin f32 [S-2048]: the 2048th score minus
    the 2049th, 0 for a tie at the boundary), or (None, None) when S <= 2048."""
    S = x.shape[0]
    if S <= TOPK:
        return None, None
    qi = linear(cq, W("self_attn.indexer.wq_b.weight")).view(S, NI, DI)
    ki = F.layer_norm(linear(x, W("self_attn.indexer.wk.weight")), (DI,),
                      W("self_attn.indexer.k_norm.weight"), W("self_attn.indexer.k_norm.bias"), 1e-6)
    qi = torch.cat([rope(qi[..., :DR], cos[:, None], sin[:, None]), qi[..., DR:]], dim=-1)
    ki = torch.cat([rope(ki[..., :DR], cos, sin), ki[..., DR:]], dim=-1)
    wts = F.linear(x.to(F32), W("self_attn.indexer.weights_proj.weight").to(F32)) * NI ** -0.5  # [S, 32]
    kf = ki.to(F32)
    idx_out = torch.empty(S - TOPK, TOPK, dtype=torch.int32)
    margin = torch.empty(S - TOPK, dtype=F32)
    for i0 in range(TOPK, S, block):
        i1 = min(S, i0 + block)
        sc = torch.relu(torch.matmul(qi[i0:i1].to(F32), kf[:i1].T) * DI ** -0.5)       # [B, 32, i1]
        I = torch.matmul(wts[i0:i1].unsqueeze(1), sc).squeeze(1)                         # [B, i1]
        del sc
        pos = torch.arange(i0, i1)
        I.masked_fill_(torch.arange(i1)[None, :] > pos[:, None], float("-inf"))
        sel = I.topk(TOPK, dim=-1).indices
        kth = I.gather(1, sel).min(dim=1).values
        nxt = I.scatter(1, sel, float("-inf")).max(dim=1).values
        margin[i0 - TOPK:i1 - TOPK] = kth - nxt
        idx_out[i0 - TOPK:i1 - TOPK] = sel.sort(dim=1).values.to(torch.int32)
    return idx_out, margin


def attention(x, W, full, prev_topk, cos, sin, args):
    """MLA over one sequence (positions 0..S-1). Returns (out [S, D] bf16, topk, margin)."""
    S = x.shape[0]
    cq = rms_norm(linear(x, W("self_attn.q_a_proj.weight")), W("self_attn.q_a_layernorm.weight"), 1e-6)
    q = linear(cq, W("self_attn.q_b_proj.weight")).view(S, NH, DQK)
    q = torch.cat([q[..., :DN], rope(q[..., DN:], cos[:, None], sin[:, None])], dim=-1)  # [S, H, 256]
    ckv = linear(x, W("self_attn.kv_a_proj_with_mqa.weight"))
    c = rms_norm(ckv[:, :RKV], W("self_attn.kv_a_layernorm.weight"), 1e-6)
    k_rot = rope(ckv[:, RKV:], cos, sin)                                                 # [S, 64]
    kv = linear(c, W("self_attn.kv_b_proj.weight")).view(S, NH, DN + DV)
    K = torch.empty(NH, S, DQK, dtype=BF16)
    K[:, :, :DN] = kv[:, :, :DN].transpose(0, 1)
    K[:, :, DN:] = k_rot.unsqueeze(0)
    V = kv[:, :, DN:].transpose(0, 1).contiguous()                                       # [H, S, 256]
    del kv, ckv, c
    margin = None
    if full:
        topk, margin = indexer_topk(x, cq, W, cos, sin, args.index_block)
    else:
        topk = prev_topk
    del cq
    Wo = W("self_attn.o_proj.weight")
    out = torch.empty(S, D, dtype=F32 if COMBINE == "f32" else BF16)
    if COMBINE == "f32":
        Wo = Wo.to(F32)
    B, G = args.attn_block, args.head_group
    for i0 in range(0, S, B):
        i1 = min(S, i0 + B)
        pos = torch.arange(i0, i1)
        allowed = torch.arange(i1)[None, :] <= pos[:, None]                               # causal
        if topk is not None and i1 > TOPK:
            r0 = max(i0, TOPK)
            sub = torch.zeros(i1 - r0, i1, dtype=torch.bool)
            sub.scatter_(1, topk[r0 - TOPK:i1 - TOPK].long(), True)
            allowed[r0 - i0:] = sub
        blocked = ~allowed
        qb = q[i0:i1].transpose(0, 1)                                                     # [H, B, 256]
        o = torch.empty(NH, i1 - i0, DV, dtype=BF16)
        for h0 in range(0, NH, G):
            s = torch.matmul(qb[h0:h0 + G], K[h0:h0 + G, :i1].transpose(1, 2)) * ATT_SCALE   # bf16
            s.masked_fill_(blocked, float("-inf"))
            p = torch.softmax(s, dim=-1, dtype=F32).to(BF16)
            del s
            o[h0:h0 + G] = torch.matmul(p, V[h0:h0 + G, :i1])
            del p
        out[i0:i1] = out_linear(o.transpose(0, 1).reshape(i1 - i0, NH * DV), Wo)
    return out, topk, margin


def dense_mlp(x, W):
    """Layers 0-2: FP8 per-tensor weights, intermediate 12288."""
    g, sg = fp8(W.ck, W.p + ".mlp.gate_proj")
    u, su = fp8(W.ck, W.p + ".mlp.up_proj")
    d, sd = fp8(W.ck, W.p + ".mlp.down_proj")
    out = torch.empty(x.shape, dtype=F32 if COMBINE == "f32" else BF16)
    for i in range(0, x.shape[0], 2048):
        xi = x[i:i + 2048]
        h = silu_mul(scaled_linear(xi, g, sg), scaled_linear(xi, u, su))
        out[i:i + 2048] = out_scaled_linear(h, d, sd)
    return out


def bf16_mlp(x, W, p):
    return out_linear(silu_mul(linear(x, W(p + ".gate_proj.weight")), linear(x, W(p + ".up_proj.weight"))),
                  W(p + ".down_proj.weight"))


# ---------------------------------------------------------------------------------------------- expert sources
# A source gives one routed expert's weights in two steps: load(e) only reads (it runs on the I/O threads),
# prepare(raw) converts on the compute thread and returns ("scaled", (g, sg), (u, su), (d, sd)) with exact f32
# values and a per-tensor scale, or ("bf16", g, u, d).

class CkptExperts:
    """The checkpoint's own experts: NVFP4 for layers 3-77, BF16 for the MTP layer."""

    def __init__(self, ck, l):
        self.ck, self.p, self.quantised = ck, f"model.layers.{l}.mlp.experts", l != MTP

    def load(self, e):
        if self.quantised:
            return [nvfp4_raw(self.ck, f"{self.p}.{e}.{m}_proj") for m in ("gate", "up", "down")]
        return [self.ck.get(f"{self.p}.{e}.{m}_proj.weight") for m in ("gate", "up", "down")]

    def prepare(self, raw):
        if self.quantised:
            return ("scaled", *[nvfp4_dequant(*t) for t in raw])
        return ("bf16", *raw)


class MasterExperts:
    """BF16 master experts from fetch_master.py's directory (tensors/<name>.bin, raw little-endian bf16)."""

    SHAPES = {"gate": (DFF, D), "up": (DFF, D), "down": (D, DFF)}

    def __init__(self, root, l):
        self.root, self.p = root, f"model.layers.{l}.mlp.experts"

    @staticmethod
    def available(root, l):
        return os.path.exists(os.path.join(root, "tensors", f"model.layers.{l}.mlp.experts.255.down_proj.weight.bin"))

    def load(self, e):
        out = []
        for m in ("gate", "up", "down"):
            path = os.path.join(self.root, "tensors", f"{self.p}.{e}.{m}_proj.weight.bin")
            with open(path, "rb") as f:
                buf = bytearray(f.read())
            os.posix_fadvise(os.open(path, os.O_RDONLY), 0, 0, os.POSIX_FADV_DONTNEED)
            out.append(torch.frombuffer(buf, dtype=BF16).view(*self.SHAPES[m]))
        return out

    def prepare(self, raw):
        return ("bf16", *raw)


def quant_formats():
    """The quantiser's formats (../quant/formats.py), imported on first use."""
    qdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quant")
    if qdir not in sys.path:
        sys.path.insert(0, qdir)
    import formats
    return formats


def requantise(fmt, g, u, d):
    """BF16 expert -> the format's stored bytes -> exact f32 values and a per-tensor scale."""
    fm = quant_formats()
    g, u, d = g.to(F32), u.to(F32), d.to(F32)
    if fmt == "nvfp4_rtn":
        amax = torch.maximum(g.abs().max(), u.abs().max())     # gate and up share weight_scale_2
        out = []
        for w, a in ((g, amax), (u, amax), (d, None)):
            pk = fm.nvfp4_rtn(w, a)
            out.append(nvfp4_dequant(torch.from_numpy(pk["weight"]),
                                     torch.from_numpy(pk["weight_scale"]).view(torch.float8_e4m3fn),
                                     float(pk["weight_scale_2"])))
        return ("scaled", *out)
    raise ValueError(f"format {fmt}: not wired yet (only nvfp4_rtn until docs/quantiser.md names the others)")


class FormatExperts:
    """A base source's experts (BF16: the master or the MTP layer) requantised on the fly into `fmt`."""

    def __init__(self, base, fmt):
        self.base, self.fmt = base, fmt

    def load(self, e):
        return self.base.load(e)

    def prepare(self, raw):
        kind, *w = self.base.prepare(raw)
        assert kind == "bf16", "requantising needs BF16 experts"
        return requantise(self.fmt, *w)


# The quantiser's formats from qprep.py's cache (tracks mode, --qcache). A format name, optionally with the
# suffix "/sub": the format only on --sub-experts, nvfp4_rtn (NVIDIA's recipe) on the others.
#   s3x2gu_nvfp4d  gate and up S3x2 (e2m1sub3x2_g16), down nvfp4_rtn (its own amax)
#   s3x2           all three S3x2
#   k3, k4         all three trellis K3 / K4 (exllamav3's code, data-free; dequantised f32, scale 1)
QFORMATS = {"s3x2gu_nvfp4d": ("e2m1sub3x2_g16", "e2m1sub3x2_g16", "nvfp4"),
            "s3x2": ("e2m1sub3x2_g16",) * 3, "k3": ("trellis_k3",) * 3, "k4": ("trellis_k4",) * 3}


class CachedFormatExperts:
    """The master's experts in one of QFORMATS, read from qprep.py's cache; nvfp4_rtn for experts outside `only`."""

    def __init__(self, master, qcache, l, fmt, only=None):
        self.m, self.l, self.only = master, l, only
        self.parts = QFORMATS[fmt]
        self.root = os.path.join(os.path.expanduser(qcache), f"L{l}")

    def load(self, e):
        if self.only is not None and e not in self.only:
            return ("master", self.m.load(e))
        out = []
        need_master = "nvfp4" in self.parts
        bf = self.m.load(e) if need_master else None
        for i, (m, f) in enumerate(zip(("gate", "up", "down"), self.parts)):
            if f == "nvfp4":
                out.append(("nvfp4", bf[i]))
            elif f.startswith("e2m1"):
                z = np.load(os.path.join(self.root, f, f"{e}.{m}.npz"))
                out.append(("s3", torch.from_numpy(z["v2"]), torch.from_numpy(z["sc8"]), float(z["s2"])))
            else:
                out.append(("f32", torch.from_numpy(np.load(os.path.join(self.root, f, f"{e}.{m}.npy")))))
        return ("cached", out)

    def prepare(self, raw):
        kind, data = raw
        if kind == "master":
            return requantise("nvfp4_rtn", *data)
        fm = quant_formats()
        res = []
        for t in data:
            if t[0] == "nvfp4":
                pk = fm.nvfp4_rtn(t[1].to(F32))
                res.append(nvfp4_dequant(torch.from_numpy(pk["weight"]),
                                         torch.from_numpy(pk["weight_scale"]).view(torch.float8_e4m3fn),
                                         float(pk["weight_scale_2"])))
            elif t[0] == "s3":     # (E2M1 level x e4m3 group scale), exact in f32; s2 applied after the GEMM
                v2, sc8, s2 = t[1], t[2], t[3]
                R = v2.shape[0]
                w = ((v2.to(F32) / 2).view(R, -1, 16) * sc8.view(torch.float8_e4m3fn).to(F32).unsqueeze(-1)).view(R, -1)
                res.append((w, s2))
            else:
                res.append((t[1], 1.0))
        return ("scaled", *res)


def moe(x, W, l, pool, src=None, hook=None):
    """Routed experts + shared expert. Returns (out bf16, routing record, calibration sums)."""
    N = x.shape[0]
    logits = F.linear(x.to(F32), W("mlp.gate.weight").to(F32))
    s = logits.sigmoid()
    sel = s + W("mlp.gate.e_score_correction_bias")
    idx = sel.topk(K8, dim=-1, sorted=True).indices                                       # [N, 8]
    eighth = sel.gather(1, idx[:, -1:]).squeeze(1)
    margin = eighth - sel.scatter(1, idx, float("-inf")).max(dim=1).values
    w = s.gather(1, idx)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) * ROUTE_SCALE
    del logits, sel
    src = src or CkptExperts(W.ck, l)
    acc = torch.zeros(N, D, dtype=F32)
    cnt = torch.zeros(NE, dtype=torch.int64)
    mass = torch.zeros(NE, dtype=torch.float64)
    gu_sq = torch.zeros(NE, D, dtype=F32)
    dn_sq = torch.zeros(NE, DFF, dtype=F32)
    hit = torch.zeros(NE, dtype=torch.bool)
    hit[idx.unique()] = True
    experts = [e for e in range(NE) if hit[e]]
    futs = {}

    def prefetch(j):
        if j < len(experts) and experts[j] not in futs:
            e = experts[j]
            futs[e] = pool.submit(src.load, e)

    for j in range(min(3, len(experts))):
        prefetch(j)
    for j, e in enumerate(experts):
        prefetch(j + 3)
        kind, *wts = src.prepare(futs.pop(e).result())
        r, k = (idx == e).nonzero(as_tuple=True)
        xe = x[r]
        if kind == "scaled":
            (g, sg), (u, su), (d, sd) = wts
            h = silu_mul(scaled_linear(xe, g, sg), scaled_linear(xe, u, su))
            y = out_scaled_linear(h, d, sd)
        else:
            g, u, d = wts
            h = silu_mul(linear(xe, g), linear(xe, u))
            y = out_linear(h, d)
        if hook is not None:
            hook(e, xe)
        we = w[r, k]
        acc.index_add_(0, r, y.to(F32) * we[:, None])
        cnt[e] = len(r)
        mass[e] = we.double().sum()
        gu_sq[e] = xe.to(F32).pow(2).sum(0)
        dn_sq[e] = h.to(F32).pow(2).sum(0)
        del wts, g, u, d, xe, h, y
    if COMBINE == "f32":
        out = acc + bf16_mlp(x, W, "mlp.shared_experts")                                  # f32, rounded with the residual
    else:
        out = acc.to(BF16)
        del acc
        out = out + bf16_mlp(x, W, "mlp.shared_experts")
    route = {"route_idx": idx.to(torch.uint8).numpy(), "route_w": w.numpy(), "route_margin": margin.numpy()}
    calib = {"count": cnt.numpy(), "mass": mass.numpy(), "gate_up_in_sqsum": gu_sq.numpy(),
             "down_in_sqsum": dn_sq.numpy()}
    return out, route, calib


# ---------------------------------------------------------------------------------------------- driver

class Run:
    def __init__(self, args):
        self.args = args
        self.ck = Ckpt(args.ckpt)
        with open(os.path.join(args.ckpt, "config.json")) as f:
            cfg = json.load(f)
        assert (cfg["hidden_size"], cfg["num_hidden_layers"], cfg["n_routed_experts"], cfg["index_topk"],
                cfg["q_lora_rank"], cfg["kv_lora_rank"], cfg["vocab_size"]) == (D, NL, NE, TOPK, RQ, RKV, VOCAB)
        self.full = [t == "full" for t in cfg["indexer_types"]]
        self.out = args.out_dir
        self.state = os.path.join(self.out, "state")
        os.makedirs(self.state, exist_ok=True)
        self.seqs = []
        off = 0
        for path in args.tokens:
            with open(path) as f:
                t = json.load(f)
            ids = t["tokens"]
            assert len(ids) <= 65536, "index sets are stored as uint16"
            self.seqs.append({"name": t["name"], "off": off, "n": len(ids), "ids": ids})
            off += len(ids)
        self.N = off
        self.ids = torch.tensor([i for s in self.seqs for i in s["ids"]], dtype=torch.int64)
        self.sample = self.sample_rows()
        self.pool = ThreadPoolExecutor(4)
        self.write_manifest()

    def sample_rows(self):
        rows = []
        for s in self.seqs:
            n = s["n"]
            if n <= self.args.all_positions_below:
                ps = range(n)
            else:
                ps = sorted({p for p in range(n) if p % self.args.sample_every == 0 or p < 32 or p >= n - 32}
                            | {p for p in (TOPK - 1, TOPK, TOPK + 1) if p < n})
            rows += [s["off"] + p for p in ps]
        return torch.tensor(rows, dtype=torch.int64)

    def write_manifest(self):
        m = {"ckpt": os.path.basename(os.path.normpath(self.args.ckpt)),
             "sequences": [{k: s[k] for k in ("name", "off", "n")} for s in self.seqs],
             "rows": self.N, "contract": "glmref.py docstring", "combine": COMBINE, "files": FILES}
        with open(os.path.join(self.out, "manifest.json"), "w") as f:
            json.dump(m, f, indent=1)
        np.save(os.path.join(self.out, "tokens.npy"), self.ids.to(torch.int32).numpy())
        np.save(os.path.join(self.out, "sample_rows.npy"), self.sample.to(torch.int32).numpy())

    # -- state
    def progress(self):
        p = os.path.join(self.state, "progress.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return {"next": 0}

    def save_state(self, nxt, **tensors):
        for k, v in tensors.items():
            torch.save(v, os.path.join(self.state, k + ".tmp"))
            os.replace(os.path.join(self.state, k + ".tmp"), os.path.join(self.state, k + ".pt"))
        with open(os.path.join(self.state, "progress.tmp"), "w") as f:
            json.dump({"next": nxt, "combine": COMBINE}, f)
        os.replace(os.path.join(self.state, "progress.tmp"), os.path.join(self.state, "progress.json"))

    def load_state(self, k):
        return torch.load(os.path.join(self.state, k + ".pt"))

    def ldir(self, l):
        d = os.path.join(self.out, "layers", f"L{l:02d}")
        os.makedirs(d, exist_ok=True)
        return d

    def topk_from_disk(self, l):
        """The index sets of the last full layer at or below l, as saved (resume at a shared layer)."""
        lf = max(i for i in range(l + 1) if self.full[i])
        sets = {}
        for s in self.seqs:
            p = os.path.join(self.ldir(lf), f"topk_{s['name']}.npy")
            sets[s["name"]] = torch.from_numpy(np.load(p).astype(np.int32)) if os.path.exists(p) else None
        return sets

    # -- one decoder layer over all sequences
    def layer(self, l, h, prev, seqs, src=None, quant=None, cache=None, hook=None):
        W = LayerW(self.ck, l, quant, cache)
        t0 = time.time()
        x = rms_norm(h, W("input_layernorm.weight"), 1e-5)
        a = torch.empty_like(h)
        topk, saved = {}, {}
        for s in seqs:
            r0, n = s["off"], s["n"]
            cos, sin = self.rope[:n], self.rope_s[:n]
            a[r0:r0 + n], tk, mg = attention(x[r0:r0 + n], W, self.full[l] if l != MTP else True,
                                             prev.get(s["name"]), cos, sin, self.args)
            topk[s["name"]] = tk
            if mg is not None:
                saved[s["name"]] = (tk, mg)
        t_att = time.time() - t0
        h = add_residual(h, a)
        del a, x
        x = rms_norm(h, W("post_attention_layernorm.weight"), 1e-5)
        route = calib = None
        if l < 3:
            m = dense_mlp(x, W)
        else:
            m, route, calib = moe(x, W, l, self.pool, src, hook)
        h = add_residual(h, m)
        del m, x
        return h, topk, saved, route, calib, t_att, time.time() - t0

    def write_layer(self, l, h, saved, route, calib, times, rows_map=None):
        d = self.ldir(l)
        for name, (tk, mg) in saved.items():
            np.save(os.path.join(d, f"topk_{name}.npy"), tk.numpy().astype(np.uint16))
            np.save(os.path.join(d, f"topk_margin_{name}.npy"), mg.numpy())
        if route is not None:
            for k, v in route.items():
                np.save(os.path.join(d, k + ".npy"), v)
            np.savez(os.path.join(d, "calib.npz"), **calib)
        rows = self.sample if rows_map is None else rows_map
        np.save(os.path.join(d, "hidden.npy"), h[rows].to(F32).numpy())
        with open(os.path.join(d, "timing.json"), "w") as f:
            json.dump(times, f)

    def head_outputs(self, hn, rows_next, d, full_rows):
        """Logits in f32 from the normed hidden: top-20 logprobs, lse, the next token's logprob, and
        the full logits at full_rows (indices into hn)."""
        os.makedirs(d, exist_ok=True)
        Wh = self.ck.get("lm_head.weight").to(F32)
        n = hn.shape[0]
        top_i = np.empty((n, 20), np.int32)
        top_lp = np.empty((n, 20), np.float32)
        lse = np.empty(n, np.float32)
        nxt = np.full(n, np.nan, np.float32)
        full = np.empty((len(full_rows), VOCAB), np.float32)
        fr = {int(r): j for j, r in enumerate(full_rows)}
        for i in range(0, n, 1024):
            lg = F.linear(hn[i:i + 1024].to(F32), Wh)                                     # [B, V] f32
            ls = torch.logsumexp(lg, dim=-1)
            tv, ti = lg.topk(20, dim=-1)
            top_i[i:i + 1024] = ti.numpy()
            top_lp[i:i + 1024] = (tv - ls[:, None]).numpy()
            lse[i:i + 1024] = ls.numpy()
            for j in range(lg.shape[0]):
                t = int(rows_next[i + j])
                if t >= 0:
                    nxt[i + j] = float(lg[j, t] - ls[j])
                if i + j in fr:
                    full[fr[i + j]] = lg[j].numpy()
        del Wh
        np.save(os.path.join(d, "top20_ids.npy"), top_i)
        np.save(os.path.join(d, "top20_logprobs.npy"), top_lp)
        np.save(os.path.join(d, "lse.npy"), lse)
        np.save(os.path.join(d, "next_logprob.npy"), nxt)
        np.save(os.path.join(d, "logits_rows.npy"), np.asarray(full_rows, np.int32))
        np.save(os.path.join(d, "logits.npy"), full)
        return top_i[:, 0]

    def run(self):
        a = self.args
        nmax = max(s["n"] for s in self.seqs)
        self.rope, self.rope_s = rope_tables(nmax)
        if a.mtp_from:
            return self.run_mtp_from()
        if a.track:
            return self.run_tracks()
        prog = self.progress()
        if prog["next"] != 0:  # a resume must keep the run's rounding order (runs before --combine were bf16)
            assert prog.get("combine", "bf16") == COMBINE, f"this run used --combine {prog.get('combine', 'bf16')}"
        nxt = prog["next"]
        last = a.layers if a.layers is not None else NL
        if nxt == 0:
            h = self.ck.rows("model.embed_tokens.weight", self.ids.tolist())
            np.save(os.path.join(self.out, "embed_sample.npy"), h[self.sample].to(F32).numpy())
            prev = {}
        elif isinstance(nxt, int):
            h = self.load_state("h")
            prev = self.topk_from_disk(nxt - 1)
            log(f"resuming at layer {nxt}")
        t_pass = time.time()
        if isinstance(nxt, int):
            for l in range(nxt, last):
                hook = self.cov_hook(l) if a.cov_out and l in a.cov_layers else None
                h, topk, saved, route, calib, t_att, t_all = self.layer(l, h, prev, self.seqs, None, hook=hook)
                prev = topk
                self.write_layer(l, h, saved, route, calib, {"attention_s": t_att, "layer_s": t_all})
                self.save_state(l + 1, h=h)
                log(f"layer {l:2d} {'full' if self.full[l] else 'shared'} {t_all:7.1f} s (attention {t_att:6.1f} s)"
                    f"  rss {rss_gib():.1f} GiB")
                if a.cov_out and l >= max(a.cov_layers):
                    log(f"covariances done (layers {a.cov_layers}); stopping after layer {l}")
                    return
            nxt = "head"
            hn = rms_norm(h, self.ck.get("model.norm.weight"), 1e-5)
            self.save_state("head", hfinal=hn)
        if nxt == "head":
            hn = self.load_state("hfinal")
            t0 = time.time()
            nexts = torch.full((self.N,), -1, dtype=torch.int64)
            for s in self.seqs:
                nexts[s["off"]:s["off"] + s["n"] - 1] = self.ids[s["off"] + 1:s["off"] + s["n"]]
            full_rows = self.sample.tolist()
            np.save(os.path.join(self.out, "final_hidden_sample.npy"), hn[self.sample].to(F32).numpy())
            argmax = self.head_outputs(hn, nexts, os.path.join(self.out, "final"), full_rows)
            np.save(os.path.join(self.out, "final", "argmax.npy"), argmax)
            log(f"head {time.time() - t0:.1f} s")
            self.save_state("mtp" if not a.no_mtp else "done")
            nxt = "mtp" if not a.no_mtp else "done"
        if nxt == "mtp":
            self.run_mtp()
            self.save_state("done")
        log(f"pass done ({time.time() - t_pass:.0f} s this invocation)")

    def cov_hook(self, l):
        """Experiment 3b: per routed expert of layer l, the second moments of its inputs over the rows routed to
        it: gate/up input x (the post-attention RMSNorm output) and down input h = silu(x Wg^T) * (x Wu^T) with
        the BF16 master's weights (bf16 GEMMs, as the reference). Saved for quant/run.py --cov as
        layer{l}_expert{e}.pt: {'gate_up': f16 [6144, 6144], 'down': f16 [2048, 2048], 'count': rows,
        'scale': {'gate_up': s, 'down': s}}, each matrix the f32 mean x x^T (f32 GEMM) divided by its largest
        diagonal entry s (run.py's ratio does not depend on the scale)."""
        a = self.args
        m = MasterExperts(os.path.expanduser(a.master), l)
        d = os.path.expanduser(a.cov_out)
        os.makedirs(d, exist_ok=True)

        def norm16(S):
            s = float(S.diagonal().max())
            return (S / s).to(torch.float16), s

        def hook(e, xe):
            g, u, _ = m.load(e)
            xf = xe.to(F32)
            hm = silu_mul(linear(xe, g), linear(xe, u)).to(F32)
            n = xe.shape[0]
            sx, kx = norm16(xf.T @ xf / n)
            sh, kh = norm16(hm.T @ hm / n)
            path = os.path.join(d, f"layer{l}_expert{e}.pt")
            torch.save({"gate_up": sx, "down": sh, "count": n, "scale": {"gate_up": kx, "down": kh}}, path + ".tmp")
            os.replace(path + ".tmp", path)
        return hook

    def run_mtp_from(self):
        """Experiment 5: replay the MTP layer over a finished run's final hidden states (its state/hfinal.pt),
        with the MTP experts from --mtp-experts. The run's final/ is linked in for the comparisons."""
        a = self.args
        src_ids = np.load(os.path.join(a.mtp_from, "tokens.npy"))
        assert np.array_equal(src_ids, self.ids.numpy()), "--mtp-from run has other tokens"
        fin = os.path.join(self.out, "final")
        if not os.path.exists(fin):
            os.symlink(os.path.join(os.path.abspath(a.mtp_from), "final"), fin)
        hn = torch.load(os.path.join(a.mtp_from, "state", "hfinal.pt"))
        src = None if a.mtp_experts == "bf16" else FormatExperts(CkptExperts(self.ck, MTP), a.mtp_experts)
        self.run_mtp(hn, src)
        log("MTP replay done")

    def run_tracks(self):
        """Experiment 3: carry a master track and one track per --track FORMAT@LAYERS through the pass.
        Master track: BF16 master experts where --master has the layer, the checkpoint's NVFP4 elsewhere.
        A format track: at its layers that the master has, the master's experts requantised into FORMAT;
        everywhere else the master track's experts. Reports per layer the hidden-state difference and the
        routing-set agreement against the master track, and at the end KL(master || track) and top-1."""
        a = self.args
        mroot = os.path.expanduser(a.master) if a.master else None
        have = {l for l in range(3, NL) if mroot and MasterExperts.available(mroot, l)}
        tracks = [("master", None, set())]
        quant = {"master": None}
        for t in a.track:
            if t.startswith("ne:"):                     # non-expert precision preset, experts as the master's
                tracks.append((t, None, set()))
                quant[t] = ne_preset(t[3:])
                continue
            fmt, _, ls = t.partition("@")
            lay = have if ls in ("", "all") else {int(x) for x in ls.split(",")} & have
            tracks.append((t, fmt, lay))
            quant[t] = None
        refs = ["master"] + [r for r in a.track_ref if r in quant]
        log(f"master layers {sorted(have)}; tracks {[(n, sorted(l)) for n, _, l in tracks]}")

        sub = {int(x) for x in a.sub_experts.split(",") if x}

        def source(fmt, lay, l):
            if l < 3 or l not in have:
                return None
            m = MasterExperts(mroot, l)
            if l not in lay:
                return m
            base, _, only = fmt.partition("/")
            if base in QFORMATS:
                return CachedFormatExperts(m, a.qcache, l, base, sub if only == "sub" else None)
            assert not only, fmt
            return FormatExperts(m, fmt)

        def signature(n, fmt, lay, l):
            """The weights a track uses at layer l. Two tracks entering layer l with the same state object and the
            same signature leave it with the same state: it is computed once and shared (never mutated)."""
            return (repr(quant[n]), fmt if l in lay else None)

        names = [n for n, _, _ in tracks]
        kind = {n: ("ne" if n.startswith("ne:") else "fmt" if f else "master") for n, f, _ in tracks}
        lays = {n: frozenset(y) for n, _, y in tracks}

        def applies(r, n):
            """A reference is compared with the master-relative tracks of its own kind and layer set."""
            if r == n:
                return False
            if r == "master":
                return True
            return kind[r] == kind[n] and (kind[r] == "ne" or lays[r] == lays[n])
        pairs = [(r, n) for r in refs for n in names if applies(r, n)
                 and not (n in refs and r != "master" and names.index(n) < names.index(r))]

        # Disk-backed states (RSS stays near one track's working set): state/tracks/L<next>/s<k>.pt holds one
        # distinct state {"H": residual stream bf16 [N, 6144], "prev": the index sets}, and index.json maps
        # every track to its state, with the report so far. Tracks that enter a layer with the same state and use
        # the same weights there share the output state: computed once. A kill resumes at the last whole layer.
        sroot = os.path.join(self.state, "tracks")
        os.makedirs(sroot, exist_ok=True)
        tid = {"tracks": [(n, f, sorted(y)) for n, f, y in tracks], "refs": refs, "combine": COMBINE,
               "tokens": [s["name"] for s in self.seqs], "N": self.N, "sub": sorted(sub), "qcache": a.qcache}
        cur = os.path.join(sroot, "current.json")

        def sfile(l, k):
            return os.path.join(sroot, f"L{l}", f"s{k}.pt")

        if os.path.exists(cur):
            with open(cur) as f:
                st = json.load(f)
            assert st["id"] == json.loads(json.dumps(tid)), "state/tracks is from a different track set: new --out-dir"
            first, smap, report = st["next"], st["map"], st["report"]
            report["layers"] = {int(k): v for k, v in report["layers"].items()}
            log(f"resuming tracks at layer {first}")
        else:
            report = {"tracks": names, "master_layers": sorted(have), "layers": {}, "final": {}}
            os.makedirs(os.path.join(sroot, "L0"), exist_ok=True)
            torch.save({"H": self.ck.rows("model.embed_tokens.weight", self.ids.tolist()), "prev": {}}, sfile(0, 0))
            first, smap = 0, {n: 0 for n in names}
        last = a.layers if a.layers is not None else NL
        for l in range(first, last):
            t0 = time.time()
            os.makedirs(os.path.join(sroot, f"L{l + 1}"), exist_ok=True)
            done, nmap, routes = {}, {}, {}
            for n, fmt, lay in tracks:
                key = (smap[n], signature(n, fmt, lay, l))
                if key not in done:
                    st = torch.load(sfile(l, smap[n]), weights_only=False)
                    h, pv, _, rt, _, _, _ = self.layer(l, st["H"], st["prev"], self.seqs,
                                                       source(fmt, lay, l) if fmt else source(None, set(), l),
                                                       quant[n], None)
                    del st
                    k = len(done)
                    torch.save({"H": h, "prev": pv}, sfile(l + 1, k))
                    routes[k] = None if rt is None else rt["route_idx"]
                    done[key] = k
                    del h, pv, rt
                nmap[n] = done[key]
            km = nmap["master"]
            hm = torch.load(sfile(l + 1, km), weights_only=False)["H"].to(F32)
            rel_k = {}
            for k in set(nmap.values()):
                hk = hm if k == km else torch.load(sfile(l + 1, k), weights_only=False)["H"].to(F32)
                rel_k[k] = ((hk - hm).norm(dim=1) / hm.norm(dim=1)).numpy()
                del hk
            del hm
            rl = {}
            for n in names[1:]:
                k = nmap[n]
                r = {"hidden_rel_median": float(np.median(rel_k[k])), "hidden_rel_max": float(rel_k[k].max())}
                if routes[k] is not None:
                    r["route_sets_equal"] = float((np.sort(routes[k], 1) == np.sort(routes[km], 1)).all(1).mean())
                rl[n] = r
            report["layers"][l] = rl
            log(f"layer {l:2d} {time.time() - t0:6.1f} s, {len(done)} distinct  " + "  ".join(
                f"{n}: rel {v['hidden_rel_median']:.2e} routes {v.get('route_sets_equal', 1):.4f}" for n, v in rl.items())
                + f"  rss {rss_gib():.1f} GiB")
            with open(os.path.join(self.out, "tracks.json"), "w") as f:
                json.dump(report, f, indent=1)
            with open(cur + ".tmp", "w") as f:
                json.dump({"id": tid, "next": l + 1, "map": nmap, "report": report}, f)
            os.replace(cur + ".tmp", cur)
            shutil.rmtree(os.path.join(sroot, f"L{l}"), ignore_errors=True)
            smap = nmap
        self.tracks_head(names, refs, pairs, smap, report, lambda k: sfile(last, k))

    def tracks_head(self, names, refs, pairs, smap, report, sfile):
        """The head on every distinct final state, streamed over row blocks and vocabulary slabs (RSS stays small):
        f32 logits (as head_outputs), and per row in f64 with online rescaling: each state's logsumexp and
        argmax; per pair (r, n) KL(r || n) = sum p_r (z_r - z_n) - lse_r + lse_n, the top-1 agreement and the
        largest |z_n - z_r|."""
        t0 = time.time()
        Wn = self.ck.get("model.norm.weight")
        ks = sorted(set(smap.values()))
        hdir = os.path.join(self.state, "tracks", "hn")
        os.makedirs(hdir, exist_ok=True)
        for k in ks:                       # the normed final states, raw bf16 on disk, read a row block at a time
            x = rms_norm(torch.load(sfile(k), weights_only=False)["H"], Wn, 1e-5).contiguous()
            x.view(torch.int16).numpy().tofile(os.path.join(hdir, f"s{k}.bin"))
            del x

        class Rows:
            def __init__(self, k):
                self.path = os.path.join(hdir, f"s{k}.bin")

            def __getitem__(self, sl):
                w = D_ * 2
                with open(self.path, "rb") as f:
                    f.seek(sl.start * w)
                    buf = bytearray(f.read((sl.stop - sl.start) * w))
                return torch.frombuffer(buf, dtype=BF16).view(sl.stop - sl.start, D_)
        D_ = D
        hn = {k: Rows(k) for k in ks}
        N, B, C = self.N, 1024, 8192
        lse = {k: np.empty(N) for k in ks}
        amx = {k: np.empty(N, np.int64) for k in ks}
        kl = {p: np.empty(N) for p in pairs}
        dmax = {p: np.empty(N) for p in pairs}
        kp = {p: (smap[p[0]], smap[p[1]]) for p in pairs}
        rk = sorted({smap[r] for r in refs})
        for i0 in range(0, N, B):
            i1 = min(N, i0 + B)
            m = {k: torch.full((i1 - i0,), -float("inf"), dtype=torch.float64) for k in ks}
            e = {k: torch.zeros(i1 - i0, dtype=torch.float64) for k in ks}
            am = {k: torch.zeros(i1 - i0, dtype=torch.int64) for k in ks}
            A = {p: torch.zeros(i1 - i0, dtype=torch.float64) for p in pairs}
            DM = {p: torch.zeros(i1 - i0, dtype=torch.float64) for p in pairs}
            xb = {k: hn[k][slice(i0, i1)] for k in ks}                                     # bf16 row blocks
            for v0 in range(0, VOCAB, C):
                v1 = min(VOCAB, v0 + C)
                Wc = self.ck.slab("lm_head.weight", v0, v1).to(F32)

                def update(k, z):
                    cm, ci = z.max(-1)
                    am[k] = torch.where(cm > m[k], ci + v0, am[k])     # first index of the maximum overall
                    mn = torch.maximum(m[k], cm)
                    sc = torch.exp(m[k] - mn)
                    e[k] = e[k] * sc + torch.exp(z - mn[:, None]).sum(-1)
                    m[k] = mn
                    return sc

                zr = {k: F.linear(xb[k].to(F32), Wc).double() for k in rk}
                scr = {k: update(k, zr[k]) for k in rk}
                for p in pairs:
                    A[p] = A[p] * scr[kp[p][0]]
                for k in ks:
                    z = zr[k] if k in zr else F.linear(xb[k].to(F32), Wc).double()
                    if k not in zr:
                        update(k, z)
                    for p in pairs:
                        a_, b_ = kp[p]
                        if b_ == k:
                            A[p] = A[p] + (torch.exp(zr[a_] - m[a_][:, None]) * (zr[a_] - z)).sum(-1)
                            DM[p] = torch.maximum(DM[p], (z - zr[a_]).abs().max(-1).values)
                    del z
                del zr, Wc
            for k in ks:
                lse[k][i0:i1] = (m[k] + e[k].log()).numpy()
                amx[k][i0:i1] = am[k].numpy()
            for p in pairs:
                a_, b_ = kp[p]
                kl[p][i0:i1] = (A[p] / e[a_]).numpy() - lse[a_][i0:i1] + lse[b_][i0:i1]
                dmax[p][i0:i1] = DM[p].numpy()
            log(f"head rows {i1}/{N}  rss {rss_gib():.1f} GiB")
        ids = self.ids.numpy()
        tag = lambda x: x.replace("@", "_at_").replace(",", "-").replace(":", "_").replace("/", "_")
        for p in pairs:
            r, n = p
            a_, b_ = kp[p]
            same = amx[a_] == amx[b_]
            res = {}
            for sq in self.seqs:
                o, c = sq["off"], sq["n"]
                res[sq["name"]] = {"kl_mean": float(kl[p][o:o + c].mean()),
                                   "kl_p99": float(np.percentile(kl[p][o:o + c], 99)),
                                   "kl_max": float(kl[p][o:o + c].max()),
                                   "top1_agree": float(same[o:o + c].mean()),
                                   "max_abs_dlogit": float(dmax[p][o:o + c].max()),
                                   "mean_max_abs_dlogit": float(dmax[p][o:o + c].mean())}
            report["final"][f"{n} vs {r}"] = res
            np.save(os.path.join(self.out, f"kl_{tag(n)}_vs_{tag(r)}.npy"), kl[p])
        amm = amx[smap["master"]]
        report["final"]["master_top1_vs_next"] = {
            sq["name"]: float((amm[sq["off"]:sq["off"] + sq["n"] - 1] == ids[sq["off"] + 1:sq["off"] + sq["n"]]).mean())
            for sq in self.seqs}
        with open(os.path.join(self.out, "tracks.json"), "w") as f:
            json.dump(report, f, indent=1)
        log(f"head {time.time() - t0:.0f} s")
        log("tracks final " + json.dumps(report["final"]))

    def run_mtp(self, hn=None, src=None):
        t0 = time.time()
        hn = self.load_state("hfinal") if hn is None else hn
        W = LayerW(self.ck, MTP)
        # MTP rows: for each sequence, rows t = 0..n-2 (input: hidden of t, token t+1; target: token t+2)
        mseqs, rows_main, tok_in = [], [], []
        off = 0
        for s in self.seqs:
            n = s["n"] - 1
            mseqs.append({"name": s["name"], "off": off, "n": n})
            rows_main += range(s["off"], s["off"] + n)
            tok_in += s["ids"][1:]
            off += n
        rows_main = torch.tensor(rows_main)
        e = self.ck.rows("model.embed_tokens.weight", tok_in)
        for s in mseqs:
            e[s["off"]] = 0  # position 0 (vLLM DeepSeekMTP: inputs_embeds zeroed where positions == 0)
        e = rms_norm(e, W("enorm.weight"), 1e-5)
        g = rms_norm(hn[rows_main], W("hnorm.weight"), 1e-5)
        del hn
        x = linear(torch.cat([e, g], dim=-1), W("eh_proj.weight"))
        del e, g
        keep = set(self.sample.tolist())
        msample = torch.tensor([i for i, r in enumerate(rows_main.tolist()) if r in keep])
        h, topk, saved, route, calib, t_att, t_all = self.layer(MTP, x, {}, mseqs, src)
        self.write_layer(MTP, h, saved, route, calib, {"attention_s": t_att, "layer_s": t_all}, rows_map=msample)
        log(f"layer 78 (MTP) {t_all:.1f} s (attention {t_att:.1f} s)")
        hd = rms_norm(h, W("shared_head.norm.weight"), 1e-5)
        targets = torch.full((h.shape[0],), -1, dtype=torch.int64)
        for s, ms in zip(self.seqs, mseqs):
            targets[ms["off"]:ms["off"] + ms["n"] - 1] = self.ids[s["off"] + 2:s["off"] + s["n"]]
        d = os.path.join(self.out, "mtp")
        argmax = self.head_outputs(hd, targets, d, msample.tolist())
        np.save(os.path.join(d, "argmax.npy"), argmax)
        np.save(os.path.join(d, "rows_main.npy"), rows_main.to(torch.int32).numpy())
        log(f"MTP {time.time() - t0:.1f} s")


FILES = {
    "tokens.npy": "[N] int32, all sequences concatenated (manifest.sequences gives offsets)",
    "sample_rows.npy": "[P] int32, rows whose hidden states and full logits are saved",
    "embed_sample.npy": "[P, 6144] f32 (bf16 values), the embedding rows",
    "layers/LNN/hidden.npy": "[P, 6144] f32 (bf16 values), residual stream after layer NN",
    "layers/LNN/topk_SEQ.npy": "full-indexer layers: [n-2048, 2048] uint16, key positions chosen for positions 2048..n-1, ascending",
    "layers/LNN/topk_margin_SEQ.npy": "[n-2048] f32, the 2048th index score minus the 2049th (0 = a tie at the boundary)",
    "layers/LNN/route_idx.npy": "MoE layers: [N, 8] uint8, experts chosen, by descending biased score",
    "layers/LNN/route_w.npy": "[N, 8] f32, their weights (normalised, x 2.5)",
    "layers/LNN/route_margin.npy": "[N] f32, 8th minus 9th biased score",
    "layers/LNN/calib.npz": "count [256] int64, mass [256] f64 (sum of weights), gate_up_in_sqsum [256, 6144] f32 and down_in_sqsum [256, 2048] f32: per-expert sums of x^2 of the inputs (diagonal of X^T X)",
    "final_hidden_sample.npy": "[P, 6144] f32, after model.norm",
    "final/top20_ids.npy, top20_logprobs.npy": "[N, 20] int32 / f32, from f32 logits",
    "final/lse.npy, next_logprob.npy, argmax.npy": "[N] logsumexp, log p(next token) (NaN at a sequence's last row), argmax",
    "final/logits.npy, logits_rows.npy": "[P, 154880] f32 full logits at the sample rows",
    "layers/L78/*": "the MTP layer, over the MTP rows (row t of a sequence: hidden t, token t+1)",
    "mtp/*": "as final/, for the MTP rows; next_logprob is log p(token t+2); rows_main maps MTP rows to main rows",
}


def rss_gib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 2**20
    return 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokens", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=None, help="stop after this many layers (a wiring test; logits then meaningless)")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--mtp-from", default=None, help="experiment 5: replay only the MTP layer over this finished run")
    ap.add_argument("--mtp-experts", default="bf16", help="with --mtp-from: bf16 (as shipped) or a format, e.g. nvfp4_rtn")
    ap.add_argument("--track", action="append", default=[], metavar="FORMAT@LAYERS",
                    help="experiment 3: a track with the master's experts requantised into FORMAT at LAYERS "
                         "(comma list or 'all'); repeatable; compared with the master track")
    ap.add_argument("--track-ref", action="append", default=[],
                    help="tracks mode: another track to compare every track against, besides the master (e.g. ne:fp8)")
    ap.add_argument("--master", default=None, help="fetch_master.py's directory of BF16 master experts")
    ap.add_argument("--qcache", default=None, help="tracks mode: qprep.py's cache of the quantiser's formats")
    ap.add_argument("--sub-experts", default="0,51,102,153,204,255",
                    help="tracks mode: the experts a FORMAT/sub track quantises (the quantiser's trellis sample)")
    ap.add_argument("--cov-out", default=None, help="experiment 3b: write per-expert input covariances here "
                    "(with --master; the pass stops after the last --cov-layers layer)")
    ap.add_argument("--cov-layers", type=int, nargs="+", default=[3, 40])
    ap.add_argument("--combine", choices=("f32", "bf16"), default="f32",
                    help="f32: sublayer outputs summed in f32 and rounded once with the residual (the contract, TP=4 "
                         "engine order); bf16: transformers' per-GEMM rounding")
    ap.add_argument("--attn-block", type=int, default=256, help="query rows per attention block")
    ap.add_argument("--head-group", type=int, default=16, help="heads per attention matmul")
    ap.add_argument("--index-block", type=int, default=128, help="query rows per indexer block")
    ap.add_argument("--sample-every", type=int, default=256)
    ap.add_argument("--all-positions-below", type=int, default=1024, help="sequences this short save every position")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)  # raises on any op without a deterministic implementation
    global COMBINE
    COMBINE = args.combine
    log("args", vars(args))
    Run(args).run()


if __name__ == "__main__":
    main()
