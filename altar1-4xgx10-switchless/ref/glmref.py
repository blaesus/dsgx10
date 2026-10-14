"""altar-1 layer-streaming CPU reference: the numerics contract (DESIGN decision 2, docs/model.md) as torch code,
run teacher-forced on altar-1's own checkpoint (compressed-tensors INT4 + BF16), one layer's weights at a time.
A copy of glm53's ref/glmref.py with altar-1's formats; everything not named here is glm53's, line for line.

  python3 glmref.py --ckpt CKPT --out-dir OUT --tokens A.tokens.json [B.tokens.json ...] [--threads 8]
                    [--combine f32|bf16] [--verified SHA256SUMS --wait]

Each tokens file is {"name": ..., "tokens": [...]} (make_tokens.py). All sequences go through each layer
together, so every layer's weights are read once a pass; each sequence attends only to itself, from
position 0. The pass is resumable: after every layer the residual stream is checkpointed under
OUT/state, and a rerun with the same arguments continues from the next layer. With --verified FILE (a
sha256sum listing of the shards whose SHA-256 equals the revision's manifest) no other shard is opened;
with --wait the pass sleeps until the shard it needs is listed, so it can run while the checkpoint arrives.

The contract, as implemented here (each line cites the glm53 model.md section it follows; altar-1's model.md
gives the deltas):
- activations are BF16 between operations; every GEMM accumulates in f32 and rounds its output once
  (§ 2). BF16 weights use torch's bf16 GEMM (f32 accumulation, checked on this CPU by glm53's reference).
- INT4 weights (altar-1 model.md § 4): the routed experts and q_b_proj, kv_b_proj, o_proj of layers 4-76 are
  compressed-tensors pack-quantized (group 32 along K, asymmetric). The weight is the library's own
  dequantisation, W = bf16_rne((q - z) * s): q = nibble - 8 (weight_packed I32 [N, K/8], k = 8j + i at bits
  4i), z = nibble - 8 (weight_zero_point I32 [N/8, K/32], row n = 8w + i at bits 4i), s = weight_scale BF16
  [N, K/32], computed with the library's ops in its order (int8 -> bf16, bf16 subtract, bf16 multiply:
  compressed_tensors 0.18.0 quantization/lifecycle/forward_helpers.py _dequantize, lines 562-567), checked
  equal on every value against compressed_tensors' PackedQuantizationCompressor.decompress (check_int4.py).
  The dequantised weight is then an ordinary BF16 weight. Everything else is BF16 as shipped: the dense
  MLPs of layers 0-2, the 168 experts of layers 3, 77 and the MTP layer 78, all other projections.
- RMSNorm: f32 normalisation, cast to bf16, then a bf16 weight multiply (§ 2.2); eps 1e-5, 1e-6 for the
  two latent norms.
- RoPE interleaved, bf16 arithmetic with bf16 cos/sin, output de-interleaved (§ 2.4.1).
- Attention in the reference's expanded form, eager numerics: bf16 QK^T, scaled by 1/16, masked,
  f32 softmax rounded to bf16, bf16 PV (§ 2.4). Query p attends to all causal keys when p < 2048, else
  to the indexer's top-2048 (§ 2.5).
- Indexer: q, k bf16 (k through a LayerNorm with bias), RoPE on the first 64; f32 scores, ReLU, f32 head
  weights from an f32 GEMM of fp32(x); top-2048 by torch.topk (ties unspecified, § 2.5). Shared layers
  reuse the last full layer's sets.
- Router in f32 over 168 experts, bias for selection only, top-8 (taken sorted), weights renormalised x 2.5
  (§ 2.7; altar-1 model.md § 6).
- Sublayer outputs (--combine f32, the default and the contract, glm53 DESIGN decision 8): o_proj, the dense
  MLP's down projection, each routed expert's down projection (x its f32 weight) and the shared expert's down
  projection stay f32; they are summed in f32 (routed experts in expert-id order, then the shared expert) and
  rounded once, as bf16(f32(h) + sum), when added to the residual. This is the TP=4 engine's prefill order.
  --combine bf16 is transformers' order instead (each GEMM output rounded to bf16; routed experts weighted
  and summed in f32, rounded; shared expert and residual added in bf16). Both are one forward from position
  0 (prefill); the engine's decode sums a strip's K in another order by default (glm53 DESIGN decision 11).
- Logits are kept in f32 (bf16 hidden x bf16 head, f32 accumulation, no output rounding); transformers'
  bf16 logits are bf16() of them up to summation order.
- MTP (§ 3): e = enorm(embed(token t+1)) (zeroed at position 0), g = hnorm(post-final-norm hidden of row t),
  eh_proj(cat(e, g)), decoder layer 78 with its own indexer and BF16 experts at position t,
  shared_head.norm, the shared lm_head: the draft for token t+2.
"""
import argparse
import json
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

BF16, F32 = torch.bfloat16, torch.float32

# Dimensions (glm53 docs/model.md § 1; altar-1 model.md § 2); checked against config.json at start.
D, NH, RQ, RKV, DN, DR, DQK, DV = 6144, 64, 2048, 512, 192, 64, 256, 256
NI, DI, TOPK = 32, 128, 2048
NE, K8, DFF = 168, 8, 2048
NL, MTP = 78, 78
VOCAB = 154880
ROUTE_SCALE = 2.5
ATT_SCALE = DQK ** -0.5
GROUP = 32  # INT4 group size along K
ROW_CHUNK = 4096  # rows a chunk where only memory, not a bit, depends on it

ST_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn,
             "I32": torch.int32, "I64": torch.int64}

# E2M1 codes (the NVFP4 MTP-experts option only): bit 3 the sign, bits 0-2 index the magnitudes.
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=F32)
NIB_LUT = torch.stack([E2M1[torch.arange(256) & 15], E2M1[torch.arange(256) >> 4]], dim=1)  # [256, 2] f32
SHIFTS = torch.arange(0, 32, 4, dtype=torch.int32)  # nibble i of an int32 word at bits 4i


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------------------------- INT4

def unpack_nibbles(words):
    """I32 [..., W] -> int32 [..., W, 8]: nibble i of each word (bits 4i..4i+3), lowest first, 0..15."""
    return (words.unsqueeze(-1) >> SHIFTS) & 15


def int4_dequant(packed, scale, zp):
    """compressed-tensors pack-quantized INT4, group 32, asymmetric -> the BF16 weight [N, K], as the library
    computes it (altar-1 model.md § 4.2): x_q = nibble - 8 (int8), z = nibble - 8 (int8, packed along N),
    W = (x_q.to(bf16) - z.to(bf16)) * scale, in bf16 (one rounding, of the product)."""
    N, K = packed.shape[0], packed.shape[1] * 8
    G = scale.shape[1]
    assert scale.dtype == BF16 and scale.shape[0] == N and K == G * GROUP, (packed.shape, scale.shape)
    assert zp.shape == (N // 8, G), (zp.shape, N, G)
    xq = (unpack_nibbles(packed).reshape(N, G, GROUP) - 8).to(torch.int8)
    z = (unpack_nibbles(zp).permute(0, 2, 1).reshape(N, G) - 8).to(torch.int8)       # row 8w + i <- nibble i
    w = xq.to(BF16) - z.to(BF16).unsqueeze(-1)
    w = w * scale.unsqueeze(-1)
    return w.reshape(N, K)


# ---------------------------------------------------------------------------------------------- checkpoint

class Ckpt:
    """Tensors by name with plain preads (no mmap, so nothing stays in RSS), page cache dropped after.
    verified: a sha256sum listing; only shards named in it are opened (wait: poll for them)."""

    def __init__(self, root, index=None, verified=None, wait=False):
        self.root = root
        with open(index or os.path.join(root, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self.shards, self.lock = {}, threading.Lock()
        self.verified, self.wait = verified, wait

    def _ok(self, fname):
        if self.verified is None:
            return True
        try:
            with open(self.verified) as f:
                return any(line.split()[-1].lstrip("*") == fname for line in f if line.strip())
        except FileNotFoundError:
            return False

    def _shard(self, fname):
        with self.lock:
            if fname not in self.shards:
                said = False
                while not self._ok(fname):
                    if not self.wait:
                        raise RuntimeError(f"{fname} is not in {self.verified} (SHA-256 not verified yet)")
                    if not said:
                        log(f"waiting for {fname} to be verified")
                        said = True
                    time.sleep(60)
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

    def has(self, name):
        return name in self.map

    def is_int4(self, p):
        """p: a module prefix (no '.weight')."""
        return self.has(p + ".weight_packed")

    def int4_raw(self, p):
        return (self.get(p + ".weight_packed"), self.get(p + ".weight_scale"), self.get(p + ".weight_zero_point"),
                self.get(p + ".weight_shape"))

    def weight(self, p):
        """A module's weight as BF16: as stored, or INT4 dequantised by the contract's rule."""
        if self.is_int4(p):
            return int4_from_raw(self.int4_raw(p))
        return self.get(p + ".weight")


def int4_from_raw(raw):
    packed, scale, zp, shape = raw
    w = int4_dequant(packed, scale, zp)
    assert list(w.shape) == shape.tolist(), (w.shape, shape)
    return w


def nvfp4_dequant(q, s, s2):
    """NVFP4 weight as (W' f32 [N, K] = E2M1 · E4M3 block scale, exactly; weight_scale_2)."""
    n, k2 = q.shape
    w = NIB_LUT[q.long()].view(n, k2 * 2 // 16, 16) * s.to(F32).unsqueeze(-1)
    return w.view(n, k2 * 2), s2


# ---------------------------------------------------------------------------------------------- the math

def rms_norm(x, w, eps):
    x32 = x.to(F32)
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return w * y.to(BF16)


def linear(x, w):
    """BF16 weight: bf16 GEMM, f32 accumulation, one rounding."""
    return F.linear(x, w)


# Where the sublayer outputs are rounded (--combine). "f32", the contract: the engine at TP=4 holds only
# partial sums of o_proj, of the dense/shared down projections and of every routed expert's down
# projection, all-reduces them in f32, and rounds once when adding to the residual:
# h = bf16(f32(h) + Σ f32 partials). "bf16": transformers' order (each GEMM output rounded to bf16; routed
# experts summed in f32 and rounded; + shared expert in bf16; + residual in bf16).
COMBINE = "f32"


def out_linear(x, w):
    """A sublayer's last GEMM on a BF16 weight (w may be pre-widened to f32): bf16 output (transformers) or
    f32 (the contract)."""
    if COMBINE == "f32":
        return F.linear(x.to(F32), w.to(F32))
    return linear(x, w)


def out_scaled_linear(x, w32, scale, chunk=4096):
    """The same for a scaled (NVFP4) weight."""
    if COMBINE != "f32":
        return scaled_linear(x, w32, scale, chunk)
    out = torch.empty(x.shape[0], w32.shape[0], dtype=F32)
    for i in range(0, x.shape[0], chunk):
        out[i:i + chunk] = F.linear(x[i:i + chunk].to(F32), w32) * scale
    return out


def add_residual(h, y):
    """bf16 residual + sublayer output: one rounding of the f32 sum (contract), or a bf16 add. Elementwise, in
    row chunks (no full-size f32 temporaries)."""
    out = torch.empty_like(h)
    for r0 in range(0, h.shape[0], ROW_CHUNK):
        hh, yy = h[r0:r0 + ROW_CHUNK], y[r0:r0 + ROW_CHUNK]
        out[r0:r0 + ROW_CHUNK] = (hh.to(F32) + yy).to(BF16) if y.dtype == F32 else hh + yy
    return out


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
    """One layer's non-expert weights, read on demand: `name` is the checkpoint's name of a BF16 tensor; a
    '.weight' that the checkpoint stores INT4 is dequantised (once a layer, cached)."""

    def __init__(self, ck, l):
        self.ck, self.p = ck, f"model.layers.{l}"
        self.cache = {}

    def __call__(self, name):
        full = f"{self.p}.{name}"
        if self.ck.has(full):
            return self.ck.get(full)
        assert name.endswith(".weight") and self.ck.is_int4(full[:-7]), full
        if name not in self.cache:
            self.cache[name] = self.ck.weight(full[:-7])
        return self.cache[name]


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
    """MLA over one sequence (positions 0..S-1). Returns (out [S, D] bf16 or f32, topk, margin)."""
    S = x.shape[0]
    cq = rms_norm(linear(x, W("self_attn.q_a_proj.weight")), W("self_attn.q_a_layernorm.weight"), 1e-6)
    ckv = linear(x, W("self_attn.kv_a_proj_with_mqa.weight"))
    c = rms_norm(ckv[:, :RKV], W("self_attn.kv_a_layernorm.weight"), 1e-6)
    k_rot = rope(ckv[:, RKV:], cos, sin)                                                 # [S, 64]
    # q, K and V are built in row chunks (a GEMM's rows are independent: batch invariance, glm53 reference
    # § 6.5), so the full q_b and kv_b outputs (0.9 and 1.5 GB at 26.8k rows) never exist at once.
    Wqb, Wkvb = W("self_attn.q_b_proj.weight"), W("self_attn.kv_b_proj.weight")
    q = torch.empty(S, NH, DQK, dtype=BF16)
    K = torch.empty(NH, S, DQK, dtype=BF16)
    V = torch.empty(NH, S, DV, dtype=BF16)
    for r0 in range(0, S, ROW_CHUNK):
        r1 = min(S, r0 + ROW_CHUNK)
        qc = linear(cq[r0:r1], Wqb).view(r1 - r0, NH, DQK)
        q[r0:r1] = torch.cat([qc[..., :DN], rope(qc[..., DN:], cos[r0:r1, None], sin[r0:r1, None])], dim=-1)
        kv = linear(c[r0:r1], Wkvb).view(r1 - r0, NH, DN + DV)
        K[:, r0:r1, :DN] = kv[:, :, :DN].transpose(0, 1)
        V[:, r0:r1] = kv[:, :, DN:].transpose(0, 1)
        del qc, kv
    K[:, :, DN:] = k_rot.unsqueeze(0)
    del ckv, c
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


def bf16_mlp(x, W, p, chunk=2048, add_to=None):
    """A BF16 SwiGLU MLP (the dense MLPs of layers 0-2, intermediate 12288; the shared experts), over row
    chunks so the intermediate stays small. Row chunking changes no bit (glm53 reference § 6.5). add_to: an f32
    tensor the output is added to in place (acc + out, elementwise: the same bits as a separate sum)."""
    g, u, d = W(p + ".gate_proj.weight"), W(p + ".up_proj.weight"), W(p + ".down_proj.weight")
    if COMBINE == "f32":
        d = d.to(F32)
    out = add_to if add_to is not None else torch.empty(x.shape, dtype=F32 if COMBINE == "f32" else BF16)
    for i in range(0, x.shape[0], chunk):
        xi = x[i:i + chunk]
        y = out_linear(silu_mul(linear(xi, g), linear(xi, u)), d)
        if add_to is not None:
            out[i:i + chunk] += y
        else:
            out[i:i + chunk] = y
    return out


# ---------------------------------------------------------------------------------------------- expert sources
# A source gives one routed expert's weights in two steps: load(e) only reads (it runs on the I/O threads),
# prepare(raw) converts on the compute thread and returns ("bf16", g, u, d), or ("scaled", (g, sg), (u, su),
# (d, sd)) with exact f32 values and a per-tensor scale (the NVFP4 MTP option).

class CkptExperts:
    """The checkpoint's own experts, by the tensors' names: INT4 (weight_packed) on layers 4-76, BF16 on 3, 77
    and the MTP layer 78."""

    def __init__(self, ck, l):
        self.ck, self.p = ck, f"model.layers.{l}.mlp.experts"

    def load(self, e):
        out = []
        for m in ("gate", "up", "down"):
            p = f"{self.p}.{e}.{m}_proj"
            out.append(("int4", self.ck.int4_raw(p)) if self.ck.is_int4(p) else ("bf16", self.ck.get(p + ".weight")))
        return out

    def prepare(self, raw):
        return ("bf16", *[int4_from_raw(t) if k == "int4" else t for k, t in raw])


def quant_formats():
    """glm53's quantiser formats (glm53-4xgx10-switchless/quant/formats.py, for nvfp4_rtn), imported on use."""
    here = os.path.dirname(os.path.abspath(__file__))
    qdir = os.path.join(os.path.dirname(os.path.dirname(here)), "glm53-4xgx10-switchless", "quant")
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
    raise ValueError(f"format {fmt}: only nvfp4_rtn")


class FormatExperts:
    """A BF16 source's experts (the MTP layer's) requantised on the fly into `fmt` (DESIGN decision 5's lever)."""

    def __init__(self, base, fmt):
        self.base, self.fmt = base, fmt

    def load(self, e):
        return self.base.load(e)

    def prepare(self, raw):
        kind, *w = self.base.prepare(raw)
        assert kind == "bf16", "requantising needs BF16 experts"
        return requantise(self.fmt, *w)


def moe(x, W, l, pool, src=None):
    """Routed experts + shared expert. Returns (out, routing record, calibration sums)."""
    N = x.shape[0]
    logits = F.linear(x.to(F32), W("mlp.gate.weight").to(F32))
    s = logits.sigmoid()
    bias = W("mlp.gate.e_score_correction_bias")
    assert bias.dtype == F32 and bias.shape == (NE,), (bias.dtype, bias.shape)
    sel = s + bias
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
        we = w[r, k]
        acc.index_add_(0, r, y.to(F32) * we[:, None])
        cnt[e] = len(r)
        mass[e] = we.double().sum()
        gu_sq[e] = xe.to(F32).pow(2).sum(0)
        dn_sq[e] = h.to(F32).pow(2).sum(0)
        del wts, g, u, d, xe, h, y
    if COMBINE == "f32":
        out = bf16_mlp(x, W, "mlp.shared_experts", add_to=acc)                            # f32, rounded with the residual
    else:
        out = acc.to(BF16)
        del acc
        out = out + bf16_mlp(x, W, "mlp.shared_experts")
    route = {"route_idx": idx.to(torch.uint8).numpy(), "route_w": w.numpy(), "route_margin": margin.numpy()}
    calib = {"count": cnt.numpy(), "mass": mass.numpy(), "gate_up_in_sqsum": gu_sq.numpy(),
             "down_in_sqsum": dn_sq.numpy()}
    return out, route, calib


# ---------------------------------------------------------------------------------------------- driver

def check_config(cfg):
    assert (cfg["hidden_size"], cfg["num_hidden_layers"], cfg["n_routed_experts"], cfg["index_topk"],
            cfg["q_lora_rank"], cfg["kv_lora_rank"], cfg["vocab_size"], cfg["num_experts_per_tok"],
            cfg["routed_scaling_factor"], cfg["n_group"], cfg["first_k_dense_replace"]) == \
        (D, NL, NE, TOPK, RQ, RKV, VOCAB, K8, ROUTE_SCALE, 1, 3), "config differs from altar-1's"
    q = cfg["quantization_config"]
    assert q["format"] == "pack-quantized", q["format"]
    for g in q["config_groups"].values():
        wq = g["weights"]
        assert (wq["num_bits"], wq["group_size"], wq["symmetric"], wq["strategy"], wq["type"]) == \
            (4, GROUP, False, "group", "int"), wq
        assert g.get("input_activations") is None


class Run:
    def __init__(self, args):
        self.args = args
        self.ck = Ckpt(args.ckpt, args.index, args.verified, args.wait)
        with open(os.path.join(args.ckpt, "config.json")) as f:
            cfg = json.load(f)
        check_config(cfg)
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
            if args.max_tokens:
                ids = ids[:args.max_tokens]
            assert len(ids) <= 65536, "index sets are stored as uint16"
            self.seqs.append({"name": t["name"] + (f"{len(ids)}" if args.max_tokens and len(ids) < len(t["tokens"]) else ""),
                              "off": off, "n": len(ids), "ids": ids})
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
        m = {"ckpt": os.path.basename(os.path.normpath(self.args.ckpt)), "model": "AikidoSec/altar-1",
             "sequences": [{k: s[k] for k in ("name", "off", "n")} for s in self.seqs],
             "rows": self.N, "contract": "glmref.py docstring", "combine": COMBINE,
             "order": ("prefill, one forward from position 0; --combine " + COMBINE +
                       (" (the contract: f32 sublayer sums, one rounding)" if COMBINE == "f32" else
                        " (transformers: per-GEMM bf16 rounding)")),
             "files": FILES}
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
    def layer(self, l, h, prev, seqs, src=None):
        W = LayerW(self.ck, l)
        t0 = time.time()
        x = rms_norm(h, W("input_layernorm.weight"), 1e-5)
        h1 = torch.empty_like(h)
        topk, saved = {}, {}
        for s in seqs:
            r0, n = s["off"], s["n"]
            cos, sin = self.rope[:n], self.rope_s[:n]
            a, tk, mg = attention(x[r0:r0 + n], W, self.full[l] if l != MTP else True,
                                  prev.get(s["name"]), cos, sin, self.args)
            h1[r0:r0 + n] = add_residual(h[r0:r0 + n], a)
            del a
            topk[s["name"]] = tk
            if mg is not None:
                saved[s["name"]] = (tk, mg)
        t_att = time.time() - t0
        h = h1
        del h1, x
        x = rms_norm(h, W("post_attention_layernorm.weight"), 1e-5)
        route = calib = None
        if l < 3:
            m = bf16_mlp(x, W, "mlp")
        else:
            m, route, calib = moe(x, W, l, self.pool, src)
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
        """Logits in f32 from the normed hidden: top-20 logprobs, lse, the next token's logprob, and the full
        logits at full_rows (indices into hn). The head stays BF16 in memory (1.9 GB) and is widened to f32 a
        vocabulary slab at a time (glm53 held it all in f32: 10.9 GiB peak)."""
        os.makedirs(d, exist_ok=True)
        Wh = self.ck.get("lm_head.weight")
        n = hn.shape[0]
        top_i = np.empty((n, 20), np.int32)
        top_lp = np.empty((n, 20), np.float32)
        lse = np.empty(n, np.float32)
        nxt = np.full(n, np.nan, np.float32)
        full = np.empty((len(full_rows), VOCAB), np.float32)
        fr = {int(r): j for j, r in enumerate(full_rows)}
        B, C = 1024, 16384
        for i in range(0, n, B):
            xb = hn[i:i + B].to(F32)
            lg = torch.empty(xb.shape[0], VOCAB, dtype=F32)
            for v0 in range(0, VOCAB, C):
                lg[:, v0:v0 + C] = F.linear(xb, Wh[v0:v0 + C].to(F32))                    # [B, V] f32
            ls = torch.logsumexp(lg, dim=-1)
            tv, ti = lg.topk(20, dim=-1)
            top_i[i:i + B] = ti.numpy()
            top_lp[i:i + B] = (tv - ls[:, None]).numpy()
            lse[i:i + B] = ls.numpy()
            for j in range(lg.shape[0]):
                t = int(rows_next[i + j])
                if t >= 0:
                    nxt[i + j] = float(lg[j, t] - ls[j])
                if i + j in fr:
                    full[fr[i + j]] = lg[j].numpy()
            del lg
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
        prog = self.progress()
        if prog["next"] != 0:  # a resume must keep the run's rounding order
            assert prog.get("combine") == COMBINE, f"this run used --combine {prog.get('combine')}"
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
                h, topk, saved, route, calib, t_att, t_all = self.layer(l, h, prev, self.seqs, None)
                prev = topk
                self.write_layer(l, h, saved, route, calib, {"attention_s": t_att, "layer_s": t_all})
                self.save_state(l + 1, h=h)
                malloc_trim()
                log(f"layer {l:2d} {'full' if self.full[l] else 'shared'} {t_all:7.1f} s (attention {t_att:6.1f} s)"
                    f"  rss {rss_gib():.1f} GiB  hwm {rss_gib('VmHWM'):.1f} GiB")
            if last < NL:
                log(f"stopped after {last} layers (--layers)")
                return
            nxt = "head"
            hn = rms_norm(h, self.ck.get("model.norm.weight"), 1e-5)
            self.save_state("head", hfinal=hn)
            del h
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
            log(f"head {time.time() - t0:.1f} s  hwm {rss_gib('VmHWM'):.1f} GiB")
            self.save_state("mtp" if not a.no_mtp else "done")
            nxt = "mtp" if not a.no_mtp else "done"
        if nxt == "mtp":
            self.run_mtp()
            self.save_state("done")
        log(f"pass done ({time.time() - t_pass:.0f} s this invocation)  hwm {rss_gib('VmHWM'):.1f} GiB")

    def run_mtp_from(self):
        """Replay the MTP layer over a finished run's final hidden states (its state/hfinal.pt), with the MTP
        experts from --mtp-experts (DESIGN decision 5's lever). The run's final/ is linked in for comparisons."""
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
    "layers/LNN/route_idx.npy": "MoE layers: [N, 8] uint8, experts chosen (0..167), by descending biased score",
    "layers/LNN/route_w.npy": "[N, 8] f32, their weights (normalised, x 2.5)",
    "layers/LNN/route_margin.npy": "[N] f32, 8th minus 9th biased score",
    "layers/LNN/calib.npz": "count [168] int64, mass [168] f64 (sum of weights), gate_up_in_sqsum [168, 6144] f32 and down_in_sqsum [168, 2048] f32: per-expert sums of x^2 of the inputs (diagonal of X^T X)",
    "final_hidden_sample.npy": "[P, 6144] f32, after model.norm",
    "final/top20_ids.npy, top20_logprobs.npy": "[N, 20] int32 / f32, from f32 logits",
    "final/lse.npy, next_logprob.npy, argmax.npy": "[N] logsumexp, log p(next token) (NaN at a sequence's last row), argmax",
    "final/logits.npy, logits_rows.npy": "[P, 154880] f32 full logits at the sample rows",
    "layers/L78/*": "the MTP layer, over the MTP rows (row t of a sequence: hidden t, token t+1)",
    "mtp/*": "as final/, for the MTP rows; next_logprob is log p(token t+2); rows_main maps MTP rows to main rows",
    "state/hfinal.pt": "the final hidden after model.norm, every row, bf16 [N, 6144] (gold_prep.py reads it)",
}


def malloc_trim():
    """Hand freed heap back to the system after a layer (glibc keeps it otherwise: RSS 4-5 GiB between layers)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def rss_gib(key="VmRSS"):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1]) / 2**20
    return 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", default=None, help="the index json, if not CKPT/model.safetensors.index.json")
    ap.add_argument("--verified", default=None, help="sha256sum listing of the verified shards: open no other")
    ap.add_argument("--wait", action="store_true", help="with --verified: wait for a shard to be listed")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokens", nargs="+", required=True)
    ap.add_argument("--max-tokens", type=int, default=None, help="cut each sequence to this many tokens (its "
                    "name gets the length appended): the layer checks' 2,400-token input")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=None, help="stop after this many layers (a wiring test)")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--mtp-from", default=None, help="replay only the MTP layer over this finished run")
    ap.add_argument("--mtp-experts", default="bf16", help="with --mtp-from: bf16 (as shipped) or nvfp4_rtn")
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
