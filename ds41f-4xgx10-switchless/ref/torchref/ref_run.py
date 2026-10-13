"""Layer-streaming reference forward of DeepSeek-V4.1-Flash on one GB10, around the publisher's
model.py, unchanged (our pure-torch kernel.py shadows the tilelang one). Teacher-forced prefill of
one token sequence, start_pos 0, world_size 1. Writes golden tensors as .npy plus manifest.json.

  python3 ref_run.py --ckpt /models --tokens x.tokens.json --out-dir /out/x \
      [--positions all | sparse:EVERY:LAST] [--record auto | none | p,q,...] [--full-logits all | N]

Hooks, not edits: the Engram table embedding reads rows straight from the checkpoint (building the
publisher's would allocate 98 GB), the indexer's score einsum runs per query chunk (one [s, 32, t]
tensor is 25 GB at 20k tokens), and forward hooks capture sublayer tensors at the saved positions.
"""
import argparse
import gc
import json
import mmap
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
INF = os.path.join(HERE, "..", "inference")
sys.path.insert(0, HERE)  # our kernel.py first
sys.path.insert(1, INF)   # then the publisher's model.py, engram.py, ...
import kernel as K  # noqa: E402
import model as M  # noqa: E402
from engram import EngramLayout, NgramHashState  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
INDEX_CHUNK = 512  # query rows per indexer score einsum
READERS = ThreadPoolExecutor(8)


ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I8": torch.int8,
             "U8": torch.uint8, "I32": torch.int32, "I64": torch.int64, "BOOL": torch.bool,
             "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu}


class CkptIndex:
    """Tensors by name with plain preads: torch's copy from safetensors' mmap faults pages in 4 KiB
    at a time (~140 MB/s); a pread of the whole tensor runs at disk speed."""

    def __init__(self, root):
        self.root = root
        self.map = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.shards = {}
        self.lock = threading.Lock()

    def _shard(self, f):
        with self.lock:
            if f not in self.shards:
                fd = os.open(os.path.join(self.root, f), os.O_RDONLY)
                n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
                self.shards[f] = (fd, 8 + n, json.loads(os.pread(fd, n, 8)))
            return self.shards[f]

    def get(self, name, device="cuda"):
        return self.read_cpu(name).to(device)

    def read_cpu(self, name):
        fd, base, hdr = self._shard(self.map[name])
        e = hdr[name]
        a, b = e["data_offsets"]
        buf = bytearray(b - a)
        mv, got = memoryview(buf), 0
        while got < len(buf):
            k = os.preadv(fd, [mv[got:]], base + a + got)
            if k <= 0:
                raise IOError(f"short read of {name}")
            got += k
        # drop the pages again: on GB10 the page cache competes with CUDA allocations (a second run
        # started while the cache was full failed with cudaErrorMemoryAllocation)
        os.posix_fadvise(fd, base + a, b - a, os.POSIX_FADV_DONTNEED)
        with torch.device("cpu"):
            t = torch.frombuffer(buf, dtype=torch.uint8) if buf else torch.empty(0, dtype=torch.uint8)
            return t.view(ST_DTYPES[e["dtype"]]).reshape(e["shape"])

    def raw_span(self, name):
        """(path, byte offset, shape, dtype) of a tensor, for memmap access."""
        _, base, hdr = self._shard(self.map[name])
        e = hdr[name]
        return os.path.join(self.root, self.map[name]), base + e["data_offsets"][0], e["shape"], e["dtype"]


class NoEngramTable(torch.nn.Module):
    """Stands in for ParallelEngramEmbedding while a Block is built, so nothing is allocated."""

    def __init__(self, num_embeddings, dim):
        super().__init__()


class MMapEngramEmbed(torch.nn.Module):
    """ParallelEngramEmbedding at world_size 1, reading rows straight from the checkpoint mmap."""

    def __init__(self, ck, layer_id):
        super().__init__()
        pw, ow, sw, _ = ck.raw_span(f"layers.{layer_id}.engram.embed.weight")
        ps, os_, ss, _ = ck.raw_span(f"layers.{layer_id}.engram.embed.scale")
        self.w = np.memmap(pw, dtype=np.uint8, mode="r", offset=ow, shape=tuple(sw))
        self.s = np.memmap(ps, dtype=np.uint8, mode="r", offset=os_, shape=tuple(ss))
        self.paths = {pw, ps}
        for m in (self.w, self.s):  # rows are scattered: readahead only fills the page cache
            m._mmap.madvise(mmap.MADV_RANDOM)

    def forward(self, indices):
        idx = indices.cpu().numpy().reshape(-1)
        w = torch.from_numpy(np.ascontiguousarray(self.w[idx])).cuda().view(torch.float8_e4m3fn)
        s = torch.from_numpy(np.ascontiguousarray(self.s[idx])).cuda().view(torch.float8_e8m0fnu)
        for path in self.paths:  # drop the touched pages too (GB10: the cache competes with CUDA)
            fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        v = w.float().unflatten(-1, (-1, 32)) * s.float().unsqueeze(-1)
        return v.flatten(-2).to(torch.bfloat16).view(*indices.shape, -1)


def load_module(mod, ck, prefix):
    params = list(mod.named_parameters())
    for c0 in range(0, len(params), 64):  # read 64 tensors at a time on 8 threads: disk queue depth
        chunk = params[c0 : c0 + 64]
        for (name, p), t in zip(chunk, READERS.map(lambda np_: ck.read_cpu(prefix + np_[0]), chunk)):
            load_param(p, name, prefix + name, t.to("cuda"), ck)
    # Linear keeps `weight.scale` pointing at the scale Parameter; re-link after the .data swaps
    for m in mod.modules():
        if isinstance(m, M.Linear) and m.scale is not None:
            m.weight.scale = m.scale


def load_param(p, name, full, t, ck):
    if name.endswith("wo_a.weight"):
        sc = ck.get(full.replace("weight", "scale"))
        t = K.dequant_fp8_weight(t, sc, 32).to(torch.bfloat16)  # what convert.py does
    if p.dtype == torch.float4_e2m1fn_x2:
        p.data = t.view(torch.float4_e2m1fn_x2)
    elif p.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        p.data = t.view(p.dtype) if t.dtype != p.dtype else t
    else:
        assert tuple(t.shape) == tuple(p.shape), (full, t.shape, p.shape)
        p.data.copy_(t.to(p.dtype))


# ---------------------------------------------------------------------------------------------
# The publisher's Indexer.forward with the score einsum run per query chunk (same ops, same
# dtypes), recording the selected positions and layer 20's pool at a few query positions.

REC = {"layer": None, "positions": None, "topk": {}, "pool": {}}


def chunked_indexer_forward(self, x, qr, latent, start_pos, offset):
    assert self.freqs_cis is not None
    bsz, seqlen, _ = x.size()
    ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen

    if self.owns_k and latent is not None:
        freqs = (
            self.freqs_cis[: seqlen - seqlen % ratio : ratio]
            if start_pos == 0
            else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        )
        k = self.k_norm(self.wk(latent))
        M.apply_rotary_emb(k[..., -rd:], freqs)
        M.fp4_act_quant(k, M.fp4_block_size, True)
        self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
        M.shared_attn.index_k = self.k_cache

    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
    M.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
    M.fp4_act_quant(q, M.fp4_block_size, True)

    index_k = M.shared_attn.index_k[:bsz, : end_pos // ratio]
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
    index_score = torch.empty(bsz, seqlen, index_k.size(1), dtype=q.dtype, device=q.device)
    for s0 in range(0, seqlen, INDEX_CHUNK):
        s1 = min(seqlen, s0 + INDEX_CHUNK)
        part = torch.einsum("bshd,btd->bsht", q[:, s0:s1], index_k)
        index_score[:, s0:s1] = (part.relu_() * weights[:, s0:s1].unsqueeze(-1)).sum(dim=2)
        del part
    assert M.world_size == 1

    if start_pos == 0:
        compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
        index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
    else:
        compress_lens = end_pos // ratio

    if self.is_candidate_source:
        M.shared_attn.candidates = M.select_candidate_blocks(
            index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
        )
        if REC["positions"] is not None:
            cand = M.shared_attn.candidates[0, REC["positions"]]
            REC["pool"][REC["layer"]] = cand[:, :: self.candidate_block_size].cpu()
    elif self.uses_candidates:
        index_score = index_score.masked_fill(~M.shared_attn.candidates, -torch.inf)

    topk = min(self.index_topk, end_pos // ratio)
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    idxs = torch.where(idxs < compress_lens, idxs + offset, -1).int()
    if REC["positions"] is not None:
        sel = idxs[0, REC["positions"]].long()
        REC["topk"][REC["layer"]] = torch.where(sel >= 0, sel - offset, -1).int().cpu()
    return idxs


M.Indexer.forward = chunked_indexer_forward


# ---------------------------------------------------------------------------------------------

class Writer:
    """Writes each golden tensor as <name>.npy as soon as it exists; manifest.json at the end."""

    def __init__(self, out_dir):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.entries = []

    def put(self, name, t, meaning, positions):
        a = to_numpy(t)
        np.save(os.path.join(self.dir, name + ".npy"), a)
        self.entries.append({"name": name, "file": name + ".npy", "shape": list(a.shape),
                             "dtype": str(a.dtype), "positions": positions, "meaning": meaning})

    def finish(self, meta):
        meta = dict(meta, tensors=self.entries)
        json.dump(meta, open(os.path.join(self.dir, "manifest.json"), "w"), indent=1)


def to_numpy(t):
    if isinstance(t, np.ndarray):
        return t
    t = t.detach()
    if t.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return t.float().cpu().numpy()
    if t.dtype in (torch.bool, torch.int64):
        return t.cpu().numpy()
    return t.to(torch.int32).cpu().numpy()


def instrument(blk, sink, pos):
    """Capture the sublayer tensors of one Block at the saved positions into `sink`."""
    def sel(t):
        return t[0, pos].float().cpu()

    hooks = [
        blk.attn.register_forward_pre_hook(lambda m, a: sink.__setitem__("attn_in", sel(a[0]))),
        blk.attn.register_forward_hook(lambda m, a, o: sink.__setitem__("attn_out", sel(o))),
        blk.ffn.register_forward_pre_hook(lambda m, a: sink.__setitem__("ffn_in", sel(a[0]))),
        blk.ffn.register_forward_hook(lambda m, a, o: sink.__setitem__("ffn_out", sel(o))),
    ]

    def gate_hook(m, a, o):  # the gate sees x flattened to [tokens, dim]
        sink["route_w"], sink["route_idx"] = o[0][pos].float().cpu(), o[1][pos].int().cpu()

    hooks.append(blk.ffn.gate.register_forward_hook(gate_hook))
    orig_post, orig_mixes, n = blk.hc_post, blk.hc_mixes, {"post": 0, "mixes": 0}

    def hc_post(x, residual, post, comb):
        y = orig_post(x, residual, post, comb)
        if n["post"] == 0:
            sink["mid"] = sel(y)
        n["post"] += 1
        return y

    def hc_mixes(x, fn, scale, base):
        pre, post, comb = orig_mixes(x, fn, scale, base)
        tag = "attn" if n["mixes"] == 0 else "ffn"
        n["mixes"] += 1
        sink[f"hc_{tag}_pre"], sink[f"hc_{tag}_post"], sink[f"hc_{tag}_comb"] = sel(pre), sel(post), sel(comb)
        return pre, post, comb

    blk.hc_post, blk.hc_mixes = hc_post, hc_mixes
    return hooks


SUBLAYER_MEANING = {
    "mid": "residual after the attention sublayer (hc_post output), [P, hc, dim]",
    "attn_in": "attention input: hc_pre(x, pre_mix) then attn_norm, [P, dim]",
    "attn_out": "attention output after wo_b, before hc_post, [P, dim]",
    "ffn_in": "MoE input: hc_pre(mid, hc_attn_pre) then ffn_norm, [P, dim]",
    "ffn_out": "MoE output incl. the shared expert, before hc_post, [P, dim]",
    "route_w": "routing weights of the 6 chosen experts (normalised, x route_scale), [P, 6]",
    "route_idx": "chosen expert ids, in topk order, [P, 6]",
    "hc_attn_pre": "attention sublayer hc_mixes pre: becomes the FFN's input mix, [P, hc]",
    "hc_attn_post": "attention sublayer hc_mixes post, [P, hc]",
    "hc_attn_comb": "attention sublayer hc_mixes comb (after Sinkhorn), [P, hc, hc]",
    "hc_ffn_pre": "FFN sublayer hc_mixes pre: becomes the next layer's pre_mix, [P, hc]",
    "hc_ffn_post": "FFN sublayer hc_mixes post, [P, hc]",
    "hc_ffn_comb": "FFN sublayer hc_mixes comb (after Sinkhorn), [P, hc, hc]",
}


def parse_positions(spec, L):
    if spec == "all":
        return list(range(L))
    if spec.startswith("ranges:"):  # ranges:A-B,C-D (B, D exclusive)
        ps = set()
        for r in spec[len("ranges:"):].split(","):
            a, b = (int(v) for v in r.split("-"))
            ps |= set(range(max(0, a), min(L, b)))
        return sorted(ps)
    _, every, last = spec.split(":")
    return sorted(set(range(0, L, int(every))) | set(range(max(0, L - int(last)), L)))


def parse_record(spec, L):
    if spec == "none":
        return []
    if spec == "auto":
        cand = [7, 63, 127, 511, 1023, 2047, 4095, 8191, 12287, 16383, 16384, 16391, 16447, 17407,
                19455, L // 2, L - 257, L - 129, L - 2, L - 1]
    else:
        cand = [int(x) for x in spec.split(",")]
    return sorted({p for p in cand if 0 <= p < L})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/models")
    ap.add_argument("--tokens", required=True, help="json: {'tokens': [...], 'n_prompt': n}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--positions", default="all", help="all | sparse:EVERY:LAST | ranges:A-B,...")
    ap.add_argument("--record", default="auto", help="query positions for indexer/pool records")
    ap.add_argument("--full-logits", default="8", help="all | N: full logits for the last N positions")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--ablate", default="", help="experiments only: 'engram' skips the Engram layers")
    ap.add_argument("--engram-off", default="", metavar="A-B,...",
                    help="experiments only: shut the Engram gate at these positions (token_mask False)")
    ap.add_argument("--engram-off-layers", default="", metavar="I,J",
                    help="experiments only: apply --engram-off at these Engram layers only (default all)")
    ap.add_argument("--alt-ckpt", default="", help="experiments only: a second checkpoint (e.g. /models-alt)")
    ap.add_argument("--alt-layers", default="", metavar="A-B,...",
                    help="experiments only: load these blocks (B exclusive) from --alt-ckpt instead")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="override a ModelArgs field, for experiments only (e.g. candidate_topk_blocks=4096)")
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    cfg = json.load(open(os.path.join(INF, "config.json")))
    tj = json.load(open(a.tokens))
    toks, n_prompt = tj["tokens"], tj.get("n_prompt", 0)
    L = len(toks)
    args = M.ModelArgs(**cfg)
    args.max_batch_size = 1
    args.max_seq_len = ((L + 255) // 256) * 256 + 256
    for kv in a.set:
        k, v = kv.split("=", 1)
        assert hasattr(args, k), k
        setattr(args, k, json.loads(v))
    M.world_size, M.rank, M.default_dtype = 1, 0, torch.float8_e4m3fn
    M.ParallelEngramEmbedding = NoEngramTable

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    ck = CkptIndex(a.ckpt)
    ck_alt = CkptIndex(a.alt_ckpt) if a.alt_ckpt else None
    alt_layers = set(parse_positions("ranges:" + a.alt_layers, args.n_layers)) if a.alt_layers else set()
    layout = EngramLayout.from_args(args)
    W = Writer(a.out_dir)
    plist = parse_positions(a.positions, L)
    rlist = parse_record(a.record, L)
    pos = torch.tensor(plist, dtype=torch.long)
    pspec = "all" if a.positions == "all" else "positions.npy"
    W.put("tokens", np.array(toks, dtype=np.int32), "token ids of the whole sequence, [L]", "all")
    W.put("positions", np.array(plist, dtype=np.int32), "the saved positions P", "-")
    W.put("record_positions", np.array(rlist, dtype=np.int32), "query positions R of the indexer records", "-")
    REC["positions"] = torch.tensor(rlist, dtype=torch.long) if rlist else None

    t0 = time.time()
    hasher = NgramHashState(args, layout, tok)
    ids = torch.tensor([toks], dtype=torch.long)
    hashes = hasher(ids, 0, None)  # [1, L, n_engram_layers, 24]
    embed_w = ck.get("embed.weight")
    h = torch.nn.functional.embedding(ids, embed_w)
    del embed_w
    W.put("hashes", hashes[0].long(), "Engram hash row ids per position, [L, n_engram_layers, 24]", "all")
    W.put("embed", h[0, pos], "token embedding (bf16), [P, dim]", pspec)
    print(f"L={L} n_prompt={n_prompt} saved={len(plist)} records={len(rlist)}; "
          f"hash+embed {time.time()-t0:.1f}s", flush=True)

    engram_mask = None
    if a.engram_off:
        engram_mask = torch.ones(1, L, dtype=torch.bool)
        engram_mask[0, parse_positions("ranges:" + a.engram_off, L)] = False
    h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, args.hc_mult)
    main_hiddens = []
    for i in range(args.n_layers):
        t1 = time.time()
        torch.cuda.reset_peak_memory_stats()
        with torch.device("cuda"):
            blk = M.Block(i, args, layout)
        load_module(blk, ck_alt if i in alt_layers else ck, f"layers.{i}.")
        t_load = time.time() - t1
        if blk.engram is not None and a.ablate != "engram":
            blk.engram.embed = MMapEngramEmbed(ck, i)
            off_here = not a.engram_off_layers or str(i) in a.engram_off_layers.split(",")
            h = blk.engram(h, hashes[:, :, blk.engram.layer_hash_index, :], engram_mask if off_here else None)
            W.put(f"L{i:02d}.engram_out", h[0, pos], "residual after Engram, before the block, [P, hc, dim]", pspec)
        if i in args.dspark_target_layer_ids:
            main_hiddens.append(h.mean(dim=2)[0, pos])  # what Transformer.forward appends, at P
        sink = {}
        hooks = instrument(blk, sink, pos)
        REC["layer"] = i
        h, pre_mix = blk(h, 0, pre_mix, None)
        for hk in hooks:
            hk.remove()
        W.put(f"L{i:02d}.out", h[0, pos], "residual after the block, [P, hc, dim]", pspec)
        W.put(f"L{i:02d}.pre_mix", pre_mix[0, pos], "pre_mix the block returns (= hc_ffn_pre), [P, hc]", pspec)
        for k, v in sink.items():
            W.put(f"L{i:02d}.{k}", v, SUBLAYER_MEANING[k], pspec)
        if i in REC["topk"]:
            W.put(f"L{i:02d}.index_topk", REC["topk"].pop(i),
                  "indexer's selected compressed positions (ascending, -1 = none) per record position, "
                  "[R, k]; compressed position j covers tokens j*ratio .. j*ratio+ratio-1",
                  "record_positions.npy")
        if i in REC["pool"]:
            W.put(f"L{i:02d}.pool_blocks", REC["pool"].pop(i),
                  "candidate pool: block b (compressed positions 8b..8b+7) kept, per record position, "
                  "[R, n_blocks]", "record_positions.npy")
        blk.__dict__.pop("hc_post", None)
        blk.__dict__.pop("hc_mixes", None)
        del blk, sink
        gc.collect()
        torch.cuda.empty_cache()
        print(f"layer {i:2d} {time.time()-t1:6.1f}s (load {t_load:5.1f}s) "
              f"peak {torch.cuda.max_memory_allocated()/2**30:5.1f} GiB |h|={h.float().abs().mean().item():.4f}",
              flush=True)

    x = M.Block.hc_pre(None, h, pre_mix)  # hc_pre uses no block state
    norm = M.RMSNorm(args.dim, args.norm_eps)
    norm.weight.data.copy_(ck.get("norm.weight"))
    xn = norm(x)
    W.put("final_pre_norm", x[0, pos], "hc_pre(h, last pre_mix), before the final norm, [P, dim]", pspec)
    W.put("final_hidden", xn[0, pos], "final norm output = head input, [P, dim]", pspec)
    W.put("main_hidden", torch.cat(main_hiddens, -1),
          "DSpark input: mean over hc copies of the inputs of layers 37, 38, 39 (after Engram), "
          "concatenated, [P, 3*dim]", pspec)
    head_w = ck.get("head.weight").float()
    n_full = L if a.full_logits == "all" else min(L, int(a.full_logits))
    topv, topi, full, lse, nxt = [], [], [], [], []
    ids_next = torch.tensor(toks[1:] + [0], dtype=torch.long)
    for c0 in range(0, L, 2048):
        c1 = min(L, c0 + 2048)
        lg = torch.nn.functional.linear(xn[0, c0:c1].float(), head_w)
        v, ix = lg.topk(a.topk, dim=-1)
        topv.append(v.cpu())
        topi.append(ix.cpu())
        lse.append(torch.logsumexp(lg, -1).cpu())
        nxt.append(lg.gather(1, ids_next[c0:c1, None]).squeeze(1).cpu() - lse[-1])
        if c1 > L - n_full:
            full.append(lg[max(0, L - n_full - c0):].cpu())
        del lg
    topi = torch.cat(topi)
    W.put("logits_topv", torch.cat(topv), f"top-{a.topk} logits (fp32) per position, descending, [L, {a.topk}]", "all")
    W.put("logits_topi", topi, f"token ids of logits_topv, [L, {a.topk}]", "all")
    W.put("logits_lse", torch.cat(lse), "logsumexp of the full fp32 logits per position, [L]", "all")
    next_lp = torch.cat(nxt)[: L - 1]
    W.put("next_logprob", next_lp, "log-probability of the actual next token, [L-1] (position p predicts p+1)",
          "all but the last")
    W.put("logits_full", torch.cat(full), f"full fp32 logits of the last {n_full} positions, [{n_full}, vocab]",
          f"last {n_full}")

    am = topi[:, 0].tolist()
    agree_all = sum(1 for p in range(L - 1) if am[p] == toks[p + 1])
    comp = range(max(0, n_prompt - 1), L - 1)
    agree_comp = sum(1 for p in comp if am[p] == toks[p + 1])
    misses = [p for p in comp if am[p] != toks[p + 1]]
    W.finish({"tokens_file": os.path.basename(a.tokens), "L": L, "n_prompt": n_prompt,
              "positions": a.positions, "record": a.record, "overrides": a.set, "ablate": a.ablate, "engram_off": a.engram_off,
              "engram_off_layers": a.engram_off_layers, "alt_ckpt": a.alt_ckpt, "alt_layers": a.alt_layers,
              "seconds": round(time.time() - t0),
              "argmax_next_all": [agree_all, L - 1], "argmax_next_completion": [agree_comp, len(comp)],
              "completion_misses": misses,
              "mean_next_logprob_all": round(next_lp.mean().item(), 4),
              "mean_next_logprob_completion": round(next_lp[max(0, n_prompt - 1):].mean().item(), 4)
              if L - 1 > max(0, n_prompt - 1) else None,
              "notes": "fp32 copies of bf16 tensors; P = saved positions (all, or positions.npy); "
                       "R = record_positions.npy; LNN is the block index"})
    print(f"done {time.time()-t0:.0f}s; argmax==next: all {agree_all}/{L-1}, "
          f"completion {agree_comp}/{len(comp)}; misses at {misses}", flush=True)
    print("greedy next after last:", repr(tok.decode([am[-1]])), flush=True)


if __name__ == "__main__":
    main()
