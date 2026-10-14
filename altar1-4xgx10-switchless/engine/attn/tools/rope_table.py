"""The RoPE table the engine loads: [max_pos][32] u32, bf16 cos (low half) and bf16 sin (high half), computed
by torch exactly as GlmMoeDsaRotaryEmbedding.forward does (fp32 inv_freq and angle, fp32 cos/sin, cast to bf16).
A double-precision build differs from torch's in 29 of the 2^20 x 64 values, so the table is torch's.
usage: python3 -I rope_table.py OUT.npy [max_pos]   (default 1,048,576 positions: 128 MiB)"""
import sys

import numpy as np
import torch

THETA, ROT = 8000000.0, 64


def table(max_pos):
    inv = 1.0 / (THETA ** (torch.arange(0, ROT, 2, dtype=torch.float) / ROT))
    out = np.empty((max_pos, ROT // 2), np.uint32)
    for p0 in range(0, max_pos, 1 << 18):
        pos = torch.arange(p0, min(max_pos, p0 + (1 << 18))).float()
        fr = (inv[None, :, None] @ pos[None, None, :]).transpose(1, 2)[0]  # as rotary.forward
        c = fr.cos().to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16).astype(np.uint32)
        s = fr.sin().to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16).astype(np.uint32)
        out[p0:p0 + len(pos)] = c | (s << 16)
    return out


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 20
    np.save(sys.argv[1], table(n))
