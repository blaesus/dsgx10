"""Print model.py's YaRN frequency intermediates as fp32 bit patterns (debug aid for rope_table_build)."""
import argparse, math
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out")
ap.parse_args()
dim, base, orig, factor = 64, 160000.0, 65536, 16
with torch.device("cuda"):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    def corrected_dim(rotations):
        return dim * math.log(orig / (rotations * 2 * math.pi)) / (2 * math.log(base))
    low = max(math.floor(corrected_dim(32)), 0)
    high = min(math.ceil(corrected_dim(1)), dim - 1)
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
    smooth = 1 - ramp
    out = freqs / factor * (1 - smooth) + freqs * smooth
h = lambda t: [f"{v:08x}" for v in t.view(torch.int32).cpu().tolist()]
print("low", low, "high", high)
for name, t in [("freqs", freqs), ("ramp", ramp), ("smooth", smooth), ("out", out)]:
    print(name, " ".join(h(t)))
