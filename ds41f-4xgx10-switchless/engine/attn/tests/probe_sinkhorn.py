"""Which fp32 operation orders torch uses inside torchref's hc_split_sinkhorn (debug aid for mhc.cu):
the 4-element sums over dim -1 and dim -2 of a [1, T, 4, 4] tensor, and softmax(-1) over 4."""
import argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out")
ap.parse_args()
torch.manual_seed(0)
c = torch.rand(1, 4096, 4, 4, device="cuda") * 0.6 + 0.01


def same(a, b):
    return f"{(a == b).float().mean().item() * 100:.1f}%"


for dim in (-1, -2):
    s = c.sum(dim)
    t = [c.select(dim, i) for i in range(4)]
    cand = {
        "seq ((0+1)+2)+3": ((t[0] + t[1]) + t[2]) + t[3],
        "pair (0+1)+(2+3)": (t[0] + t[1]) + (t[2] + t[3]),
        "butterfly (0+2)+(1+3)": (t[0] + t[2]) + (t[1] + t[3]),
        "rev ((3+2)+1)+0": ((t[3] + t[2]) + t[1]) + t[0],
    }
    print(f"sum({dim}):", ", ".join(f"{k} {same(s, v)}" for k, v in cand.items()))
x = torch.randn(1, 4096, 4, 4, device="cuda") * 3
sm = x.softmax(-1)
mx = x.amax(-1, keepdim=True)
e = torch.exp(x - mx)
t = [e[..., i] for i in range(4)]
for name, s in {"seq": ((t[0] + t[1]) + t[2]) + t[3], "pair": (t[0] + t[1]) + (t[2] + t[3]),
                "butterfly": (t[0] + t[2]) + (t[1] + t[3])}.items():
    s = s.unsqueeze(-1)
    print(f"softmax sum {name}: e/s {same(sm, e / s)}, e*(1/s) {same(sm, e * (1 / s))}")
d = c / (c.sum(-1, keepdim=True) + 1e-6)
s = c.sum(-1, keepdim=True)
print("div by (sum+eps):", same(d, c / (s + 1e-6)), " mul by recip:", same(d, c * (1 / (s + 1e-6))))
