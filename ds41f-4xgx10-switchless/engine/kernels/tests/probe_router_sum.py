"""Which summation order does torch use for the router's 6-weight sum? Prints how many of the
reference weights each candidate order reproduces bit for bit (run like gen_ref.py)."""
import numpy as np
import torch
import torch.nn.functional as F

D = "/out"
lg = torch.from_numpy(np.load(f"{D}/gate_logits.npy")).cuda()
b = torch.from_numpy(np.load(f"{D}/gate_b.npy")).cuda()
ref = torch.from_numpy(np.load(f"{D}/gate_wts.npy")).cuda()
s = F.softplus(lg).sqrt()
idx = (s + b).topk(6, dim=-1)[1]
w = s.gather(1, idx)
tsum = w.sum(-1, keepdim=True)
print("torch pipeline reproduces reference:", torch.equal(w / (tsum + 1e-20) * 1.5, ref))
f = lambda x: x.float()
W = [w[:, i:i + 1] for i in range(6)]
cands = {
    "sequential": ((((W[0] + W[1]) + W[2]) + W[3]) + W[4]) + W[5],
    "(04+15)+(23)": ((W[0] + W[4]) + (W[1] + W[5])) + (W[2] + W[3]),
    "((04+15)+2)+3": (((W[0] + W[4]) + (W[1] + W[5])) + W[2]) + W[3],
    "(02+4)+(13+5)": ((W[0] + W[2]) + W[4]) + ((W[1] + W[3]) + W[5]),
    "(01+23)+(45)": ((W[0] + W[1]) + (W[2] + W[3])) + (W[4] + W[5]),
    "(0+4+2)+(1+5+3)": ((W[0] + W[4]) + W[2]) + ((W[1] + W[5]) + W[3]),
}
for k, v in cands.items():
    out = w / (v + 1e-20) * 1.5
    print(f"{k:18s} sum==torch {int((v == tsum).sum())}/8  weights==ref {int((out == ref).sum())}/48")
