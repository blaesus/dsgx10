"""How torch.topk breaks ties on CUDA bf16 (what the reference's indexer relies on).
python3 check_topk.py --out /out/x   (the --out is unused; gen.sh passes it)"""
import argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out")
ap.parse_args()
torch.manual_seed(0)
ok_low = ok_any = 0
trials = 200
for t in range(trials):
    n = [300, 5000, 70000, 140000][t % 4]
    k = [64, 512, 512, 2048][t % 4]
    # few distinct values -> many ties at the threshold
    v = torch.randint(0, 40, (2, n), device="cuda").to(torch.bfloat16) / 8
    v[:, n // 3] = float("-inf")
    idx = v.topk(k, dim=-1, sorted=False).indices.sort(-1).values
    kth = v.topk(k, dim=-1).values[:, -1:]
    above = v > kth
    need = k - above.sum(-1, keepdim=True)
    eq = (v == kth)
    rank = eq.cumsum(-1)
    lowest = above | (eq & (rank <= need))
    exp_idx = lowest.nonzero()[:, 1].view(2, -1)
    ok_low += int(torch.equal(exp_idx, idx))
    ok_any += int(torch.equal(v.gather(-1, idx).sort(-1).values, v.gather(-1, exp_idx).sort(-1).values))
print(f"ties broken by lowest index in {ok_low}/{trials}; same value multiset {ok_any}/{trials}")
