#!/usr/bin/env python3
"""ring.cpp's segment arithmetic (split_at, rs_split, direct_segs, relay_seg, transcribed) checked by
enumeration, from the audit in docs/ring.md's send-buffer race section: for every post size to 4 MiB,
every sender and several slow-link shares, each rank gets exactly what it needs of the post, once,
16-byte aligned, and each relay forwards the part its neighbour sent it as A. ~6 s:
  python3 engine/comm/tests/seg_model.py [SLOW_LINK]     (default 0, ring_config.h's kSlowLink)"""
import sys
K, SLOW = 4, 0  # kRanks; kSlowLink (ring_config.h)

def split_at(n): return (n // 2) & ~63
def rs_split(sender, q, pct):
    part = (q * pct // 100) & ~63
    next_slow = sender == SLOW or (sender + 1) % K == SLOW
    return q - part if next_slow else part
def direct_segs(rs, sender, to_next, n, pct):
    if rs:
        q = n // 4; hq = rs_split(sender, q, pct); opp = (sender + 2) % K * q
        if to_next: return (opp + hq, q - hq), ((sender + 1) % K * q, q)
        return (opp, hq), ((sender + K - 1) % K * q, q)
    h = split_at(n)
    return ((h, n - h), (0, h)) if to_next else ((0, h), (h, n - h))
def relay_seg(rs, rank, to_prev, n, pct):
    if to_prev: return direct_segs(rs, (rank + 1) % K, False, n, pct)[0]
    return direct_segs(rs, (rank + K - 1) % K, True, n, pct)[0]

def cover(segs, lo, hi):  # the segments tile [lo, hi) exactly (empty ones skipped), 16-byte aligned
    segs = sorted(s for s in segs if s[1])
    x = lo
    for off, ln in segs:
        if off != x or off % 16 or ln % 16 or ln < 0: return False
        x = off + ln
    return x == hi

def main():
    bad = 0; cases = 0
    slow = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    global SLOW; SLOW = slow
    cap = 4 << 20
    for rs in (False, True):
        step = 64 if rs else 16
        pcts = (50, 0, 25, 37, 75, 100) if rs else (50,)
        for pct in pcts:
            for n in range(step, cap + 1, step):
                q = n // 4
                for s in range(K):
                    cases += 1
                    nx, pv, op = (s + 1) % K, (s + K - 1) % K, (s + 2) % K
                    an, bn = direct_segs(rs, s, True, n, pct)   # to next
                    ap, bp = direct_segs(rs, s, False, n, pct)  # to prev
                    r_nx = relay_seg(rs, nx, False, n, pct)  # next relays s's A on to its next (= op)
                    r_pv = relay_seg(rs, pv, True, n, pct)   # prev relays s's A on to its prev (= op)
                    ok = r_nx == an and r_pv == ap
                    if rs:  # each rank gets its own quarter of s's contribution
                        ok &= cover([bn], nx * q, nx * q + q) and cover([bp], pv * q, pv * q + q)
                        ok &= cover([r_nx, r_pv], op * q, op * q + q)
                        ok &= cover([an], an[0], an[0] + an[1]) and an[0] >= op * q and an[0] + an[1] <= op * q + q
                    else:  # each rank gets all of it
                        ok &= cover([an, bn], 0, n) and cover([ap, bp], 0, n) and cover([r_nx, r_pv], 0, n)
                    if not ok:
                        bad += 1
                        if bad <= 5: print('BAD', 'rs' if rs else 'ae', pct, n, s, an, bn, ap, bp, r_nx, r_pv)
    print('slow link %d: %d cases (all-exchange n = 16 B .. 4 MiB step 16; reduce-scatter n = 64 B .. 4 MiB step 64, '
          'slow_pct 0/25/37/50/75/100; 4 senders): %d bad' % (slow, cases, bad))
    return bad

if __name__ == '__main__':
    sys.exit(1 if main() else 0)
