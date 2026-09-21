"""定稿前最后一次检验：底部阈值定 0，并对比两种顶部口径。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, store

H, TOL = store.H, store.TOL
COOL = store.COOL_TRADING


def scan(s, c, kind, thr):
    dates = list(s.index)
    s = s.to_numpy(float); cl = c.to_numpy(float)
    out = []; last = -10 ** 9; n = len(s)
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if kind == "bot":
            hit = s[i - 1] <= thr and s[i] > s[i - 1]
        else:   # top A：向上突破 100
            hit = s[i - 1] <= thr and s[i] > thr
        if not hit: continue
        last = i
        c0 = cl[i]; seg = cl[i + 1:i + H + 1]
        out.append((dates[i], seg[-1] / c0 - 1, seg.min() / c0 - 1, seg.max() / c0 - 1, s[i]))
    return out


def scan_top_b(s, c, thr=100.0):
    """top B：从 100 上方回落（过热衰竭）"""
    dates = list(s.index)
    s = s.to_numpy(float); cl = c.to_numpy(float)
    out = []; last = -10 ** 9; n = len(s)
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if s[i - 1] >= thr and s[i] < thr:
            last = i
            c0 = cl[i]; seg = cl[i + 1:i + H + 1]
            out.append((dates[i], seg[-1] / c0 - 1, seg.min() / c0 - 1, seg.max() / c0 - 1, s[i]))
    return out


def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


P = store.load()

print("=" * 120)
print("底部：阈值 0（= 需求原话「低于 0 是底部」）")
print("=" * 120)
rows = []
for b in C.BOARD_ORDER:
    ev = scan(P[b]["score"], P[b]["close"], "bot", 0.0)
    rows += [(b,) + e for e in ev]
    o = sum(1 for x in ev if x[1] > 0 and x[2] >= -TOL)
    print(f"  {C.BOARDS[b]['name']:<8} n={len(ev):>2}  命中 {o}/{len(ev)}")
    for d, r, m, mx, s0 in ev:
        print(f"        {str(d.date())}  分值 {s0:6.1f}  T+20 {r*100:+6.2f}%  最深 {m*100:+6.2f}%")
o = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
nt = sum(1 for x in rows if x[3] >= -TOL)
print(f"\n  合计 样本 {len(rows)}  命中 {o} = {o/len(rows)*100:.1f}%  "
      f"未被套 {nt}/{len(rows)} = {nt/len(rows)*100:.1f}%  均T+20 {np.mean([x[2] for x in rows])*100:+.2f}%  "
      f"最差 {min(x[3] for x in rows)*100:+.2f}%  威尔逊下界 {wilson(o,len(rows))*100:.1f}%")

print("\n" + "=" * 120)
print("顶部：两种口径对比")
print("=" * 120)
for tag, ev_all in (("A 向上突破 100（现状）",
                     [(b,) + e for b in C.BOARD_ORDER
                      for e in scan(P[b]["score"], P[b]["close"], "top", 100.0)]),
                    ("B 从 100 上方回落",
                     [(b,) + e for b in C.BOARD_ORDER
                      for e in scan_top_b(P[b]["score"], P[b]["close"])])):
    if not ev_all:
        print(f"  {tag}: 无样本"); continue
    # 顶部成功 = T+20 下跌 且 期间涨幅 ≤ 3%
    ok = sum(1 for x in ev_all if x[2] < 0 and x[4] <= TOL)
    print(f"  {tag:<22} n={len(ev_all):>3}  命中 {ok}/{len(ev_all)} = {ok/len(ev_all)*100:5.1f}%  "
          f"均T+20 {np.mean([x[2] for x in ev_all])*100:+6.2f}%  "
          f"最大反弹 {max(x[4] for x in ev_all)*100:+6.2f}%")

print("\n【如果顶部改为「回落」口径，各板块信号】")
for b in C.BOARD_ORDER:
    ev = scan_top_b(P[b]["score"], P[b]["close"])
    if not ev: continue
    print(f"  {C.BOARDS[b]['name']}")
    for d, r, m, mx, s0 in ev:
        print(f"        {str(d.date())}  分值 {s0:6.1f}  T+20 {r*100:+6.2f}%  期间最大涨 {mx*100:+6.2f}%")
