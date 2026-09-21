"""去掉磨底项（K_G=0）后的生产面板复核：阈值重扫 + 逐信号明细 + 稳定性。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, store

H, TOL, COOL = store.H, store.TOL, store.COOL_TRADING


def entries(s, c, thr):
    dates = list(s.index)
    s = s.to_numpy(float); cl = c.to_numpy(float)
    out = []; last = -10 ** 9; n = len(s)
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if s[i - 1] <= thr and s[i] > s[i - 1]:
            last = i
            c0 = cl[i]; seg = cl[i + 1:i + H + 1]
            out.append((dates[i], seg[-1] / c0 - 1, seg.min() / c0 - 1, s[i]))
    return out


def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def line(rows, label):
    if not rows:
        print(f"  {label:<22} 无样本"); return
    o = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
    nt = sum(1 for x in rows if x[3] >= -TOL)
    print(f"  {label:<22} n={len(rows):>3}  命中 {o}/{len(rows):<3} {o/len(rows)*100:5.1f}%   "
          f"未被套 {nt/len(rows)*100:5.1f}%   均T+20 {np.mean([x[2] for x in rows])*100:+6.2f}%   "
          f"最差 {min(x[3] for x in rows)*100:+7.2f}%   下界 {wilson(o,len(rows))*100:5.1f}%")


P = store.load()
print("=" * 120)
print("生产面板 · K_G = 0（去掉磨底项）· 分值区间")
print("=" * 120)
for b in C.BOARD_ORDER:
    p = P[b]; sc = p["score"].dropna()
    print(f"  {C.BOARDS[b]['name']:<8} {len(p):>4} 行  分值 {sc.min():6.1f} ~ {sc.max():6.1f}   "
          f"低于0的天数 {(sc < 0).sum():>3}   空值 {p['score'].isna().sum()}")

print("\n" + "=" * 120)
print("阈值重扫")
print("=" * 120)
for t in [8, 5, 2, 0, -3, -6, -9, -12]:
    rows = []
    for b in C.BOARD_ORDER:
        rows += [(b,) + e for e in entries(P[b]["score"], P[b]["close"], float(t))]
    line(rows, f"阈值 {t:>3}")

print("\n" + "=" * 120)
print("阈值 0 · 逐信号明细")
print("=" * 120)
allrows = []
for b in C.BOARD_ORDER:
    ev = entries(P[b]["score"], P[b]["close"], 0.0)
    allrows += [(C.BOARDS[b]["name"],) + e for e in ev]
    for d, r, m, s0 in ev:
        flag = "✔" if (r > 0 and m >= -TOL) else "✘"
        print(f"  {flag} {C.BOARDS[b]['name']:<8} {str(d.date())}  分值 {s0:6.1f}  "
              f"T+20 {r*100:+6.2f}%  最深 {m*100:+6.2f}%")
line(allrows, "合计")
