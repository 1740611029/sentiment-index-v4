"""K_G 形状扫描：确认「去掉磨底项」是真最优，而不是单调曲线上的一点。

若 K_G 越小越好且负值继续变好 → 说明只是单调拟合，不可信；
若在 0 附近见顶、两侧都变差 → 0 是真最优（= 这一项本身无效，应当移除）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C
from senti import model as M
import longhist
from d54 import (make_anchors, score, entries, wilson, line, EXCLUDE, THR, W2)

PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
A = {nm: make_anchors(raw) for nm, raw in PAN.items()}

print("=" * 120)
print("K_G 扫描（长历史 8 宽基，扩展因果锚，阈值 0）")
print("=" * 120)
for kg in [40, 30, 20, 10, 0, -10, -20, -30]:
    rows = []
    for nm, raw in PAN.items():
        sc = score(raw, A[nm], kg)
        rows += [(nm,) + e for e in entries(sc, raw["close"], raw["amt_pct"], THR)]
    line(rows, f"K_G = {kg:>4}")

print("\n同时看 2015-2016 那段（最难）")
for kg in [40, 30, 20, 10, 0, -10, -20, -30]:
    rows = []
    for nm, raw in PAN.items():
        sc = score(raw, A[nm], kg)
        rows += [(nm,) + e for e in entries(sc, raw["close"], raw["amt_pct"], THR)]
    hard = [x for x in rows if x[1].year in (2015, 2016)]
    line(hard, f"K_G = {kg:>4} · 2015-2016")

print("\n" + "=" * 120)
print("K_G = 0 时，分值区间是否还能跌破 0（需求要求 <0 表示底部）")
print("=" * 120)
for nm in sorted(PAN.keys()):
    raw = PAN[nm]
    sc = score(raw, A[nm], 0.0)
    seg = sc[sc.index >= W2]
    print(f"  {nm:<8} 全程 {np.nanmin(sc.to_numpy()):7.1f} ~ {np.nanmax(sc.to_numpy()):6.1f}   "
          f"交付窗口 {np.nanmin(seg.to_numpy()):7.1f} ~ {np.nanmax(seg.to_numpy()):6.1f}   "
          f"低于0的天数 全程 {(sc < 0).sum():>3} / 窗口 {(seg < 0).sum():>3}")
