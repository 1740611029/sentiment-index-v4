"""诊断 3：近 3 年只有 2 个真底 + 1~2 个真顶。
检查合成 H 在真实拐点日的读数，以及各板块 H 的极值日是否命中真拐点。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, model

big = data.build_stock_indicators()
panels = model.build_all(big)

KEYS = ["2024-02-05", "2024-09-23", "2026-05-13", "2026-06-25", "2026-06-30", "2026-06-22"]

print("="*104)
print("【1】真实拐点日的 H 读数  （大盘/中证2000/中证1000 底=2024-02-05；科创/创业/沪深300 底≈2024-09-23）")
print("="*104)
print(f"{'板块':>8} " + " ".join(f"{k:>12}" for k in KEYS))
for b in C.BOARD_ORDER:
    p = panels[b]
    line = f"{C.BOARDS[b]['name']:>8} "
    for k in KEYS:
        ts = pd.Timestamp(k)
        near = p.index[(p.index >= ts - pd.Timedelta(days=6)) & (p.index <= ts + pd.Timedelta(days=6))]
        v = p.loc[near, "H"].min() if len(near) else np.nan
        line += f"{v:>12.2f}" if pd.notna(v) else f"{'-':>12}"
    print(line)

print("\n" + "="*104)
print("【2】各板块 H 最低 6 天 / 最高 6 天（回测窗口内）—— 看极值日是否就是真拐点")
print("="*104)
for b in C.BOARD_ORDER:
    p = panels[b]
    s = p["H"].dropna(); s = s[s.index >= pd.Timestamp(C.BACKTEST_START)]
    lo = s.nsmallest(6); hi = s.nlargest(6)
    print(f"\n{C.BOARDS[b]['name']}  (真实低点 {p['close'].idxmin().date()} / 真实高点 {p['close'].idxmax().date()})")
    print("   最低:", "  ".join(f"{d.date()}({v:.2f})" for d, v in lo.items()))
    print("   最高:", "  ".join(f"{d.date()}({v:.2f})" for d, v in hi.items()))

print("\n" + "="*104)
print("【3】真实拐点前后 20 日走势（判定真底/真顶的地面真相）")
print("="*104)
for b in C.BOARD_ORDER:
    p = panels[b]
    c = p["close"]
    for lab, d in (("底", c.idxmin()), ("顶", c.idxmax())):
        i = p.index.get_loc(d)
        seg = c.iloc[max(0, i+1): i+21]
        if len(seg) == 0: continue
        print(f"  {C.BOARDS[b]['name']:>8} {lab} {d.date()}  20日后 {(seg.iloc[-1]/c.iloc[i]-1)*100:+6.2f}%  "
              f"最深 {(seg.min()/c.iloc[i]-1)*100:+6.2f}%  最高 {(seg.max()/c.iloc[i]-1)*100:+6.2f}%")
