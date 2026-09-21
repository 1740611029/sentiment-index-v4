"""交叉校验：长历史面板 vs 生产 3 年面板，在重叠区间是否一致。

如果两者在 2023-09~2026-09 这段对不上，说明 longhist 的拼接/累加有 bug，
d42 的结论就不可信。同时对 2015~2016 做数据体检（停牌、涨跌停占比）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors
import longhist

print("载入长历史面板 + 生产面板 ...", flush=True)
PAN = longhist.build()

PAIR = {"HS300": "HS300", "CSI1000": "CSI1000", "CSI2000": "CSI2000", "CHINEXT": "CHINEXT"}
si = None

print("\n" + "=" * 100)
print("重叠区间一致性（生产口径 vs 长历史口径）")
print("=" * 100)
for lk, pk in PAIR.items():
    try:
        prod = factors.build_board_raw(pk)
    except Exception as e:
        print(f"  {lk} 生产面板构建失败: {e}"); continue
    lp = PAN[lk]
    j = prod.join(lp, how="inner", lsuffix="_p", rsuffix="_l")
    if j.empty:
        print(f"  {lk} 无重叠"); continue
    print(f"\n  【{lk}】重叠 {len(j)} 天  {j.index.min().date()} ~ {j.index.max().date()}")
    for col in ("close", "b20", "b60", "r5", "nh", "lim", "disp", "rsi", "bias", "ret20"):
        a = j[f"{col}_p"].astype(float); b = j[f"{col}_l"].astype(float)
        m = a.notna() & b.notna()
        if not m.any():
            print(f"    {col:<6} 无可比样本"); continue
        d = (a[m] - b[m]).abs()
        rel = (d / (a[m].abs() + 1e-9))
        print(f"    {col:<6} 均绝对差 {d.mean():.6f}   最大 {d.max():.6f}   "
              f"中位相对差 {rel.median()*100:.4f}%   相关系数 {a[m].corr(b[m]):.4f}")

print("\n" + "=" * 100)
print("2015~2016 数据体检（停牌 / 涨跌停 / 广度是否失真）")
print("=" * 100)
for nm, raw in list(PAN.items())[:4]:
    sub = raw.loc["2015-06-01":"2016-03-31"]
    if sub.empty:
        continue
    print(f"\n  【{nm}】{len(sub)} 天")
    print(f"    b20  区间 {sub['b20'].min():.3f} ~ {sub['b20'].max():.3f}   "
          f"最低日 {sub['b20'].idxmin().date()}")
    print(f"    lim  区间 {sub['lim'].min():.3f} ~ {sub['lim'].max():.3f}   "
          f"最低日 {sub['lim'].idxmin().date()}")
    print(f"    disp 区间 {sub['disp'].min():.4f} ~ {sub['disp'].max():.4f}")
    print(f"    rsi  区间 {sub['rsi'].min():.1f} ~ {sub['rsi'].max():.1f}")
    print(f"    close 区间 {sub['close'].min():.1f} ~ {sub['close'].max():.1f}")

print("\n" + "=" * 100)
print("成分股覆盖数（每日有数据的成分股个数，检验停牌/退市影响）")
print("=" * 100)
for nm in ["HS300", "CSI1000", "CHINEXT"]:
    raw = PAN[nm]
    print(f"  【{nm}】")
    for y in (2013, 2015, 2016, 2018, 2020, 2022, 2024, 2026):
        s = raw.loc[raw.index.year == y, "b20"]
        print(f"    {y}: 有值 {s.notna().sum():>4}/{len(s):<4}", end="")
    print()
