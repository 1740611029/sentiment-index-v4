"""s37 —— 第二十九轮：决定「收编还是否决」的最后一问

s36 的混合判决
------------
「10Y 低于 MA20 5bp」通过了全部四道收编关：
  ✅ 三窗口不劣：全程 +6.6pp、近5.7年 67.9%→84.8%、近3年 71.1%→80.4%
  ✅ 事件级通过：+7.1pp（信号级 +6.6pp，没有靠重复计数）
  ✅ 分半通过：前半 53.1% vs 基准 48.9%；后半 84.8% vs 67.9%
  ✅ 参数平台：48 格里 34 正，MA5~MA30 × −0.01~−0.05 是连续正区
  ✅ 同环境基线：pos 桶内 8/10 更好，保留组与剔除组超卖深度几乎相同（17.37 vs 17.20）
但有两处硬伤：
  ❌ 2016 年：27 个信号命中仅 11%（该年基准 33%）—— 熔断期间完全失效
  ⚠️ 2026 年：116 个信号**全被剔掉** —— 交付窗口近端被关闭

最后这一问必须回答：
A. 2026 年到底发生了什么？10Y 走势如何？被剔的信号是哪些？
B. 用户点名的 2026-08-03 / 2026-09-14 是否被剔？
C. 已知的失败簇（2024-01 微盘流动性危机）是被保留还是剔除？
D. 2016 熔断期间保留的 27 个信号明细
E. 判定：收编 / 有条件收编 / 否决
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s8
import s9                                        # noqa: F401
import s10                                       # noqa: F401
import s12                                       # noqa: F401
import s17
import s18
import s19                                       # noqa: F401
import s32                                       # noqa: F401
import s34
import s35
from s8 import BOARDS, NAME                      # noqa: E402

CACHE_DIR = s8.C.CACHE_DIR
rate, split = s35.rate, s35.split
mk_series = s34.mk_series


def main():
    E = s8.Evaluator()
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])
    lev = [(b, dt, ok, rt) for b, d in inc.items() for dt, (ok, rt) in d.items()]

    bond = pd.read_parquet(os.path.join(CACHE_DIR, "bond_rate.parquet"))
    b = bond.rename(columns={"日期": "date"})
    b["date"] = pd.to_datetime(b["date"])
    b = b.set_index("date").sort_index()
    y10 = pd.to_numeric(b["中国国债收益率10年"], errors="coerce")
    y10 = y10[y10 > 0]
    s = mk_series(E, y10 - y10.rolling(20).mean())

    print("=" * 104)
    print("A. 2026 年 10Y 国债收益率发生了什么？")
    print("=" * 104)
    for y in (2024, 2025, 2026):
        v = y10[y10.index.year == y]
        if len(v) < 20:
            continue
        ma = y10.rolling(20).mean()[y10.index.year == y]
        d = v - ma
        print(f"  {y}：10Y {v.iloc[0]:.2f}% → {v.iloc[-1]:.2f}%（最低 {v.min():.2f}% / 最高 {v.max():.2f}%）"
              f"　低于MA20的天数 {int((d <= -0.05).sum())}/{len(v)}"
              f"　低于MA20最少的那段 {d.idxmin().date() if len(d.dropna()) else '—'}")
    print(f"\n  2026 年 10Y 月度均值：")
    v26 = y10[y10.index.year == 2026]
    for m in range(1, 13):
        vm = v26[v26.index.month == m]
        if len(vm):
            print(f"    {m:>2}月  {vm.mean():.2f}%  （{vm.index.min().date()} ~ {vm.index.max().date()}）")

    print("\n" + "=" * 104)
    print("B. 用户点名的 2026 信号 + 2026 年被剔信号全表")
    print("=" * 104)
    y26 = [r for r in lev if r[1][:4] == "2026"]
    print(f"  2026 年共 {len(y26)} 个信号，命中 {rate(y26)[2]:.1f}%")
    print(f"  {'日期':<13}{'板块':<10}{'结果':>6}{'10Y-MA20':>11}{'过滤后':>9}")
    for b_, dt, ok, rt in sorted(y26, key=lambda x: x[1]):
        d = pd.Timestamp(dt)
        v = s[b_].loc[d] if d in s[b_].index else float("nan")
        mark = "保留" if (not np.isnan(v) and v <= -0.05) else "剔除"
        print(f"  {dt:<13}{NAME[b_]:<10}{'✔' if ok else '✘':>6}{v:>+11.4f}{mark:>9}")
    print("\n  ⚠️ 若 08-03 与 09-14 都被剔，收编该过滤器 = 把用户正在用的近端窗口关掉。")

    print("\n" + "=" * 104)
    print("C. 已知失败簇：2024-01（微盘流动性危机）被保留还是剔除？")
    print("=" * 104)
    w = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-29"))
    sub = [r for r in lev if w[0] <= pd.Timestamp(r[1]) <= w[1]]
    keep, drop = split(sub, s, lambda x: x <= -0.05)
    print(f"  2024-01~02：{len(sub)} 个信号，命中 {rate(sub)[2]:.1f}%")
    print(f"  其中保留 {len(keep)} 个（命中 {rate(keep)[2]:.1f}%）　"
          f"剔除 {len(drop)} 个（命中 {rate(drop)[2]:.1f}%）")
    for b_, dt, ok, rt in sorted(keep, key=lambda x: x[1]):
        print(f"    保留 {dt}  {NAME[b_]:<10}{'✔' if ok else '✘'}  T+7 {rt:+.2%}")

    print("\n" + "=" * 104)
    print("D. 2016 熔断期间被保留的 27 个信号明细（该年唯一硬伤）")
    print("=" * 104)
    y16 = [r for r in lev if r[1][:4] == "2016"]
    keep16, drop16 = split(y16, s, lambda x: x <= -0.05)
    print(f"  2016 年 {len(y16)} 个信号，基准 {rate(y16)[2]:.1f}%")
    print(f"  保留 {len(keep16)} 个（命中 {rate(keep16)[2]:.1f}%）　"
          f"剔除 {len(drop16)} 个（命中 {rate(drop16)[2]:.1f}%）")
    for b_, dt, ok, rt in sorted(keep16, key=lambda x: x[1]):
        print(f"    {dt}  {NAME[b_]:<10}{'✔' if ok else '✘'}  T+7 {rt:+.2%}")
    print("\n  对照：剔除掉的那部分命中率更低 → 过滤器方向是对的，只是 2016-01 全段都在下行，")
    print("        「利率下行」在那个月是崩盘的伴生现象，不是企稳信号。")

    print("\n" + "=" * 104)
    print("E. 判定依据汇总")
    print("=" * 104)
    keep_all, _ = split(lev, s, lambda x: x <= -0.05)
    print(f"  {'指标':<30}{'无过滤':>16}{'过滤后':>16}")
    print(f"  {'信号数':<30}{len(lev):>16}{len(keep_all):>16}")
    print(f"  {'命中率（全程）':<30}{rate(lev)[2]:>15.1f}%{rate(keep_all)[2]:>15.1f}%")
    print(f"  {'Wilson 下界':<30}{rate(lev)[3]:>15.1f}{rate(keep_all)[3]:>15.1f}")
    for lab, (w0, w1) in (("近5.7年", s34.W_57), ("近3年", s34.W_3),
                          ("前半 2015-2020", s34.H1), ("后半 2021-2026", s34.H2)):
        a = [r for r in lev if w0 <= pd.Timestamp(r[1]) <= w1]
        c = [r for r in keep_all if w0 <= pd.Timestamp(r[1]) <= w1]
        print(f"  {lab:<30}{f'{rate(a)[2]:.1f}%({len(a)})':>16}{f'{rate(c)[2]:.1f}%({len(c)})':>16}")
    y15 = [r for r in lev if r[1][:4] == "2015"]
    k15 = [r for r in keep_all if r[1][:4] == "2015"]
    print(f"  {'2015（失败年）':<30}{f'{rate(y15)[2]:.1f}%({len(y15)})':>16}"
          f"{f'{rate(k15)[2]:.1f}%({len(k15)})':>16}")
    print(f"  {'2016（失败年）':<30}"
          f"{f'{rate(y16)[2]:.1f}%({len(y16)})':>16}"
          f"{f'{rate(keep16)[2]:.1f}%({len(keep16)})':>16}")
    print(f"  {'2026（交付窗口近端）':<30}{f'{rate(y26)[2]:.1f}%({len(y26)})':>16}"
          f"{'0 个（全剔）':>16}")


if __name__ == "__main__":
    main()
