"""s38 —— 第三十轮：修正 MA 计算口径后，重新判决「10Y 快速下行」过滤器

发现的 bug
---------
`ak.bond_zh_us_rate()` 的「中国国债收益率10年」有 **228 个 NaN**（非交易日/缺失）。
s36 直接对它做 `y10.rolling(20).mean()` —— pandas 默认 `min_periods=20`，
**窗口内只要有 1 个 NaN 就整体返回 NaN**。
20 行窗口含 NaN 的概率 ≈ 1−(1−228/3660)^20 ≈ 72%，
于是 MA20 大面积失效、`y10 − MA20` 只剩零星有效值：

    含 NaN 版：y10 − MA20 ≤ −0.05 的天数 = **368**
    干净版  ：同上                        = **685**（差 1.9 倍）

s36 那张「48 格 34 正、MA5~MA30 连续正区」的漂亮平台，
是在**半截 MA20** 上算出来的 —— 典型的「数据清洗 bug 伪装成模型发现」。

正确做法：先 `dropna()` 压实序列，再算 rolling。本脚本用干净口径重做全部检验。

A. 修正口径后的二维网格（与 s36 A 段对照）
B. 事件级口径（铁律五）
C. 三窗口 + 分半 + 逐年
D. 2026 / 2016 / 2024 三个关键窗口
E. 判决
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
W_Q, W_57, W_3, H1, H2 = s34.W_Q, s34.W_57, s34.W_3, s34.H1, s34.H2
rate, split, in_win = s35.rate, s35.split, s35.in_win
mk_series = s34.mk_series
EXCL = s35.EXCL


def main():
    E = s8.Evaluator()
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])
    lev = [(b, dt, ok, rt) for b, d in inc.items() for dt, (ok, rt) in d.items()]
    base_all = [r for r in lev if not in_win(r, EXCL)]

    bond = pd.read_parquet(os.path.join(CACHE_DIR, "bond_rate.parquet"))
    b = bond.rename(columns={"日期": "date"})
    b["date"] = pd.to_datetime(b["date"])
    b = b.set_index("date").sort_index()
    y10_raw = pd.to_numeric(b["中国国债收益率10年"], errors="coerce")
    y10 = y10_raw.dropna()                      # ★ 关键：先压实，再 rolling
    print("=" * 112)
    print("0. 口径修正确认")
    print("=" * 112)
    dirty = (y10_raw - y10_raw.rolling(20).mean())
    clean = (y10 - y10.rolling(20).mean())
    print(f"  原始序列 {len(y10_raw)} 行，NaN {int(y10_raw.isna().sum())} 个")
    print(f"  含 NaN 版（s36 用的）：有效 MA20 {int(dirty.notna().sum())} 个，"
          f"≤−0.05 的 {int((dirty <= -0.05).sum())} 天")
    print(f"  干净版  （本轮用的）：有效 MA20 {int(clean.notna().sum())} 个，"
          f"≤−0.05 的 {int((clean <= -0.05).sum())} 天")

    def dev(win):
        return y10 - y10.rolling(win).mean()

    # ---------------------------------------------------------- A
    print("\n" + "=" * 112)
    print("A. 修正口径后的二维网格（排除 2015-2016 后的增益，pp）")
    print("=" * 112)
    MAS = [5, 10, 15, 20, 25, 30, 40, 60]
    THRS = [-0.01, -0.02, -0.03, -0.05, -0.08, -0.10, -0.15]
    r_base_sub = rate(base_all)[2]
    grid = {}
    for mw in MAS:
        s = mk_series(E, dev(mw))
        for t in THRS:
            keep, _ = split(lev, s, lambda x, t=t: x <= t)
            if len(keep) < 40:
                grid[(mw, t)] = None
                continue
            ka = [r for r in keep if not in_win(r, EXCL)]
            grid[(mw, t)] = (len(keep), rate(keep)[2], rate(ka)[2] - r_base_sub)
    print(f"  {'MA\\阈值':<8}" + "".join(f"{t:>11.2f}" for t in THRS))
    for mw in MAS:
        row = f"  {mw:<8}"
        for t in THRS:
            g = grid[(mw, t)]
            row += f"{'—':>11}" if g is None else f"{g[2]:>+10.1f}p"
        print(row)
    vals = [v for v in grid.values() if v]
    pos = [v for v in vals if v[2] > 0]
    print(f"\n  有效格 {len(vals)} 个，正格 {len(pos)} 个（{len(pos)/len(vals):.0%}）"
          f"　（s36 含 NaN 版：48 格 34 正 = 71%）")
    if pos:
        print("  正格明细：" + "　".join(
            f"MA{mw}/thr{t:+.2f}→{grid[(mw,t)][2]:+.1f}pp(n={grid[(mw,t)][0]})"
            for (mw, t), v in grid.items() if v and v[2] > 0))

    # ---------------------------------------------------------- B
    print("\n" + "=" * 112)
    print("B. 事件级口径（MA20 / thr −0.05）")
    print("=" * 112)

    def ev_level(rows, gap_days=4):
        by_b = {}
        for b_, dt, ok, rt in rows:
            by_b.setdefault(b_, []).append((dt, ok, rt))
        out = []
        for b_, ev in by_b.items():
            ev.sort()
            last = None
            for dt, ok, rt in ev:
                d = pd.Timestamp(dt)
                if last is None or (d - last).days > gap_days:
                    out.append((b_, dt, ok, rt))
                    last = d
        return out

    s20 = mk_series(E, dev(20))
    keep, _ = split(lev, s20, lambda x: x <= -0.05)
    lev_e, keep_e = ev_level(lev), ev_level(keep)
    print(f"  {'口径':<26}{'n':>7}{'命中':>9}{'下界':>8}{'基准':>9}{'增益':>9}")
    print(f"  {'信号级 · 无过滤':<26}{rate(lev)[0]:>7}{rate(lev)[2]:>8.1f}%"
          f"{rate(lev)[3]:>8.1f}{rate(lev)[2]:>8.1f}%{0.0:>+8.1f}pp")
    print(f"  {'信号级 · 10Y下降过滤':<26}{rate(keep)[0]:>7}{rate(keep)[2]:>8.1f}%"
          f"{rate(keep)[3]:>8.1f}{rate(lev)[2]:>8.1f}%{rate(keep)[2]-rate(lev)[2]:>+8.1f}pp")
    print(f"  {'事件级 · 无过滤':<26}{rate(lev_e)[0]:>7}{rate(lev_e)[2]:>8.1f}%"
          f"{rate(lev_e)[3]:>8.1f}{'—':>9}{'—':>9}")
    print(f"  {'事件级 · 10Y下降过滤':<26}{rate(keep_e)[0]:>7}{rate(keep_e)[2]:>8.1f}%"
          f"{rate(keep_e)[3]:>8.1f}{rate(lev_e)[2]:>8.1f}%"
          f"{rate(keep_e)[2]-rate(lev_e)[2]:>+8.1f}pp")

    # ---------------------------------------------------------- C
    print("\n" + "=" * 112)
    print("C. 三窗口 + 分半 + 逐年")
    print("=" * 112)
    print(f"  {'窗口':<20}{'无过滤':>18}{'过滤后':>18}{'差':>10}")
    for lab, (w0, w1) in (("全程 2015-2026", W_Q), ("近5.7年", W_57), ("近3年", W_3),
                          ("前半 2015-2020", H1), ("后半 2021-2026", H2),
                          ("排除 2015-2016", (pd.Timestamp("2017-01-01"), W_Q[1]))):
        a = [r for r in lev if w0 <= pd.Timestamp(r[1]) <= w1]
        c = [r for r in keep if w0 <= pd.Timestamp(r[1]) <= w1]
        print(f"  {lab:<20}{f'{rate(a)[2]:.1f}%({len(a)})':>18}"
              f"{f'{rate(c)[2]:.1f}%({len(c)})':>18}{rate(c)[2]-rate(a)[2]:>+9.1f}pp")
    print(f"\n  {'年份':<7}{'全样本':>14}{'过滤后':>14}{'差':>9}{'保留率':>9}")
    win = tot = 0
    for y in [str(v) for v in range(2015, 2027)]:
        a = [r for r in lev if r[1][:4] == y]
        c = [r for r in keep if r[1][:4] == y]
        if not a:
            continue
        ra = rate(a)[2]
        tot += 1
        if not c:
            print(f"  {y:<7}{f'{ra:.0f}%({len(a)})':>14}{'全剔':>14}{'':>9}{'0%':>9}")
            continue
        rc = rate(c)[2]
        win += 1 if rc > ra else 0
        print(f"  {y:<7}{f'{ra:.0f}%({len(a)})':>14}{f'{rc:.0f}%({len(c)})':>14}"
              f"{rc-ra:>+9.1f}{len(c)/len(a):>9.0%}")
    print(f"  → 有效年份 {win}/{tot} 变好")

    # ---------------------------------------------------------- D
    print("\n" + "=" * 112)
    print("D. 三个关键窗口")
    print("=" * 112)
    for lab, w0, w1 in (("2024-01 微盘流动性危机", "2024-01-01", "2024-02-29"),
                        ("2016-01 熔断", "2016-01-01", "2016-01-31"),
                        ("2026 交付窗口近端", "2026-01-01", "2026-12-31")):
        sub = [r for r in lev if pd.Timestamp(w0) <= pd.Timestamp(r[1]) <= pd.Timestamp(w1)]
        kp, dp = split(sub, s20, lambda x: x <= -0.05)
        print(f"  {lab}：{len(sub)} 个信号，基准 {rate(sub)[2]:.1f}%"
              f"　→ 保留 {len(kp)} 个（{rate(kp)[2]:.1f}%）/ 剔除 {len(dp)} 个（{rate(dp)[2]:.1f}%）")
    y26 = [r for r in lev if r[1][:4] == "2026"]
    k26 = [r for r in keep if r[1][:4] == "2026"]
    print(f"\n  2026 年 10Y 与 MA20 的最大偏离："
          f"{(y10 - y10.rolling(20).mean())[y10.index.year == 2026].min():+.4f}")
    print(f"  2026 年低于 MA20 达 5bp 的天数："
          f"{int(((y10 - y10.rolling(20).mean())[y10.index.year == 2026] <= -0.05).sum())}"
          f" / {int((y10.index.year == 2026).sum())}")
    print(f"  2026 年被保留的信号：{len(k26)} 个")

    # ---------------------------------------------------------- E
    print("\n" + "=" * 112)
    print("E. 判决")
    print("=" * 112)
    print("  见 DELIVERY.md 第 16 节。")


if __name__ == "__main__":
    main()
