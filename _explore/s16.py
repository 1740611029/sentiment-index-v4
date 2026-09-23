"""s16 —— 第九轮：最后一批未试过的方向，确认「推不出更好」为止。

已试并否决：价格位置族、动量族（MACD/ROC/PPO/TSI/CMO）、波动率族（ATR/BBW/vol20）、
量能族（amt_ratio/shr/amt_long/amt_pct）、分布族（skew/kurt/dnvol/ulcer）、
均线族（ma_disp/px_ma60~250）、日内族（clpos/upsh/oc）、时间路径族（consec_dn/days_hi20）、
长期位置（hi250/500/750）、外盘（前序 s6）、成分股广度（前序 s6）、相对强弱（前序 s6）。

本轮补测
-------
  1. K 线形态：长下影（承接）、首次收阳（连续下跌后）
  2. 底部背离：价格创 20 日新低但分值未创新低
  3. 日历效应：星期几、月初月末
  4. 确认强度：up1 vs up_strong（回升 > N 分）vs down1（继续下探）
  5. 冷却期：2/3/4/5/7
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s8
import s9                                       # noqa: F401
from s8 import wilson, W_ALL, W_57, W_3          # noqa: E402

MAW, LB = 20, 5


def ms():
    return (lambda df: s8.pm(df["close"].rolling(MAW).mean()
                             / df["close"].rolling(MAW).mean().shift(LB) - 1)), 1


def st(ev):
    n = len(ev)
    k = sum(1 for r in ev if r[2])
    return n, k, (100 * k / n if n else float("nan")), wilson(k, n)


def rep(E, S, thr, cool, cond, tag):
    ev = E.run(S, thr, cool, *W_ALL, cond=cond)
    n, k, r, wl = st(ev)
    if n == 0:
        print(f"  {tag:<26} n=0")
        return
    ev57 = E.run(S, thr, cool, *W_57, cond=cond)
    ev3 = E.run(S, thr, cool, *W_3, cond=cond)
    n57, k57, r57, _ = st(ev57)
    n3, k3, r3, _ = st(ev3)
    y15 = [x for x in ev if x[1][:4] == "2015"]
    f15 = f"{sum(1 for x in y15 if x[2])}/{len(y15)}" if y15 else "—"
    print(f"  {tag:<26}{n:>5}{r:>7.1f}%{wl:>7.1f}{f15:>10}"
          f"{f'{r57:.1f}%({n57})':>15}{f'{r3:.1f}%({n3})':>15}")


# ---- 确认层变体 ----
def up1(df, i, s):
    return s[i] > s[i - 1]


def mk_upstrong(d):
    return lambda df, i, s: s[i] - s[i - 1] > d


def down1(df, i, s):
    return s[i] < s[i - 1]


# ---- K 线形态（作为确认层） ----
def C_hammer(df, i, s):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    body = abs(c[i] - o[i])
    lower = min(o[i], c[i]) - lo[i]
    return lower > 2 * max(body, 1e-9)


def C_first_up(df, i, s):
    """连续 2 日下跌后首次收阳。"""
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    return c[i] > o[i] and c[i - 1] < o[i - 1] and c[i - 2] < o[i - 2]


# ---- 日历（作为确认层） ----
def C_monday(df, i, s):
    return df.index[i].weekday() == 0


def C_monthturn(df, i, s):
    d = df.index[i].day
    return d <= 5 or d >= 25


def main():
    E = s8.Evaluator()
    s8.FACTORS["__ms"] = ms()
    S = E.score("__ms")
    print(f"9 宽基基线：全程 {E.base(*W_ALL):.1f}%　近5.7年 {E.base(*W_57):.1f}%"
          f"　近3年 {E.base(*W_3):.1f}%")
    print(f"基准 ma20_slope ≤12 +up1：全程 238/66.8%\n")
    print(f"  {'方案':<26}{'n':>5}{'命中':>8}{'下界':>7}{'2015':>10}"
          f"{'近5.7年':>15}{'近3年':>15}")
    print("=" * 100)

    print("[确认层变体]")
    rep(E, S, 12, 4, None, "无确认")
    rep(E, S, 12, 4, up1, "up1 (回升)")
    rep(E, S, 12, 4, down1, "down1 (继续下探)")
    for d in (3, 5, 8, 12):
        rep(E, S, 12, 4, mk_upstrong(d), f"up_strong 回升>{d}分")
    rep(E, S, 12, 4, C_hammer, "长下影(锤子线)")
    rep(E, S, 12, 4, C_first_up, "连跌后首次收阳")

    print("\n[日历效应]")
    rep(E, S, 12, 4, C_monday, "周一")
    rep(E, S, 12, 4, C_monthturn, "月初/月末5日")

    print("\n[冷却期]")
    for cool in (2, 3, 4, 5, 7, 10):
        rep(E, S, 12, cool, up1, f"冷却 {cool}")

    print("\n[阈值（冷却4 + up1）]")
    for thr in (5, 8, 10, 12, 15, 18):
        rep(E, S, thr, 4, up1, f"阈值 {thr}")

    print("\n[背离确认层：价格创20日新低但分值未创新低]")
    def C_div(df, i, s):
        c = df["close"].to_numpy(float)
        if i < 25:
            return False
        if c[i] > c[i - 20:i + 1].min():
            return False
        return s[i] > np.nanmin(s[i - 20:i + 1])
    rep(E, S, 12, 4, C_div, "底部背离")

    s8.FACTORS.pop("__ms", None)


if __name__ == "__main__":
    main()
