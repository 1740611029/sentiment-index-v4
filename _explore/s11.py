"""s11 —— 第四轮：能不能突破「超卖抄底」范式的 2015 边界？

背景
----
s10 的结果很一致：**所有超卖类因子在 2015 年 6-8 月都是 30~45% 命中**，
而全程基线 50.9%。这不是某个因子的问题，是「跌了就买」这个范式的通病：
2015 年是**加速下跌**，任何超卖读数都会连续触发、连续失败。

本脚本测两类「跳出范式」的思路
----------------------------
A. **长期位置因子**：2015-06 是「刚见顶」，close 距 3 年最高点只有 −5%；
   而 2024-02 那种真底距 3 年最高点 −40%。长期位置能把两者分开，
   且与短期超卖正交（SENTI-1 靠广度做到了，小波段模型还没有）。
      hi250 / hi500 / hi750 = close / 近 N 日最高价 − 1

B. **止跌确认层**：现 SWING / SWING-2 都是「首次触阈就买」，没有回升确认。
   SENTI-1 用的是「昨日 ≤ 阈值 且 今日 > 昨日」。这里补上：
      up1  ：分值较前一日回升
      decel：近 5 日跌幅 < 近 20 日跌幅（跌速收窄）
      vconv：近 5 日波动 < 近 20 日波动（波动收敛）
      aconv：近 5 日成交额 < 近 20 日均值（缩量）
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s8
import s9                                       # noqa: F401
from s8 import Evaluator, W_ALL, W_57, W_3, wilson  # noqa: E402


# ---------------------------------------------------------------- A 长期位置
@ s8.F("hi250")
def _h250(df):
    return s8.pm(df["close"] / df["high"].rolling(250).max() - 1)


@ s8.F("hi500")
def _h500(df):
    return s8.pm(df["close"] / df["high"].rolling(500).max() - 1)


@ s8.F("hi750")
def _h750(df):
    return s8.pm(df["close"] / df["high"].rolling(750).max() - 1)


# ---------------------------------------------------------------- 工具
def run_cond(E: Evaluator, S, thr, cool, w0, w1, cond=None, h=7):
    """带条件过滤的逐日模拟。cond(df, i, arrs) -> bool。"""
    out = []
    for b in s8.BOARDS:
        s = S[b].to_numpy(float)
        c = E.CL[b]
        idx = c.index
        e, m = E.FW[b]
        df = E.D[b]
        last = -10 ** 9
        for i in range(1, len(s)):
            if np.isnan(s[i]) or s[i] > thr or i - last < cool:
                continue
            if not (w0 <= idx[i] <= w1) or i + h > len(e) - 1:
                continue
            if cond is not None and not cond(df, i, s):
                continue
            last = i
            out.append((b, str(idx[i].date()), bool(e[i] > 0 and m[i] >= -E.tol),
                        float(e[i])))
    return out


def stat(ev) -> tuple:
    n = len(ev)
    k = sum(1 for r in ev if r[2])
    return n, k, (100 * k / n if n else float("nan")), wilson(k, n)


def year_stat(ev, y) -> str:
    sub = [r for r in ev if r[1][:4] == y]
    if not sub:
        return "—"
    k = sum(1 for r in sub if r[2])
    return f"{k}/{len(sub)}={100*k/len(sub):.0f}%"


# ---------------------------------------------------------------- 条件
def C_up1(df, i, s):
    return s[i] > s[i - 1]


def C_decel(df, i, s):
    c = df["close"].to_numpy(float)
    if i < 21:
        return False
    return (c[i] / c[i - 5] - 1) > (c[i] / c[i - 20] - 1)


def C_vconv(df, i, s):
    c = pd.Series(df["close"].to_numpy(float))
    r = c.pct_change()
    return r.iloc[i - 4:i + 1].std() < r.iloc[i - 19:i + 1].std()


def C_aconv(df, i, s):
    a = df["amount"].to_numpy(float)
    return a[i - 4:i + 1].mean() < a[i - 19:i + 1].mean()


CONDS = {"无": None, "up1": C_up1, "decel": C_decel, "vconv": C_vconv, "aconv": C_aconv}


def main():
    E = Evaluator()
    print("=" * 104)
    print("A. 长期位置因子（全程 2013-2026，9 宽基）")
    print("=" * 104)
    print(f"  {'因子':<8}{'阈值':>6}{'n':>6}{'命中':>8}{'下界':>7}{'2015':>14}{'2016':>12}"
          f"{'近3年':>16}")
    for nm in ("hi250", "hi500", "hi750"):
        S = E.score(nm)
        for t in (2, 5, 10, 15, 20, 25, 30):
            ev = run_cond(E, S, t, 4, *W_ALL)
            n, k, r, wl = stat(ev)
            if n < 60:
                continue
            ev3 = run_cond(E, S, t, 4, *W_3)
            n3, k3, r3, _ = stat(ev3)
            print(f"  {nm:<8}{t:>6g}{n:>6}{r:>7.1f}%{wl:>7.1f}{year_stat(ev,'2015'):>14}"
                  f"{year_stat(ev,'2016'):>12}{f'{r3:.1f}%({n3})':>16}")

    print("\n" + "=" * 104)
    print("B. 止跌确认层：现模型 + 确认条件（全程）")
    print("=" * 104)
    bases = [("pos(现SWING)", "pos", 22), ("macd(现SWING-2)", "macd", 12),
             ("ma20_slope", "ma20_slope", 8), ("ppo", "ppo", 4)]
    print(f"  {'基础':<16}{'确认':<8}{'n':>6}{'命中':>8}{'下界':>7}{'2015':>14}{'2016':>12}"
          f"{'近5.7年':>16}{'近3年':>16}")
    for tag, nm, thr in bases:
        S = E.score(nm)
        for cname, cond in CONDS.items():
            ev = run_cond(E, S, thr, 4, *W_ALL, cond=cond)
            n, k, r, wl = stat(ev)
            if n < 30:
                continue
            ev57 = run_cond(E, S, thr, 4, *W_57, cond=cond)
            ev3 = run_cond(E, S, thr, 4, *W_3, cond=cond)
            n57, k57, r57, _ = stat(ev57)
            n3, k3, r3, _ = stat(ev3)
            print(f"  {tag:<16}{cname:<8}{n:>6}{r:>7.1f}%{wl:>7.1f}"
                  f"{year_stat(ev,'2015'):>14}{year_stat(ev,'2016'):>12}"
                  f"{f'{r57:.1f}%({n57})':>16}{f'{r3:.1f}%({n3})':>16}")

    print("\n" + "=" * 104)
    print("C. 长期位置当「过滤器」：只在长期也跌够时才买")
    print("=" * 104)
    H = E.score("hi750")
    print(f"  {'基础':<16}{'hi750上限':>10}{'n':>6}{'命中':>8}{'下界':>7}{'2015':>14}"
          f"{'2016':>12}{'近5.7年':>16}")
    for tag, nm, thr in bases:
        S = E.score(nm)
        for cap in (100, 50, 30, 20, 10):
            def cond(df, i, s, cap=cap, H=H):
                return True  # 占位，真实过滤在下面
            # 真实实现：把 hi750 分值作为额外门槛
            out = []
            for b in s8.BOARDS:
                sc = S[b].to_numpy(float)
                hh = H[b].reindex(S[b].index).to_numpy(float)
                c = E.CL[b]
                idx = c.index
                e, m = E.FW[b]
                last = -10 ** 9
                for i in range(1, len(sc)):
                    if np.isnan(sc[i]) or sc[i] > thr or i - last < 4:
                        continue
                    if not (W_ALL[0] <= idx[i] <= W_ALL[1]) or i + 7 > len(e) - 1:
                        continue
                    if not (hh[i] <= cap):
                        continue
                    last = i
                    out.append((b, str(idx[i].date()),
                                bool(e[i] > 0 and m[i] >= -E.tol), float(e[i])))
            n, k, r, wl = stat(out)
            if n < 30:
                continue
            print(f"  {tag:<16}{cap:>10}{n:>6}{r:>7.1f}%{wl:>7.1f}"
                  f"{year_stat(out,'2015'):>14}{year_stat(out,'2016'):>12}")


if __name__ == "__main__":
    main()
