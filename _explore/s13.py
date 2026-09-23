"""s13 —— 第六轮：候选 SWING-3 的最终评估（页面口径 + 并集增益 + 共振分档）。

前五轮净结论
-----------
有效（保留）：
  1. **ma20_slope** = MA20 的 5 日斜率分位，与现 SWING/SWING-2 正交（Jaccard 0.03~0.05）
  2. **up1 确认** = 分值较前一日回升（实时可判定，非事后挑点）——
     ma20_slope 全程命中 55%→69%，2015 年 27%→43%
  3. **vconv 确认** = 近5日波动 < 近20日波动 —— 2015 年 31%→41%，但对 macd 会砍掉一半信号
无效（否决）：长期位置 hi250/500/750、波动率族、量能族单独、分布族、decel（对 ma20_slope 无效）

本脚本回答
--------
  A. 候选参数在 6 板块口径（页面）下的 (n, 命中)
  B. 并集增益：加进「SWING ∪ SWING-2」后信号数与命中率是否**同时**上升
  C. 共振分档（跨板块同日触发的档位，与现模型同口径）
  D. 与现模型的 Jaccard，确认正交
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s8
import s9                                       # noqa: F401
import s12
from senti import config as C                    # noqa: E402
from s8 import wilson, W_ALL, W_57, W_3          # noqa: E402


def mk_ms(maw, lb):
    """MA 斜率因子（MA 周期 maw，回看 lb 日）。"""
    return (lambda df, m=maw, l=lb: s8.pm(
        df["close"].rolling(m).mean() / df["close"].rolling(m).mean().shift(l) - 1)), 1


def up1(df, i, s):
    return s[i] > s[i - 1]


def sigset(ev):
    return {(r[0], r[1]): r[2] for r in ev}


def stat(d):
    n = len(d)
    k = sum(1 for v in d.values() if v)
    return n, k, (100 * k / n if n else float("nan")), wilson(k, n)


def yr(d, y):
    sub = [v for kk, v in d.items() if kk[1][:4] == y]
    if not sub:
        return "—"
    return f"{sum(sub)}/{len(sub)}={100*sum(sub)/len(sub):.0f}%"


# ---- 候选 ----
CAND = [
    ("MS20/lb5 ≤6 +up1", (20, 5), 6, True),
    ("MS20/lb5 ≤8 +up1", (20, 5), 8, True),
    ("MS20/lb5 ≤10 +up1", (20, 5), 10, True),
    ("MS20/lb5 ≤12 +up1", (20, 5), 12, True),
    ("MS20/lb5 ≤12 无", (20, 5), 12, False),
    ("MS25/lb5 ≤3 +up1", (25, 5), 3, True),
    ("MS20/lb10 ≤3 +up1", (20, 10), 3, True),
    ("MS30/lb3 ≤3 +up1", (30, 3), 3, True),
]


def main():
    E6 = s12.Ev6()
    E9 = s8.Evaluator()
    print(f"6 板块基线：全程 {E6.base(*W_ALL):.1f}%　近5.7年 {E6.base(*W_57):.1f}%"
          f"　近3年 {E6.base(*W_3):.1f}%")
    print(f"9 宽基基线：全程 {E9.base(*W_ALL):.1f}%　近5.7年 {E9.base(*W_57):.1f}%"
          f"　近3年 {E9.base(*W_3):.1f}%\n")

    print("=" * 116)
    print("A. 候选参数（6 板块页面口径 / 9 宽基长历史）")
    print("=" * 116)
    print(f"  {'候选':<20}{'口径':<7}{'n':>6}{'命中':>8}{'下界':>7}{'2015':>13}{'2016':>11}"
          f"{'近5.7年':>15}{'近3年':>15}")
    keep = {}
    for tag, (maw, lb), thr, use_up in CAND:
        s8.FACTORS["__ms"] = mk_ms(maw, lb)
        for lab, E in (("6板块", E6), ("9宽基", E9)):
            S = E.score("__ms")
            cond = up1 if use_up else None
            ev = E.run(S, thr, 4, *W_ALL, cond=cond)
            d = sigset(ev)
            n, k, r, wl = stat(d)
            ev57 = E.run(S, thr, 4, *W_57, cond=cond)
            ev3 = E.run(S, thr, 4, *W_3, cond=cond)
            n57, k57, r57, _ = stat(sigset(ev57))
            n3, k3, r3, _ = stat(sigset(ev3))
            print(f"  {tag:<20}{lab:<7}{n:>6}{r:>7.1f}%{wl:>7.1f}"
                  f"{yr(d,'2015'):>13}{yr(d,'2016'):>11}"
                  f"{f'{r57:.1f}%({n57})':>15}{f'{r3:.1f}%({n3})':>15}")
        keep[tag] = ((maw, lb), thr, use_up)

    print("\n" + "=" * 116)
    print("B. 并集增益（6 板块口径）")
    print("=" * 116)
    for lab, E in (("6板块", E6), ("9宽基", E9)):
        A = sigset(E.run(E.score("pos"), 22, 4, *W_3))
        B = sigset(E.run(E.score("macd"), 12, 4, *W_3))
        U = {**A, **B}
        nU, kU, rU, wlU = stat(U)
        print(f"\n  [{lab}] 近 3 年：SWING {len(A)}/{100*sum(A.values())/len(A):.1f}%　"
              f"SWING-2 {len(B)}/{100*sum(B.values())/len(B):.1f}%　"
              f"并集 {nU}/{rU:.1f}% (下界 {wlU:.1f})")
        print(f"    {'候选':<20}{'新信号':>8}{'新命中':>10}{'并集n':>8}{'并集率':>9}{'Δ':>8}")
        for tag, ((maw, lb), thr, use_up) in keep.items():
            s8.FACTORS["__ms"] = mk_ms(maw, lb)
            S = E.score("__ms")
            Cm = sigset(E.run(S, thr, 4, *W_3, cond=up1 if use_up else None))
            new = {k: v for k, v in Cm.items() if k not in U}
            if len(new) < 15:
                continue
            U1 = {**U, **Cm}
            n1, k1, r1, wl1 = stat(U1)
            nk = sum(1 for v in new.values() if v)
            print(f"    {tag:<20}{len(new):>8}{f'{nk}/{len(new)}':>10}{n1:>8}{r1:>8.1f}%"
                  f"{r1-rU:>+8.1f}")
    s8.FACTORS.pop("__ms", None)

    print("\n" + "=" * 116)
    print("C. 与现模型的 Jaccard（6 板块 · 近5.7年）")
    print("=" * 116)
    A = sigset(E6.run(E6.score("pos"), 22, 4, *W_57))
    B = sigset(E6.run(E6.score("macd"), 12, 4, *W_57))
    print(f"  SWING n={len(A)}　SWING-2 n={len(B)}　J(A,B)={len(set(A)&set(B))/len(set(A)|set(B)):.3f}")
    for tag, ((maw, lb), thr, use_up) in keep.items():
        s8.FACTORS["__ms"] = mk_ms(maw, lb)
        Cm = sigset(E6.run(E6.score("__ms"), thr, 4, *W_57, cond=up1 if use_up else None))
        jA = len(set(Cm) & set(A)) / len(set(Cm) | set(A))
        jB = len(set(Cm) & set(B)) / len(set(Cm) | set(B))
        print(f"  {tag:<20} n={len(Cm):<4} J(pos)={jA:.3f}  J(macd)={jB:.3f}")
    s8.FACTORS.pop("__ms", None)


if __name__ == "__main__":
    main()
