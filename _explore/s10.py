"""s10 —— 第三轮：候选因子（ma20_slope 等）的深度验证。

s9 的结论与陷阱
--------------
s9 显示 pos 在 t=3 时 71.3%，但那是 pos≤22 的**子集**（阈值更严），已被现 SWING 覆盖，
**没有增量**。真正有价值的是与现模型信号重叠低、且自身命中高的因子。
s9 的正交性表里最亮的是：
    ma20_slope t=3   n=158  73.4%   与并集重叠 16%  增量 133
    slope_combo t=3  n=151  69.5%   重叠 21%
    maxdd20   t=10   n=152  69.1%   重叠 37%

本脚本要回答四个问题
------------------
  1. 阈值细扫：t=0.5~12 逐步收紧，命中率与信号数的取舍曲线
  2. 分年：2015 / 2016（AGENTS.md 的关）会不会连环误判
  3. 并集增益：加进「现 SWING ∪ SWING-2」后，信号数与命中率是否同时提升
  4. 参数平台：MA 窗口与斜率回看期，是平台还是尖峰（尖峰 = 过拟合）
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s8
import s9                                       # noqa: F401  注册 s9 的扩充因子
from s8 import Evaluator, W_ALL, W_57, W_3, wilson  # noqa: E402

CAND = ["ma20_slope", "slope_combo", "maxdd20", "vol_ratio", "amt_long", "shr",
        "macd_acc", "roc10", "ppo", "slope60"]
THRS = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 25]


def sig_ok(E, S, thr, cool, w0, w1) -> dict:
    """{(board,date): ok} —— 去重用。"""
    return {(r[0], r[1]): r[2] for r in E.run(S, thr, cool, w0, w1)}


def rate(d: dict) -> tuple:
    n = len(d)
    k = sum(1 for v in d.values() if v)
    return n, k, (100 * k / n if n else float("nan"))


def show_sweep(E, nm, w0, w1, tag):
    S = E.score(nm)
    print(f"\n  {nm}  [{tag}]")
    print(f"    {'阈值':>6}{'n':>6}{'命中':>8}{'下界':>7}{'2015':>11}{'2016':>11}")
    for t in THRS:
        d = sig_ok(E, S, t, 4, w0, w1)
        n, k, r = rate(d)
        if n == 0:
            continue
        y15 = [(v) for kk, v in d.items() if kk[1][:4] == "2015"]
        y16 = [(v) for kk, v in d.items() if kk[1][:4] == "2016"]
        f15 = f"{sum(y15)}/{len(y15)}" if y15 else "—"
        f16 = f"{sum(y16)}/{len(y16)}" if y16 else "—"
        print(f"    {t:>6g}{n:>6}{r:>7.1f}%{wilson(k,n):>7.1f}{f15:>11}{f16:>11}")


def by_year(E, d: dict, y0="2013") -> str:
    ys = {}
    for (b, dt), ok in d.items():
        ys.setdefault(dt[:4], [0, 0])
        ys[dt[:4]][0] += 1
        ys[dt[:4]][1] += 1 if ok else 0
    return "  ".join(f"{y}:{100*v[1]/v[0]:.0f}%({v[0]})" for y, v in sorted(ys.items()) if y >= y0)


def main():
    E = Evaluator()
    print("=" * 100)
    print("第一部分　候选因子阈值细扫（全程 2013-2026，9 宽基）")
    print("=" * 100)
    for nm in CAND:
        show_sweep(E, nm, *W_ALL, "全程")

    print("\n" + "=" * 100)
    print("第二部分　并集增益（近 5.7 年 2021-01~2026-09）")
    print("=" * 100)
    A = sig_ok(E, E.score("pos"), 22, 4, *W_57)
    B = sig_ok(E, E.score("macd"), 12, 4, *W_57)
    U = {**A, **B}
    nA, kA, rA = rate(A)
    nB, kB, rB = rate(B)
    nU, kU, rU = rate(U)
    print(f"  现 SWING  pos≤22    n={nA:<4} {rA:.1f}%")
    print(f"  现 SWING-2 macd≤12  n={nB:<4} {rB:.1f}%")
    print(f"  并集（页面默认）     n={nU:<4} {rU:.1f}%  基线 {E.base(*W_57):.1f}%")
    print(f"\n  {'候选':<13}{'阈值':>5}{'n_new':>7}{'新命中':>9}{'并集n':>7}{'并集率':>9}{'Δ率':>8}")
    for nm in CAND:
        S = E.score(nm)
        best = None
        for t in THRS:
            C = sig_ok(E, S, t, 4, *W_57)
            new = {k: v for k, v in C.items() if k not in U}
            if len(new) < 60:
                continue
            U1 = {**U, **C}
            n1, k1, r1 = rate(U1)
            if best is None or r1 > best[2]:
                best = (t, len(new), sum(1 for v in new.values() if v), n1, r1)
        if best:
            t, nn, nk, n1, r1 = best
            print(f"  {nm:<13}{t:>5g}{nn:>7}{f'{nk}/{nn}':>9}{n1:>7}{r1:>8.1f}%{r1-rU:>+8.1f}")

    print("\n" + "=" * 100)
    print("第三部分　ma20_slope 分年（全程，重点 2015/2016）")
    print("=" * 100)
    for t in (2, 3, 5, 8, 12):
        d = sig_ok(E, E.score("ma20_slope"), t, 4, *W_ALL)
        n, k, r = rate(d)
        print(f"  t={t:<4g} n={n:<4} {r:5.1f}%  {by_year(E, d)}")

    print("\n" + "=" * 100)
    print("第四部分　参数平台（MA 窗口 × 斜率回看期），看是平台还是尖峰")
    print("=" * 100)
    print(f"  {'MA窗':>6}" + "".join(f"{'lb'+str(lb):>16}" for lb in (3, 5, 8, 10, 15)))
    for maw in (10, 15, 20, 25, 30, 40, 60):
        row = f"  {maw:>6}"
        for lb in (3, 5, 8, 10, 15):
            fn = (lambda df, m=maw, l=lb: s8.pm(
                df["close"].rolling(m).mean() / df["close"].rolling(m).mean().shift(l) - 1))
            s8.FACTORS["__tmp"] = (fn, 1)
            S = E.score("__tmp")
            d = sig_ok(E, S, 3, 4, *W_57)
            n, k, r = rate(d)
            row += f"{r:>10.1f}%({n})"
        print(row)
    s8.FACTORS.pop("__tmp", None)


if __name__ == "__main__":
    main()
