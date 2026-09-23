"""s15 —— 第八轮：把「回升确认」推广到所有正交候选因子，看有没有比 ma20_slope 更好的。

逻辑
----
s11/s12 发现 up1（分值较前一日回升）能把 ma20_slope 的全程命中从 55% 抬到 69%。
既然「拐点确认」有效，就把它套到 s9 正交性表里所有与现模型低重叠的因子上，
统一比较。若某个因子的 (信号数, 命中率) 在**三个窗口**上都优于 ma20_slope，
就换掉它。

同时测两个组合：
  OR 组合   = min(ma20_slope, ppo)     任一维度极端
  AND 组合  = max(ma20_slope, ppo)     两个维度都要到位
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

CAND = ["ma20_slope", "slope_combo", "ppo", "macd_acc", "amt_long", "atr_slope",
        "tsi", "gap", "trend_align", "px_ma120", "vol_ratio", "amt_ratio", "shr",
        "bbw_slope", "ulcer", "maxdd20", "roc10", "days_hi20", "clpos", "rsi_slope"]
THRS = [1, 2, 3, 5, 6, 8, 10, 12, 15, 18, 22]


def up1(df, i, s):
    return s[i] > s[i - 1]


def mk_vconv(k=5, n=20):
    def f(df, i, s):
        c = df["close"].to_numpy(float)
        if i < n + 1:
            return False
        r = np.diff(c[i - n:i + 1]) / c[i - n:i]
        return r[-k:].std() < r.std()
    return f


def st(ev):
    n = len(ev)
    k = sum(1 for r in ev if r[2])
    return n, k, (100 * k / n if n else float("nan")), wilson(k, n)


def main():
    E = s8.Evaluator()
    print(f"9 宽基基线：全程 {E.base(*W_ALL):.1f}%　近5.7年 {E.base(*W_57):.1f}%"
          f"　近3年 {E.base(*W_3):.1f}%")
    print("目标：在全程 n 落在 150~300 的前提下，比较 无确认 / +up1 / +vconv\n")
    print("=" * 116)
    print(f"  {'因子':<13}{'阈值':>5}{'确认':<7}{'n':>6}{'命中':>8}{'下界':>7}"
          f"{'2015':>12}{'近5.7年':>15}{'近3年':>15}")
    print("=" * 116)
    rows = {}
    for nm in CAND:
        S = E.score(nm)
        for cname, cond in (("无", None), ("up1", up1), ("vconv", mk_vconv())):
            best = None
            for t in THRS:
                ev = E.run(S, t, 4, *W_ALL, cond=cond)
                n, k, r, wl = st(ev)
                if not (150 <= n <= 320):
                    continue
                if best is None or r > best[2]:
                    best = (t, n, r, wl, ev)
            if best is None:
                continue
            t, n, r, wl, ev = best
            ev57 = E.run(S, t, 4, *W_57, cond=cond)
            ev3 = E.run(S, t, 4, *W_3, cond=cond)
            n57, k57, r57, _ = st(ev57)
            n3, k3, r3, _ = st(ev3)
            y15 = [x for x in ev if x[1][:4] == "2015"]
            f15 = f"{sum(1 for x in y15 if x[2])}/{len(y15)}" if y15 else "—"
            print(f"  {nm:<13}{t:>5g}{cname:<7}{n:>6}{r:>7.1f}%{wl:>7.1f}{f15:>12}"
                  f"{f'{r57:.1f}%({n57})':>15}{f'{r3:.1f}%({n3})':>15}")
            rows[(nm, cname)] = (t, n, r, wl, r57, n57, r3, n3, f15)

    print("\n" + "=" * 116)
    print("  组合：OR = min(ms, ppo)　AND = max(ms, ppo)　（都加 up1）")
    print("=" * 116)
    S1, S2 = E.score("ma20_slope"), E.score("ppo")
    for lab, fn in (("OR(min)", np.minimum), ("AND(max)", np.maximum)):
        for t in (3, 5, 8, 10, 12, 15):
            Sc = {b: pd.Series(fn(S1[b].to_numpy(float), S2[b].to_numpy(float)),
                               index=S1[b].index) for b in s8.BOARDS}
            ev = E.run(Sc, t, 4, *W_ALL, cond=up1)
            n, k, r, wl = st(ev)
            if not (150 <= n <= 320):
                continue
            ev57 = E.run(Sc, t, 4, *W_57, cond=up1)
            ev3 = E.run(Sc, t, 4, *W_3, cond=up1)
            n57, k57, r57, _ = st(ev57)
            n3, k3, r3, _ = st(ev3)
            y15 = [x for x in ev if x[1][:4] == "2015"]
            f15 = f"{sum(1 for x in y15 if x[2])}/{len(y15)}" if y15 else "—"
            print(f"  {lab:<13}{t:>5g}{'up1':<7}{n:>6}{r:>7.1f}%{wl:>7.1f}{f15:>12}"
                  f"{f'{r57:.1f}%({n57})':>15}{f'{r3:.1f}%({n3})':>15}")

    print("\n" + "=" * 116)
    print("  小结：三个窗口都不弱于 ma20_slope+up1 的组合")
    print("=" * 116)
    ref = rows.get(("ma20_slope", "up1"))
    if ref:
        print(f"  基准 ma20_slope+up1: 全程 {ref[2]:.1f}%  近5.7年 {ref[4]:.1f}%"
              f"  近3年 {ref[6]:.1f}%")
        for k_, v in sorted(rows.items(), key=lambda kv: -kv[1][2]):
            if k_ == ("ma20_slope", "up1"):
                continue
            if v[2] >= ref[2] and v[4] >= ref[4] and v[6] >= ref[6]:
                print(f"  ★ {k_[0]}+{k_[1]}: 全程 {v[2]:.1f}%  近5.7年 {v[4]:.1f}%"
                      f"  近3年 {v[6]:.1f}%")


if __name__ == "__main__":
    main()
