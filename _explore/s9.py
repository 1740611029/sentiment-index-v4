"""s9 —— 第二轮：阈值扫描 + 正交性（Jaccard）+ 分年，找「帕累托更优且与现模型不重叠」的因子。

第一轮（s8）在统一阈值 12 下的排序：
    ma20_slope 530/65.3%  >  maxdd20 279/63.8%  >  pos 428/62.9%  >  macd 425/56.2%

但不同因子的分位分布不同，用同一个阈值不公平。本脚本：
  1. 对每个因子扫阈值 → 帕累托前沿（同样信号数下谁命中更高）
  2. 算因子两两之间的信号 Jaccard（近 5.7 年）→ 谁与现 SWING/SWING-2 正交
  3. 打印 2015 / 2016（AGENTS.md 的关）

正交性是关键：并集要同时提升「信号数」和「胜率」，新增因子必须与已有信号**不重叠**。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s8
from s8 import FACTORS, BOARDS, Evaluator, W_ALL, W_57, W_3, wilson  # noqa: E402


# ---------------------------------------------------------------- 扩充因子
@ s8.F("lo20_gap")
def _lo20(df):
    """距 20 日最低点有多远（负 = 已破新低）。"""
    return s8.pm(df["close"] / df["low"].rolling(20).min() - 1)


@ s8.F("nn20")
def _nn20(df):
    """近 20 日创 20 日新低的天数占比。"""
    lo = df["low"].rolling(20).min()
    return s8.pm((df["low"] <= lo).rolling(20).mean())


@ s8.F("vol_adj_mom")
def _vam(df):
    """波动率调整动量：10 日收益 / 20 日波动（类夏普）。"""
    r = df["close"].pct_change()
    return s8.pm((df["close"] / df["close"].shift(10) - 1)
                 / r.rolling(20).std().replace(0, np.nan))


@ s8.F("slope60")
def _sl60(df):
    m = df["close"].rolling(60).mean()
    return s8.pm(m / m.shift(10) - 1)


@ s8.F("slope_combo")
def _slc(df):
    """MA5 / MA10 / MA20 / MA60 斜率的等权平均。"""
    c = df["close"]
    parts = []
    for n in (5, 10, 20, 60):
        m = c.rolling(n).mean()
        parts.append(s8.pm(m / m.shift(5) - 1))
    return sum(parts) / len(parts)


@ s8.F("rsi_slope")
def _rsl(df):
    """RSI14 的 3 日变化（负 = RSI 还在往下）。"""
    r = s8._rsi(df["close"], 14)
    return s8.pm(r - r.shift(3))


@ s8.F("range_pos")
def _rp(df):
    """20 日区间位置（经典 %R）：0 = 收在区间最低。"""
    lo = df["low"].rolling(20).min()
    hi = df["high"].rolling(20).max()
    return s8.pm((df["close"] - lo) / (hi - lo).replace(0, np.nan))


@ s8.F("cmo")
def _cmo(df):
    """Chande 动量振荡器（20 日）。"""
    d = df["close"].diff()
    up = d.clip(lower=0).rolling(20).sum()
    dn = (-d.clip(upper=0)).rolling(20).sum()
    return s8.pm((up - dn) / (up + dn).replace(0, np.nan) * 100)


@ s8.F("ulcer")
def _ulcer(df):
    """溃疡指数：20 日回撤的均方根。"""
    c = df["close"]
    dd = c / c.rolling(20).max() - 1
    return s8.pm(np.sqrt((dd ** 2).rolling(20).mean()))


@ s8.F("atr_slope")
def _asl(df):
    """ATR 的 5 日变化率（负 = 波动在收缩）。"""
    a = s8._atr(df, 14)
    return s8.pm(a / a.shift(5) - 1)


@ s8.F("amt_long")
def _al(df):
    """成交额 / 其 60 日均值（中期缩量）。"""
    a = df["amount"]
    return s8.pm(a / a.rolling(60).mean())


@ s8.F("trend_align")
def _ta(df):
    """均线空头排列程度：MA5/MA20 − 1 与 MA20/MA60 − 1 之和（负 = 空头）。"""
    c = df["close"]
    return s8.pm((c.rolling(5).mean() / c.rolling(20).mean() - 1)
                 + (c.rolling(20).mean() / c.rolling(60).mean() - 1))


@ s8.F("bbw_slope")
def _bsl(df):
    """布林带宽的 10 日变化率（负 = 带宽收缩，挤压）。"""
    c = df["close"]
    b = 4 * c.rolling(20).std() / c.rolling(20).mean()
    return s8.pm(b / b.shift(10) - 1)


@ s8.F("macd_hist_pct")
def _mhp(df):
    """MACD 柱的 60 日分位（短窗因果分位，比 750 日锚更灵敏）。"""
    c = df["close"]
    mh = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    h = (mh - mh.ewm(span=9, adjust=False).mean()) / c
    return h.rolling(250, min_periods=120).rank(pct=True) * 100


# ---------------------------------------------------------------- 评估工具
def sig_dates(E: Evaluator, S, thr, cool, w0, w1) -> set:
    """返回信号集合 {(board, date)}。"""
    return {(r[0], r[1]) for r in E.run(S, thr, cool, w0, w1)}


def jac(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else float("nan")


def sweep(E: Evaluator, names, thrs, cool=4, win="57", w=(W_57)) -> dict:
    out = {}
    for nm in names:
        S = E.score(nm)
        rows = []
        for t in thrs:
            ev = E.run(S, t, cool, *w)
            k = sum(1 for r in ev if r[2])
            rows.append((t, len(ev), 100 * k / len(ev) if ev else float("nan"),
                         wilson(k, len(ev))))
        out[nm] = rows
    return out


THRS = [3, 5, 8, 10, 12, 15, 18, 22, 25, 30]


def pareto(rows) -> list:
    """(n, rate) 帕累托前沿：n 更大且 rate 更高的点。"""
    pts = sorted([r for r in rows if r[1] >= 40], key=lambda r: r[1])
    best, out = -1, []
    for r in reversed(pts):
        if r[2] > best:
            best = r[2]
            out.append(r)
    return list(reversed(out))


if __name__ == "__main__":
    E = Evaluator()
    names = list(FACTORS)
    print(f"因子数 {len(names)}；窗口 近5.7年 2021-01~2026-09；冷却 4；判定 T+7 期末>0 且回撤≥−3%")
    print(f"基线（近5.7年） {E.base(*W_57):.1f}%\n")

    res = sweep(E, names, THRS, 4, w=W_57)

    print("== 帕累托前沿（信号数 ≥40 才纳入；只列前沿点）==")
    for nm in names:
        pf = pareto(res[nm])
        s = "  ".join(f"n{p[1]}/{p[2]:.1f}%(t{p[0]:g})" for p in pf[:5])
        best = max(res[nm], key=lambda r: r[2])
        print(f"{nm:<13} {s}")
    print()

    # 正交性：现 SWING(pos≤22) 与 SWING-2(macd≤12) 作为基准
    A = sig_dates(E, E.score("pos"), 22, 4, *W_57)
    B = sig_dates(E, E.score("macd"), 12, 4, *W_57)
    print(f"现 SWING  n={len(A)}　SWING-2 n={len(B)}　两者 Jaccard {jac(A, B):.3f}"
          f"　并集 n={len(A | B)}")
    print("\n== 各因子（取与 SWING-2 同量级 n≈150~250 的阈值）与现模型的 Jaccard ==")
    for nm in names:
        if nm in ("pos", "macd"):
            continue
        # 选一个 n 落在 150~260 的阈值，找不到就取 n 最接近 200 的
        cand = [r for r in res[nm] if 150 <= r[1] <= 260]
        r = max(cand, key=lambda x: x[2]) if cand else min(res[nm], key=lambda x: abs(x[1] - 200))
        C_ = sig_dates(E, E.score(nm), r[0], 4, *W_57)
        if not C_:
            continue
        un = A | B
        print(f"{nm:<13} t={r[0]:<3g} n={r[1]:<4} {r[2]:5.1f}%  "
              f"J(pos)={jac(C_,A):.3f}  J(macd)={jac(C_,B):.3f}  "
              f"与并集重叠={100*len(C_&un)/len(C_):.0f}%  "
              f"并集增量 n={len(un|C_)-len(un)}")
