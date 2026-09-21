"""回测：验证溢出区（<0 / >100）是否真的是底部 / 顶部。

判定口径（主口径 T+20）：
  底部事件（score < 0）：20 日后收益 > 0，且期间最深回撤 ≥ −3%
  顶部事件（score > 100）：20 日后收益 < 0，且期间最大踏空 ≤ +3%

连续多日溢出合并为一次事件，取分值最极端的那一天作为信号日。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

HORIZONS = (10, 20, 30)
MAIN_H = 20


def _fwd_stats(close: pd.Series, dates: pd.DatetimeIndex, h: int) -> pd.DataFrame:
    """对每个信号日计算未来 h 日的收益 / 最深回撤 / 最大踏空。"""
    vals = close.values
    pos = {d: i for i, d in enumerate(close.index)}
    rows = []
    for d in dates:
        i = pos.get(d)
        if i is None or i + h >= len(vals):
            continue
        c0 = vals[i]
        seg = vals[i + 1: i + h + 1]
        rows.append({
            "date": d,
            f"ret{h}": seg[-1] / c0 - 1.0,
            f"mdd{h}": seg.min() / c0 - 1.0,      # 最深回撤（底部风险）
            f"run{h}": seg.max() / c0 - 1.0,      # 最大踏空（顶部风险）
        })
    return pd.DataFrame(rows).set_index("date")


def collect_events(panel: pd.DataFrame, low: float = 0.0, high: float = 100.0
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (底部事件, 顶部事件)，已合并连续溢出日。"""
    s = panel["score"].dropna()
    s = s[s.index >= pd.Timestamp(C.BACKTEST_START)]
    close = panel["close"]

    def pick(mask, extreme: str) -> pd.DataFrame:
        if not mask.any():
            return pd.DataFrame()
        # 合并连续段
        grp = (mask != mask.shift()).cumsum()
        out = []
        for _, seg in s[mask].groupby(grp[mask]):
            d = seg.idxmin() if extreme == "min" else seg.idxmax()
            out.append(d)
        idx = pd.DatetimeIndex(sorted(set(out)))
        df = pd.DataFrame(index=idx)
        df["score"] = s.loc[idx].values
        for h in HORIZONS:
            st = _fwd_stats(close, idx, h)
            df = df.join(st, how="left")
        return df.dropna(subset=[f"ret{MAIN_H}"])

    bot = pick(s < low, "min")
    top = pick(s > high, "max")
    return bot, top


def judge(ev: pd.DataFrame, kind: str) -> pd.Series:
    """底部/顶部事件是否成立（主口径 T+20）。"""
    h = MAIN_H
    if kind == "bottom":
        return (ev[f"ret{h}"] > 0) & (ev[f"mdd{h}"] >= -0.03)
    return (ev[f"ret{h}"] < 0) & (ev[f"run{h}"] <= 0.03)


def baseline(panel: pd.DataFrame, kind: str) -> float:
    """随机基线：随便挑一天，满足判定口径的比例。"""
    s = panel["score"].dropna()
    s = s[s.index >= pd.Timestamp(C.BACKTEST_START)]
    close = panel["close"]
    idx = s.index
    st = _fwd_stats(close, idx, MAIN_H)
    if st.empty:
        return float("nan")
    if kind == "bottom":
        ok = (st[f"ret{MAIN_H}"] > 0) & (st[f"mdd{MAIN_H}"] >= -0.03)
    else:
        ok = (st[f"ret{MAIN_H}"] < 0) & (st[f"run{MAIN_H}"] <= 0.03)
    return float(ok.mean())


def summarize(board_name: str, panel: pd.DataFrame,
              low: float = 0.0, high: float = 100.0) -> dict:
    bot, top = collect_events(panel, low, high)
    res = {"board": board_name, "n_days": int(len(panel["score"].dropna()))}
    for kind, ev in (("bottom", bot), ("top", top)):
        res[f"n_{kind}"] = int(len(ev))
        if len(ev):
            ok = judge(ev, kind)
            res[f"ok_{kind}"] = int(ok.sum())
            res[f"rate_{kind}"] = float(ok.mean())
            res[f"ret_{kind}"] = float(ev[f"ret{MAIN_H}"].mean())
            risk = f"mdd{MAIN_H}" if kind == "bottom" else f"run{MAIN_H}"
            res[f"risk_{kind}"] = float(ev[risk].mean())
            res[f"worst_{kind}"] = float(ev[risk].min() if kind == "bottom" else ev[risk].max())
        else:
            res[f"ok_{kind}"] = 0
            res[f"rate_{kind}"] = float("nan")
        res[f"base_{kind}"] = baseline(panel, kind)
    res["_bot"] = bot
    res["_top"] = top
    return res
