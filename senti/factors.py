"""因子层：把个股指标长表按板块聚合为广度因子，并叠加指数自身技术因子。
所有计算只用当日及历史数据，无未来函数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0.0)
    dn = (-diff).clip(lower=0.0)
    # Wilder 平滑
    ru = up.ewm(alpha=1.0 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = ru / rd.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def build_board_raw(board_key: str, stock_ind: pd.DataFrame | None = None) -> pd.DataFrame:
    """返回该板块每日原始因子表（含 close 供回测用）。"""
    if stock_ind is None:
        stock_ind = data.build_stock_indicators()

    codes = set(data.load_universe(board_key))
    sub = stock_ind[stock_ind["code"].isin(codes)]
    if sub.empty:
        raise RuntimeError(f"{board_key} 无成分股数据")

    g = sub.groupby("date", sort=True)
    breadth = pd.DataFrame({
        "b20": g["above_ma20"].mean(),
        "b60": g["above_ma60"].mean(),
        "r5":  g["up5"].mean(),
        "nh":  g["is_nh60"].mean() - g["is_nl60"].mean(),
        "lim": g["any_lu5"].mean() - g["any_ld5"].mean(),
        "amt": g["amount"].sum(),
        # 截面离散度：个股 5 日收益的标准差。
        # 暴跌时所有股一起跌 → 离散度低；磨底时走势高度一致 → 同样低。
        # （分母用 |均值| 无关的裸标准差，配合自身分位使用）
        "disp": g["ret5"].std(),
        "n":   g["above_ma20"].size(),
    })
    breadth.index = pd.to_datetime(breadth.index)

    idx = data.load_index(board_key).set_index("date")
    close = idx["close"].astype(float)
    amt_idx = idx["amount"].astype(float)

    df = pd.DataFrame(index=idx.index)
    df["close"] = close
    df = df.join(breadth, how="left")

    # 广度类需要足够样本才可信
    df.loc[df["n"] < 20, ["b20", "b60", "r5", "nh", "lim"]] = np.nan

    # 成交额 → 250 日分位（滚动，只用历史）
    amt = df["amt"].fillna(amt_idx)
    df["amt_pct"] = amt.rolling(250, min_periods=60).rank(pct=True)

    # ---- 指数技术因子 ----
    df["rsi"] = _rsi(close, 14)
    ma60 = close.rolling(60).mean()
    df["bias"] = close / ma60 - 1.0
    df["ret20"] = close.pct_change(20)
    df["vol"] = close.pct_change().rolling(20).std()
    df["ma20"] = close.rolling(20).mean()
    df["ma5"] = close.rolling(5).mean()

    return df.drop(columns=["amt", "n"])


def rolling_z(s: pd.Series, window: int = None, min_periods: int = None,
              clip: float = None) -> pd.Series:
    """滚动 z-score；历史不足时用 expanding 兜底（仅用过去数据）。"""
    cfg = C.MODEL
    window = window or cfg["z_window"]
    min_periods = min_periods or cfg["z_min_periods"]
    clip = clip if clip is not None else cfg["z_clip"]

    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    # 兜底：开头用 expanding（同样只用历史）
    mu_e = s.expanding(min_periods=min_periods).mean()
    sd_e = s.expanding(min_periods=min_periods).std()
    mu = mu.fillna(mu_e)
    sd = sd.fillna(sd_e)
    z = (s - mu) / sd.replace(0.0, np.nan)
    return z.clip(-clip, clip)


def to_zpanel(raw: pd.DataFrame) -> pd.DataFrame:
    """原始因子 → z-score 面板（vol 已按反向处理）。"""
    z = pd.DataFrame(index=raw.index)
    for k in C.WEIGHTS:
        col = raw["amt_pct"] if k == "amt" else raw[k]
        zz = rolling_z(col.astype(float))
        if C.INVERT.get(k):
            zz = -zz
        z[k] = zz
    return z
