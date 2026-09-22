"""d81 —— 小波段：短期因子构造 + 单因子区分度评估。

所有因子都用「因果锚」映射（rolling(750).quantile().shift(1)），
与 SENTI-1 同口径，避免前视。评估用**逐日**统计（每天都是候选信号日），
不做「取每段极值」那种事后挑点。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C
from d80 import build_full, fwd_stats, hit_mask, H, GAIN, TOL

W0 = pd.Timestamp("2023-09-20")
ANCHOR_WIN, ANCHOR_MIN = 750, 500


def pct_map(v: pd.Series, lo_q=2.0, hi_q=98.0, lc=-25.0, hc=125.0) -> pd.Series:
    x = v.astype(float)
    lo = x.rolling(ANCHOR_WIN, min_periods=ANCHOR_MIN).quantile(lo_q / 100).shift(1)
    hi = x.rolling(ANCHOR_WIN, min_periods=ANCHOR_MIN).quantile(hi_q / 100).shift(1)
    d = (hi - lo).replace(0.0, np.nan)
    return (100.0 * (x - lo) / d).clip(lc, hc)


def _rsi(close: pd.Series, n: int) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0.0); dn = (-diff).clip(lower=0.0)
    ru = up.ewm(alpha=1.0 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + ru / rd.replace(0.0, np.nan))


def short_factors(p: pd.DataFrame) -> pd.DataFrame:
    """短期因子（全部只用当日及历史数据）。"""
    close = p["close"].astype(float)
    high = p["high"].astype(float); low = p["low"].astype(float)
    f = pd.DataFrame(index=p.index)

    # --- 价格位置 / 动量 ---
    for n in (6, 7, 10, 14):
        f[f"rsi{n}"] = _rsi(close, n)
    f["ma10"] = close.rolling(10).mean()
    f["bias10"] = close / f["ma10"] - 1.0
    f["bias5"] = close / close.rolling(5).mean() - 1.0
    for n in (3, 5, 10):
        f[f"ret{n}"] = close.pct_change(n)
    # 收盘相对近 N 日最低（0 = 正好是最低）
    for n in (5, 10, 20):
        f[f"vslow{n}"] = close / low.rolling(n).min() - 1.0
        f[f"vshi{n}"] = close / high.rolling(n).max() - 1.0
    # 收盘价新低（分母用 **收盘价** 的滚动最小值，不用 low）
    # 为什么：low 在下跌途中会被不断刷新，vslow 因此长期贴在低位、假信号极多
    #   （实测 vslow5 ≤20 命中 2022 天，命中率 44.9%，比基线还低）。
    #   用 close 的滚动最小值，"cnl=0" 才真正意味着"今天收出了 N 日新低收盘"。
    for n in (5, 10, 20, 40):
        f[f"cnl{n}"] = close / close.rolling(n).min() - 1.0
    f["dret1"] = close.pct_change(1)

    # --- 当日 K 线形态 ---
    rng = (high - low).replace(0.0, np.nan)
    f["clpos"] = (close - low) / rng                      # 收盘在当日振幅中的位置（1=收最高）
    f["oppos"] = (p["open"].astype(float) - low) / rng
    f["body"] = (close - p["open"].astype(float)) / close  # 当日实体涨跌
    f["lwsh"] = (close - low) / close                      # 下影占比（越大=越长下影）

    # --- 连续性 ---
    dn = (close.diff() < 0).astype(int)
    f["dnrun"] = dn.groupby((dn == 0).cumsum()).cumsum()   # 连跌天数
    # 近 N 日下跌天数
    for n in (5, 10):
        f[f"dncnt{n}"] = dn.rolling(n).sum()
    # 恐慌尾盘：当日跌幅越大、收盘越贴近当日最低 → 越大
    f["panic"] = (-close.pct_change(1)) * (1.0 - f["clpos"])
    # 下跌加速度：近 3 日跌幅 / 近 10 日跌幅（>1 = 还在加速，<1 = 跌势趋缓）
    f["accel"] = close.pct_change(3) / close.pct_change(10).replace(0.0, np.nan)

    # --- 现有广度 / 量 ---
    f["b20"] = p["b20"]; f["r5"] = p["r5"]; f["lim"] = p["lim"]
    f["amt_pct"] = p["amt_pct"]
    f["vol"] = p["vol"]
    f["disp"] = p["disp"]
    f["ma20gap"] = close / p["ma20"] - 1.0
    return f


def norm(f: pd.DataFrame, extra_inv=()) -> pd.DataFrame:
    """把因子统一成 0~100 且**越低越超卖**（值小 = 冷 = 可能的小波段低点）。"""
    higher_is_oversold = {"dnrun", "lwsh", "dncnt5", "dncnt10", "vol", "panic"} | set(extra_inv)
    out = pd.DataFrame(index=f.index)
    for c in f.columns:
        v = pct_map(f[c])
        if c in higher_is_oversold:
            v = 100.0 - v
        out[c] = v
    return out


def main():
    panels = build_full()
    raws, hits = {}, {}
    win_ix = {}
    for b in C.BOARD_ORDER:
        p = panels[b]
        w = p.loc[p.index >= W0]
        win_ix[b] = w.index
        # 因子与因果锚都必须在**全历史**上算：
        # 若先截断到 726 天再做 rolling(750, min_periods=500)，前 500 天全是 NaN，
        # 样本只剩 31%，而且锚点没有预热，刻度本身是错的。
        raws[b] = short_factors(p)
        hits[b] = hit_mask(w["close"])
    base = pd.concat(hits.values()).mean()
    print(f"基线 {base*100:.1f}%   (口径: T+{H} 期末>0 且最深回撤 ≥ −{TOL:.0%})\n")

    nm = {b: norm(raws[b]).loc[win_ix[b]] for b in C.BOARD_ORDER}
    for b in C.BOARD_ORDER:
        hits[b] = hits[b].reindex(win_ix[b])
    cols = list(nm[C.BOARD_ORDER[0]].columns)
    rows = []
    for c in cols:
        # 因子处于低分位 = 超卖，直接当作候选信号日
        for thr in (10, 20, 30):
            n = ok = 0
            for b in C.BOARD_ORDER:
                m = (nm[b][c] <= thr).fillna(False) & hits[b].notna()
                n += int(m.sum()); ok += int(hits[b][m].sum())
            rows.append({"factor": c, "thr": thr, "n": n,
                         "hit": (ok / n * 100) if n else np.nan,
                         "lift": ((ok / n) - base) * 100 if n else np.nan})
    df = pd.DataFrame(rows)
    pv = df.pivot(index="factor", columns="thr", values="hit").round(1)
    pn = df.pivot(index="factor", columns="thr", values="n")
    pv["n@20"] = pn[20]
    pv["lift@20"] = df[df.thr == 20].set_index("factor")["lift"].round(1)
    print(pv.sort_values("lift@20", ascending=False).to_string())


if __name__ == "__main__":
    main()
