"""d80 —— 小波段研究：构建全历史面板（锚点预热）+ 定义判定口径 + 算基线。

为什么必须全历史：model.build 会把面板截断到 BACKTEST_START(2023-09-20)，
之后只有 726 天，而因果锚需要 rolling(750, min_periods=500) 预热。
若直接在截断面板上算，前 500 天全是 NaN，样本损失一半。
做法：临时把 BACKTEST_START 改到 HIST_START，拿到全历史面板后自己按窗口切。
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C, model, data

OUT = os.path.join(C.DATA_DIR, "panels_full")

# ---- 判定口径（小波段低点）----
# H   : 观察窗口（交易日），对应"4~10 天波段"取中位数 7
# TOL : 窗口内允许的最深回撤（不被套）
#
# 为什么不用「窗口内涨幅 ≥ 4%」当命中条件（d82 实测）：
#   那是**绝对**涨幅门槛，会系统性歧视低波动板块 ——
#   大盘 7 日内涨 ≥3% 的基线只有 10.8%，科创板 38.1%，差 3.5 倍。
#   拿它当命中率的分母，等于在比「谁波动大」，不是在比「谁找得准」。
#   所以主口径跟 SENTI-1 保持一致：期末为正 + 不被套，基线约 49%（各板块接近）。
#   窗口内最高涨幅另行统计，作为"这波有多大"的参考，不进命中判定。
H, TOL = 7, 0.03
GAIN = 0.03          # 仅供参考统计（波段幅度），不参与命中判定


def build_full(force=False):
    os.makedirs(OUT, exist_ok=True)
    if not force and all(os.path.exists(os.path.join(OUT, f"{b}.parquet")) for b in C.BOARD_ORDER):
        return {b: pd.read_parquet(os.path.join(OUT, f"{b}.parquet")) for b in C.BOARD_ORDER}
    C.BACKTEST_START = C.HIST_START          # 临时：拿到全历史，锚点才有预热
    t0 = time.time()
    panels = model.build_all()
    print(f"  build_all {time.time()-t0:.0f}s")
    for b, p in panels.items():
        p.to_parquet(os.path.join(OUT, f"{b}.parquet"))
    return panels


def fwd_stats(close: pd.Series, h: int = H) -> pd.DataFrame:
    """未来 h 日：最大涨幅 / 最深回撤 / 期末收益（信号日当天收盘买入）。"""
    c = close.to_numpy(dtype=float)
    n = len(c)
    g = np.full(n, np.nan); m = np.full(n, np.nan); e = np.full(n, np.nan)
    for i in range(n - h):
        seg = c[i + 1:i + h + 1]
        g[i] = seg.max() / c[i] - 1.0
        m[i] = seg.min() / c[i] - 1.0
        e[i] = seg[-1] / c[i] - 1.0
    return pd.DataFrame({"gain": g, "mdd": m, "end": e}, index=close.index)


def hit_mask(close: pd.Series, h=H, tol=TOL) -> pd.Series:
    """命中 = T+h 期末收益 > 0 且期间最深回撤 ≥ −tol（与 SENTI-1 同结构）。"""
    s = fwd_stats(close, h)
    return (s["end"] > 0) & (s["mdd"] >= -tol)


if __name__ == "__main__":
    t0 = time.time()
    panels = build_full()
    print(f"{'板块':<10}{'天数':>6}{'基线命中':>10}{'期末>0':>9}{'回撤≥-3%':>10}{'平均涨幅':>10}")
    tot = []
    for b in C.BOARD_ORDER:
        p = panels[b]
        w = p.loc[p.index >= pd.Timestamp("2023-09-20")]
        hm = hit_mask(w["close"]).dropna()
        st = fwd_stats(w["close"]).dropna()
        tot.append(hm)
        print(f"{C.BOARDS[b]['name']:<10}{len(w):>6}{hm.mean()*100:>9.1f}%"
              f"{(st['end']>0).mean()*100:>8.1f}%{(st['mdd']>=-TOL).mean()*100:>9.1f}%"
              f"{st['gain'].mean()*100:>9.2f}%")
    allh = pd.concat(tot)
    print(f"\n合计基线 {allh.mean()*100:.1f}%  (n={len(allh)})")
    print(f"口径: T+{H} 期末收益 > 0 且期间最深收盘回撤 ≥ −{TOL:.0%}")
    print(f"耗时 {time.time()-t0:.0f}s")
