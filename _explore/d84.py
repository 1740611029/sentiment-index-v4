"""d84 —— 合成 SWING 超卖度（小波段），逐日模拟评估。

语义（与 SENTI-1 的"投降"并列但不同）：
  SENTI-1 抓的是 **投降底**（恐慌放量、估值投降）—— 大波段，一年 1~2 次
  SWING   抓的是 **急跌超卖**（连续下跌后情绪宣泄到位）—— 小波段，4~10 天

画像来自 d83（事后局部低点 vs 全体日的分位差）：
  强：dnrun 连跌 / bias5 / rsi6 / ret3 / vshi(距近期高点远) / clpos(收在当日低位)
  弱且反向：vslow(贴近近期低点) / lwsh(长下影) / disp
  → **"距高点远"有用、"贴着低点磨"没用**，与 SENTI-1 里「磨底不是底」的结论同源。

单因子实测（d81，锚点修复后）：
  vol(高波动) lift +22.9 最突出，其次 vshi10 +13.1、vshi20 +12.5、ma20gap +11.7
  → 小波段低点出现在**急跌**之后，不是温和阴跌之后。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C
from d80 import build_full, hit_mask, H, TOL
from d81 import short_factors, norm

W0 = pd.Timestamp("2023-09-20")

# 语义分组：组内等权（降噪），组间加权
GROUPS = {
    "pos": ["vshi10", "vshi20", "ma20gap", "bias5", "bias10"],   # 价格位置：距近期高点多远
    "mom": ["ret3", "ret5", "ret10"],                             # 短期动量
    "osc": ["rsi6", "rsi7", "rsi10", "rsi14"],                    # 摆动指标
    "vol": ["vol"],                                               # 波动（高波动=急跌）
}


def groups_mean(nm: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=nm.index)
    for g, cols in GROUPS.items():
        out[g] = nm[[c for c in cols if c in nm.columns]].mean(axis=1)
    return out


def simulate(score: pd.Series, close: pd.Series, thr: float, cool: int,
             turn: bool = False):
    """逐日模拟：t 日收盘触发 → t 日收盘买入，冷却 cool 个交易日。

    turn=True 时额外要求「今日分值 > 昨日」（掉头），与 SENTI-1 同构。
    返回事件列表（只用当日及历史数据，无未来函数）。
    """
    s = score.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    dates = score.index
    n = len(s)
    hm = hit_mask(close).reindex(dates).to_numpy()
    out = []
    last = -10 ** 9
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]) or hm[i] is None or (isinstance(hm[i], float) and np.isnan(hm[i])):
            continue
        if i - last < cool:
            continue
        cond = s[i] <= thr
        if turn:
            cond = cond and s[i] > s[i - 1]
        if cond:
            last = i
            out.append((str(dates[i].date()), bool(hm[i])))
    return out


def evaluate(panels, w, cool=7, turn=False, verbose=True, tag=""):
    """w: 各语义组权重 dict。返回 (合计命中率, 事件数, 各板块明细, 分值表)"""
    all_ev, per, scores = [], {}, {}
    for b in C.BOARD_ORDER:
        p = panels[b]
        nm = norm(short_factors(p))
        g = groups_mean(nm)
        s = sum(g[k] * v for k, v in w.items()) / sum(w.values())
        s = s.loc[s.index >= W0]
        scores[b] = s
        ev = simulate(s, p["close"].loc[s.index], THR, cool, turn)
        all_ev += ev
        per[b] = (len(ev), sum(1 for _, ok in ev if ok))
    n = len(all_ev); ok = sum(1 for _, o in all_ev if o)
    rate = ok / n * 100 if n else np.nan
    if verbose:
        print(f"{tag:<22} 命中 {ok}/{n} = {rate:5.1f}%   冷却{cool} 掉头{turn}"
              f"   每板块 {n/len(C.BOARD_ORDER):.0f} 次")
        for b in C.BOARD_ORDER:
            nb, ob = per[b]
            print(f"    {C.BOARDS[b]['name']:<9}{ob:>4}/{nb:<4}"
                  f"{ob/nb*100 if nb else 0:6.1f}%")
    return rate, n, per, scores


THR = 20.0     # 分值阈值（0~100，越低越超卖）

if __name__ == "__main__":
    panels = build_full()
    print(f"基线 48.2%　口径 T+{H} 期末>0 且回撤≥−{TOL:.0%}　天花板(事后局部低点) 85.7%\n")

    cands = {
        "等权四组": {"pos": .25, "mom": .25, "osc": .25, "vol": .25},
        "只位置": {"pos": 1.0},
        "只波动": {"vol": 1.0},
        "位置+波动": {"pos": .5, "vol": .5},
        "位置+动量+波动": {"pos": .34, "mom": .33, "vol": .33},
        "位置+波动+摆动": {"pos": .34, "vol": .33, "osc": .33},
    }
    for tag, w in cands.items():
        evaluate(panels, w, tag=tag)
        print()

    # 目标日期检查（用户点名：科创板 2026-09-14 与 2026-08-03 必须能抓到）
    w = cands["等权四组"]
    _, _, _, scores = evaluate(panels, w, verbose=False)
    s = scores["STAR"]
    print("科创板 SWING 分值（等权四组）近段:")
    print(s.loc["2026-07-28":].round(1).to_string())
