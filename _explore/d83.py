"""d83 —— 先搞清楚两件事，再谈建模：

1) **天花板**：事后定义的"局部低点"（过去5天最低 且 未来5天最低）里，
   有多少比例满足命中口径？（T+7 期末>0 且回撤≥−3%）
   任何实时信号都超不过这个数 —— 它是定义上的上界。
   如果天花板只有 70%，那 85% 就不用想了，得回去跟用户谈口径。

2) **画像**：这些局部低点在短期因子上的分位分布，跟全体日差多少。
   只用于理解"低点长什么样"，不用于评估（评估必须逐日、实时）。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C
from d80 import build_full, fwd_stats, hit_mask
from d81 import short_factors, norm, pct_map

W0 = pd.Timestamp("2023-09-20")


def local_lows(close: pd.Series, back=5, fwd=5) -> pd.Series:
    """事后局部低点：过去 back 天最低 且 未来 fwd 天最低（含当日）。"""
    c = close
    is_lo = pd.Series(True, index=c.index)
    for k in range(1, back + 1):
        is_lo &= (c <= c.shift(k).fillna(np.inf))
    for k in range(1, fwd + 1):
        is_lo &= (c <= c.shift(-k).fillna(np.inf))
    return is_lo


def main():
    panels = build_full()
    base_all, hit_all = [], []
    prof = {}
    for b in C.BOARD_ORDER:
        p = panels[b]
        w = p.loc[p.index >= W0]
        close = w["close"]
        ll = local_lows(close)
        hm = hit_mask(close).fillna(False)
        # 因果锚必须在全历史上算，再截断到展示窗口（见 d81 的说明）
        nm = norm(short_factors(p)).loc[w.index]
        base_all.append(hm); hit_all.append(ll)
        for c in nm.columns:
            v = nm[c][ll].dropna()
            a = nm[c][hm.notna()].dropna()
            prof.setdefault(c, [[], []])
            prof[c][0].append(v); prof[c][1].append(a)

    base = pd.concat(base_all).mean()
    ll_all = pd.concat(hit_all)
    print(f"口径: T+7 期末>0 且最深回撤≥−3%　全体基线 {base*100:.1f}%")
    print(f"事后局部低点: {int(ll_all.sum())} 个（6 板块·3 年）")
    print(f"  其中命中率 = {pd.concat(base_all)[ll_all.values].mean()*100:.1f}%  ← 天花板\n")

    rows = []
    for c, (lo, al) in prof.items():
        v = pd.concat(lo); a = pd.concat(al)
        if len(v) < 5:
            continue
        rows.append({"factor": c, "n": len(v),
                     "低点分位均值": v.mean(), "全体分位均值": a.mean(),
                     "差": a.mean() - v.mean()})
    df = pd.DataFrame(rows).sort_values("差", ascending=False)
    print("因子画像（分位 0~100，越低越超卖；「差」越大 = 低点处该因子越低 = 越有区分度）")
    print(df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
