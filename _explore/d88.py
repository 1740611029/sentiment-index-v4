"""d88 —— 两件事：

1) **前沿曲线**：命中率随样本量怎么变。d87 里 top 全是 n≈30 的小样本，
   81.2% 的 95% 置信区间大约是 63%~93%，根本不可信。这里按 n 分桶看
   "想要多少样本，最多能换到多少命中率"，避免被小样本骗了。

2) **口径再检验**：T+7 期末>0 且回撤≥−3% 的天花板是 85.7%，
   等于说即使能完美识别每一个局部低点，也只有 85.7%。
   换个更贴近"买到低点"的口径（只看期末为正，回撤另行报告），
   天花板和可达命中率各是多少。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C
from d80 import build_full, fwd_stats, H, TOL
from d83 import local_lows
from d87 import prepare, sim
from d86 import build as build_breadth

W0 = pd.Timestamp("2023-09-20")


def main():
    panels = build_full()

    # ---------- 1) 各口径的天花板 ----------
    print("=== 天花板（事后局部低点：过去5天最低 且 未来5天最低）===")
    for h in (5, 7, 10):
        ll_all, k1, k2, k3 = [], [], [], []
        for b in C.BOARD_ORDER:
            p = panels[b]
            w = p.loc[p.index >= W0]
            ll = local_lows(w["close"])
            s = fwd_stats(w["close"], h).loc[w.index]
            ok = s["end"].notna()
            ll = ll & ok
            ll_all.append(ll)
            k1.append(((s["end"] > 0) & (s["mdd"] >= -0.03))[ll])
            k2.append((s["end"] > 0)[ll])
            k3.append(((s["gain"] >= 0.03) & (s["mdd"] >= -0.03))[ll])
        llc = pd.concat(ll_all)
        print(f"  h={h:<3} 低点 {int(llc.sum()):>3} 个　"
              f"期末>0且回撤≥-3% {pd.concat(k1).mean()*100:5.1f}%　"
              f"仅期末>0 {pd.concat(k2).mean()*100:5.1f}%　"
              f"涨≥3%且回撤≥-3% {pd.concat(k3).mean()*100:5.1f}%")

    # ---------- 2) 前沿曲线 ----------
    g = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d87_grid.csv"))
    print(f"\n=== 前沿曲线（d87 共 {len(g)} 个候选，口径 T+7 期末>0 且回撤≥−3%）===")
    buckets = [(30, 50), (50, 80), (80, 120), (120, 200), (200, 400), (400, 10 ** 9)]
    for lo, hi in buckets:
        sub = g[(g.n >= lo) & (g.n < hi)]
        if sub.empty:
            continue
        r = sub.loc[sub["hit"].idxmax()]
        print(f"  n∈[{lo},{hi})  最佳 {r['hit']:.1f}%  n={int(r['n']):<4} {r['nm']} ≤{r['thr']}")
        # 顺便看一下这个桶里 top5 的平均，判断是稳健还是单个尖峰
        t5 = sub.sort_values("hit", ascending=False).head(5)
        print(f"              top5 均值 {t5['hit'].mean():.1f}%"
              f"（若是尖峰，均值会明显低于最佳值）")


if __name__ == "__main__":
    main()
