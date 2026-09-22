"""d95 —— 约束驱动搜索：只保留能覆盖科创板 2026-08-03 与 2026-09-14 的候选。

d94 诊断出的关键事实：这两天科创板的 cnl5/10/20/40 全部 = 0.0（收盘创 N 日新低），
z10 14.6/8.8、osc 12.9/12.1、b20 29.2/17.2 也都在低位。
而 vol(27/73)、panic(-16/63)、dn1(55/70)、clpos(4/46) 在两天**完全不一致** ——
凡是含这些因子的组合，必然抓不全这两天。

所以：候选池只放"两天都低"的因子，再要求必须覆盖这两天，剩下的按命中率排序。
"""
from __future__ import annotations
import os, sys, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C
from d80 import build_full
from d92 import prepare, hits_for, sim

W0 = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2026-09-18")
T1, T2 = "2026-08-03", "2026-09-14"
COOL = 7

# 只保留在两天都处于低位的因子（见 d94 诊断表）
POOL = ["cnl5", "cnl10", "cnl20", "cnl40", "vslow5",
        "z5", "z10", "z20", "boll20", "osc", "mom", "pos",
        "s1_T", "s1_L", "s1_score", "b20", "r5", "b5", "b10", "dn3", "body"]
THR_GRID = [8, 10, 12, 15, 20, 25, 30, 35, 40]


def main():
    panels = build_full()
    wide = __import__("d86").build()
    D = prepare(panels, wide)
    names = D[C.BOARD_ORDER[0]]["names"]
    ix_pool = [names.index(c) for c in POOL]
    star = D["STAR"]["cols"].to_numpy(dtype=float)
    star_idx = {d: i for i, d in enumerate(D["STAR"]["index"])}
    i1, i2 = star_idx[pd.Timestamp(T1)], star_idx[pd.Timestamp(T2)]

    for h in (5, 7):
        D = hits_for(D, h)
        rows = []
        for k in (1, 2, 3):
            for combo in itertools.combinations(ix_pool, k):
                for thr in THR_GRID:
                    M = star[:, list(combo)]
                    v1 = np.nanmax(M[i1]); v2 = np.nanmax(M[i2])
                    if not (v1 <= thr and v2 <= thr):      # 硬约束：两天都要触发
                        continue
                    for rm in (1, 2):
                        n, ok, evs = sim(D, list(combo), thr, h, reso_min=rm,
                                         seg=(W0, END))
                        if n < 30:
                            continue
                        years = (END - W0).days / 365.25
                        rows.append({"combo": combo, "thr": thr, "reso": rm,
                                     "n": n, "hit": ok / n * 100,
                                     "peryear": n / (len(C.BOARD_ORDER) * years),
                                     "v1": v1, "v2": v2})
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"h={h}: 无满足约束的候选")
            continue
        df["nm"] = df["combo"].apply(lambda c: "+".join(names[i] for i in c))
        df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"d95_h{h}.csv"), index=False)
        print(f"\n########## h={h}　满足「必须覆盖 {T1} 与 {T2}」的候选 {len(df)} 个 ##########")
        print("--- 按命中率 top15 ---")
        for _, r in df.sort_values("hit", ascending=False).head(15).iterrows():
            print(f"  {r['nm']:<24} ≤{r['thr']:<3} reso≥{r['reso']}  {r['hit']:5.1f}%"
                  f"  n={int(r['n']):<4} 每板块/年 {r['peryear']:4.1f}"
                  f"  (08-03={r['v1']:.0f} 09-14={r['v2']:.0f})")
        print("--- 信号较多（每板块/年 ≥3）里命中率最高的 ---")
        sub = df[df.peryear >= 3]
        if len(sub):
            for _, r in sub.sort_values("hit", ascending=False).head(10).iterrows():
                print(f"  {r['nm']:<24} ≤{r['thr']:<3} reso≥{r['reso']}  {r['hit']:5.1f}%"
                      f"  n={int(r['n']):<4} 每板块/年 {r['peryear']:4.1f}")


if __name__ == "__main__":
    main()
