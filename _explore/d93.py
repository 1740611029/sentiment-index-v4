"""d93 —— 对候选方案做全面体检，定最终参数。

d92 的关键结果（2021-01 起大样本，含 2021-22 熊市）：
  h=7  vol≤25 + reso≥3        82.0%  n=50
  h=7  panic+r5+z5 ≤10 reso≥3 88.9%  n=27
  h=5  vol+b5 ≤25 reso≥3      92.6%  n=27
但小样本（n≈27）的置信区间太宽，必须体检后再定：
  1) 分年度是否稳定（不能只在某一年好）
  2) 分板块是否稳定（不能只靠科创板撑）
  3) 信号频率（每板块每年几次 —— 一年 1 次的"小波段指标"没法用）
  4) **用户点名的科创板 2026-08-03 / 2026-09-14 是否触发**（硬约束）
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C
from d80 import build_full, fwd_stats
from d81 import short_factors, norm
from d84 import groups_mean
from d86 import build as build_breadth, board_breadth
from d89 import z_factors, EX_IDX
from d92 import prepare, hits_for, sim

W0 = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2026-09-18")
TARGETS = ["2026-08-03", "2026-09-14", "2026-09-11", "2026-08-04", "2026-09-15"]

CANDS = [
    ("vol≤25 reso≥3", ["vol"], 25, 3),
    ("vol≤30 reso≥3", ["vol"], 30, 3),
    ("vol+b5≤25 reso≥3", ["vol", "b5"], 25, 3),
    ("vol+b5≤30 reso≥3", ["vol", "b5"], 30, 3),
    ("vol+b5≤30 reso≥2", ["vol", "b5"], 30, 2),
    ("vol+b5≤25 reso≥2", ["vol", "b5"], 25, 2),
    ("vol+dn3+b5≤30 reso≥3", ["vol", "dn3", "b5"], 30, 3),
    ("panic+r5+z5≤10 reso≥3", ["panic", "r5", "z5"], 10, 3),
    ("pos+b20+z5≤10 reso≥3", ["pos", "b20", "z5"], 10, 3),
    ("pos+mom+b10≤8 reso≥3", ["pos", "mom", "b10"], 8, 3),
    ("vol≤25 reso≥1", ["vol"], 25, 1),
    ("vol+b5≤30 reso≥1", ["vol", "b5"], 30, 1),
]


def main():
    panels = build_full()
    wide = build_breadth()
    D = prepare(panels, wide)
    names = D[C.BOARD_ORDER[0]]["names"]
    for h in (5, 7):
        D = hits_for(D, h)

    # 科创板分值序列（用于检查目标日期）
    def star_vals(cols, thr):
        ix = [names.index(c) for c in cols]
        M = D["STAR"]["cols"].to_numpy(dtype=float)[:, ix]
        s = np.nanmax(M, axis=1)
        return pd.Series(s, index=D["STAR"]["index"])

    print("科创板候选分值（≤阈值即触发）:")
    for tag, cols, thr, rm in CANDS:
        sv = star_vals(cols, thr)
        seg = sv.loc["2026-07-28":"2026-09-18"]
        hit_d = [d for d in TARGETS if d in sv.index and sv[d] <= thr]
        print(f"  {tag:<24} 08-03={sv.get('2026-08-03', float('nan')):6.1f} "
              f"09-14={sv.get('2026-09-14', float('nan')):6.1f} "
              f"09-11={sv.get('2026-09-11', float('nan')):6.1f} 触发={hit_d}")

    for h in (5, 7):
        print(f"\n##################### h={h} #####################")
        for tag, cols, thr, rm in CANDS:
            ix = [names.index(c) for c in cols]
            n, ok, evs = sim(D, ix, thr, h, reso_min=rm, seg=(W0, END))
            if n == 0:
                continue
            rate = ok / n * 100
            years = (END - W0).days / 365.25
            peryear = n / (len(C.BOARD_ORDER) * years)
            # 分年度
            ys = {}
            for b, d, hit in evs:
                y = d[:4]
                ys.setdefault(y, [0, 0])
                ys[y][0] += 1; ys[y][1] += int(hit)
            ystr = " ".join(f"{y}:{v[1]}/{v[0]}" for y, v in sorted(ys.items()))
            # 分板块
            bs = {}
            for b, d, hit in evs:
                bs.setdefault(b, [0, 0])
                bs[b][0] += 1; bs[b][1] += int(hit)
            bstr = " ".join(f"{C.BOARDS[b]['name'][:4]}{v[1]}/{v[0]}" for b, v in bs.items())
            print(f"  {tag:<24} {rate:5.1f}% n={n:<4} 每板块/年 {peryear:4.1f}")
            print(f"      分年 {ystr}")
            print(f"      分板 {bstr}")


if __name__ == "__main__":
    main()
