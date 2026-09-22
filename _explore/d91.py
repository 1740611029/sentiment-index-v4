"""d91 —— 从 AND 硬条件改成 **加权连续分值**。

为什么改：AND（"所有分量都 ≤ 阈值"）只能表达"样样都极端"，
而且画不出连续曲线（用户要的是一条指标线，不是一个开关）。
加权平均能表达"整体有多超卖"，更像 SENTI-1 那样的可视图表。

分组（组内等权降噪，组间加权）：
  pos  价格位置：距近期高点多远
  mom  短期动量
  osc  摆动（RSI）
  vol  波动（高波动 = 急）
  cnl  收盘新低
  brd  个股短期广度（项目里已验证信息量最大的一类）
  z    均线偏离（z / 布林带）

共振：不做成过滤器，而是**融进分值** —— 当日越多板块一起超卖，分值越低。
  （SENTI-1 的教训：共振做成过滤器会砍掉 3/4 的信号；这里做成扣分项更平滑）
"""
from __future__ import annotations
import os, sys, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C
from d80 import build_full, fwd_stats
from d81 import short_factors, norm
from d84 import groups_mean
from d86 import build as build_breadth, board_breadth
from d89 import z_factors, EX_IDX

W0 = pd.Timestamp("2021-01-01")
H = 7
COOL = 7
THR_GRID = [6, 8, 10, 12, 15, 18, 20, 25]


def prepare(panels, wide, h=H):
    D = {}
    for b in C.BOARD_ORDER:
        p = panels[b]
        nm = norm(short_factors(p))
        g = groups_mean(nm)
        zn = norm(z_factors(p))
        bb = board_breadth(b, wide).reindex(p.index)
        bb_nm = norm(bb, extra_inv=("dn1", "dn3"))
        g["cnl"] = nm[["cnl5", "cnl10", "cnl20"]].mean(axis=1)
        g["brd"] = bb_nm[["dn1", "dn3", "b5", "b10"]].mean(axis=1)
        g["z"] = zn.mean(axis=1)
        st = fwd_stats(p["close"], h)
        hit = ((st["end"] > 0) & (st["mdd"] >= -0.03)).reindex(g.index).fillna(False)
        D[b] = {"g": g, "hit": hit.to_numpy(dtype=bool), "index": g.index,
                "close": p["close"].reindex(g.index)}
    return D


def make_score(D, w, k_reso=0.0, reso_thr=None):
    """SWING 分值 = 加权平均 − k_reso × (当日别的板块也超卖的个数)。"""
    parts = {}
    for b in C.BOARD_ORDER:
        s = sum(D[b]["g"][k] * v for k, v in w.items() if v) / sum(v for v in w.values() if v)
        parts[b] = s
    if k_reso and reso_thr is not None:
        M = pd.DataFrame(parts)
        cnt = (M <= reso_thr).sum(axis=1)          # 含自己在内的共振个数
        for b in C.BOARD_ORDER:
            parts[b] = parts[b] - k_reso * (cnt - 1)
    return parts


def sim(D, scores, thr, cool=COOL, seg=None, h=H):
    n = ok = 0
    evs = []
    for b in C.BOARD_ORDER:
        s = scores[b].to_numpy(dtype=float)
        hit, idx = D[b]["hit"], D[b]["index"]
        last = -10 ** 9
        for i in np.flatnonzero((s <= thr) & ~np.isnan(s)):
            if i < 1 or i >= len(hit) - h or i - last < cool:
                continue
            last = i
            if seg and not (seg[0] <= idx[i] <= seg[1]):
                continue
            n += 1; ok += int(hit[i])
            evs.append((b, str(idx[i].date()), bool(hit[i])))
    return n, ok, evs


GKEYS = ["pos", "mom", "osc", "vol", "cnl", "brd", "z"]


def main():
    panels = build_full()
    wide = build_breadth()
    D = prepare(panels, wide)
    allhit = pd.concat([pd.Series(D[b]["hit"], index=D[b]["index"]) for b in C.BOARD_ORDER])
    base = allhit[allhit.index >= W0].mean() * 100
    print(f"窗口 {W0.date()} 起　h={H}　基线 {base:.1f}%\n")

    cands = {
        "等权7组": {k: 1 for k in GKEYS},
        "pos+brd+vol": {"pos": 1, "brd": 1, "vol": 1},
        "pos+brd": {"pos": 1, "brd": 1},
        "pos+brd+cnl+vol": {"pos": 1, "brd": 1, "cnl": 1, "vol": 1},
        "pos+brd+z+vol": {"pos": 1, "brd": 1, "z": 1, "vol": 1},
        "brd+vol": {"brd": 1, "vol": 1},
        "pos+brd+mom+vol+cnl": {"pos": 1, "brd": 1, "mom": 1, "vol": 1, "cnl": 1},
    }
    best = None
    for tag, w in cands.items():
        sc = make_score(D, w)
        line = f"  {tag:<22}"
        for thr in (8, 10, 12, 15, 20):
            n, ok, _ = sim(D, sc, thr, seg=(W0, pd.Timestamp("2026-09-18")))
            line += f"  ≤{thr}: {(ok/n*100 if n else 0):5.1f}%({n})"
        print(line)
        for thr in (10, 12):
            n, ok, _ = sim(D, sc, thr, seg=(W0, pd.Timestamp("2026-09-18")))
            if n >= 40 and (best is None or ok / n > best[0]):
                best = (ok / n, n, tag, thr, w)
    print(f"\n最佳（n≥40）: {best[2]} ≤{best[3]}  {best[0]*100:.1f}%  n={best[1]}")

    # ---- 共振融进分值 ----
    print("\n--- 共振融进分值（SWING − k×(其他板块同时超卖的个数)）---")
    w = best[4]; thr0 = best[3]
    for k in (0.0, 2.0, 4.0, 6.0, 8.0):
        for rt in (10, 15, 20):
            if k == 0 and rt != 15:
                continue
            sc = make_score(D, w, k_reso=k, reso_thr=rt)
            n, ok, _ = sim(D, sc, thr0, seg=(W0, pd.Timestamp("2026-09-18")))
            if n >= 25:
                print(f"  k={k:<4} reso_thr={rt:<3} ≤{thr0}  {ok/n*100:5.1f}%  n={n}")

    # ---- 目标日期检查 ----
    sc = make_score(D, w)
    s = sc["STAR"]
    print(f"\n科创板 SWING 分值（{best[2]}）近段:")
    print(s.loc["2026-07-25":].round(1).to_string())


if __name__ == "__main__":
    main()
