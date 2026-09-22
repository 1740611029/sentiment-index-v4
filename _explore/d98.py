"""d98 —— 两个方向的最终数字，供决策。

方向 A「覆盖率优先」：SWING = max(pos, s1_T, s1_L) ≤ 30
   → 能把 2026-08-03 / 09-14 这类回调低点标出来，但命中率只有 ~60%
方向 B「准确率优先」：max(vol, b5) ≤ 30 且 全场共振≥3
   → 命中率 82%+，但只在系统性恐慌期出信号，抓不到 2026 年那两个点

另外补一个**低点定位准确率**：信号日收盘价在 [t-5, t+7] 窗口里处于最低 25% 的比例。
  这是"看起来是不是真的标在底部"的视觉口径，跟"买了能不能赚"是两回事，
  但对"找低点"这个需求来说可能才是用户真正想要的那个 85%。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C
from d80 import build_full, fwd_stats
from d86 import build as build_breadth
from d92 import prepare, hits_for

W0 = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2026-09-18")
COOL = 7
H = 7

PLAN_A = (["pos", "s1_T", "s1_L"], 30, 1)
PLAN_B = (["vol", "b5"], 30, 3)


def window_rank(close: pd.Series, d, back=5, fwd=7):
    """信号日收盘在 [t-5, t+7] 窗口内的分位（0 = 窗口最低）。"""
    i = close.index.get_loc(d)
    seg = close.iloc[max(0, i - back): i + fwd + 1]
    if len(seg) < 5:
        return np.nan
    return float((seg < close.iloc[i]).mean())


def run(D, names, plan, h=H):
    cols, thr, reso_min = plan
    ix = [names.index(c) for c in cols]
    idx = D[C.BOARD_ORDER[0]]["index"]
    masks = {}
    for b in C.BOARD_ORDER:
        M = D[b]["cols"].to_numpy(dtype=float)[:, ix]
        masks[b] = np.all(M <= thr, axis=1) & ~np.isnan(M).any(axis=1)
    mk = pd.DataFrame({b: pd.Series(masks[b], index=D[b]["index"]) for b in C.BOARD_ORDER})
    cnt_all = mk.fillna(0).sum(axis=1)
    nb_by = {b: cnt_all.reindex(D[b]["index"]).fillna(0).to_numpy(dtype=int)
             for b in C.BOARD_ORDER}
    evs = []
    for b in C.BOARD_ORDER:
        hit = D[b][f"hit{h}"]
        nb = nb_by[b]
        last = -10 ** 9
        for i in np.flatnonzero(masks[b]):
            if i < 1 or i >= len(hit) - h or i - last < COOL:
                continue
            if nb[i] < reso_min:
                continue
            last = i
            if not (W0 <= idx[i] <= END):
                continue
            evs.append({"b": b, "d": idx[i], "hit": bool(hit[i]), "reso": int(nb[i])})
    return evs


def report(tag, evs, panels, h=H):
    n = len(evs); ok = sum(1 for e in evs if e["hit"])
    years = (END - W0).days / 365.25
    print(f"\n=== {tag} ===")
    print(f"  命中 {ok}/{n} = {ok/n*100:.1f}%   每板块/年 {n/(6*years):.1f}")
    ys = {}
    for e in evs:
        y = str(e["d"].year); ys.setdefault(y, [0, 0]); ys[y][0] += 1; ys[y][1] += int(e["hit"])
    print("  分年 " + "  ".join(f"{y}:{v[1]}/{v[0]}" for y, v in sorted(ys.items())))
    bs = {}
    for e in evs:
        bs.setdefault(e["b"], [0, 0]); bs[e["b"]][0] += 1; bs[e["b"]][1] += int(e["hit"])
    print("  分板 " + "  ".join(f"{C.BOARDS[b]['name'][:4]}{v[1]}/{v[0]}" for b, v in bs.items()))
    # 低点定位准确率
    rk = []
    for e in evs:
        r = window_rank(panels[e["b"]]["close"], e["d"])
        if not np.isnan(r):
            rk.append(r)
    if rk:
        rk = np.array(rk)
        print(f"  低点定位：{np.mean(rk <= 0.25)*100:.1f}% 的信号处于 [t-5,t+7] 最低 25% 分位"
              f"　（中位分位 {np.median(rk)*100:.0f}%）")
    star = sorted([str(e["d"].date()) for e in evs if e["b"] == "STAR"])
    print(f"  科创板信号 {len(star)} 个，含 08-03 / 09-14: "
          f"{'2026-08-03' in star} / {'2026-09-14' in star}")
    recent = [d for d in star if d >= "2026-07-01"]
    print(f"  科创板 2026-07 之后的信号: {recent}")


if __name__ == "__main__":
    panels = build_full()
    wide = build_breadth()
    D = prepare(panels, wide)
    names = D[C.BOARD_ORDER[0]]["names"]
    for h in (5, 7):
        D = hits_for(D, h)
    report("方向 A：max(pos,s1_T,s1_L) ≤30（覆盖率优先）", run(D, names, PLAN_A), panels)
    report("方向 B：max(vol,b5) ≤30 且共振≥3（准确率优先）", run(D, names, PLAN_B), panels)
