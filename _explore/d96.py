"""d96 —— 最终方案：分层指标。

实测出来的死结（两个要求在同一信号集上不可兼得）：
  · 命中率 82~92% 的方案必然含"急跌/高波动 + 全场共振"，
    只在 2022/2024 的系统性恐慌期出信号，抓不到 2026-08-03 / 09-14；
  · 能覆盖这两天（科创板 09-14 那天波动率分位 72.9 = **温和阴跌创新低**，不是急跌）
    的方案，命中率上限只有 64%。

所以做成**分层**，一个指标两套用法：
  SWING 分值 = max(cnl10, z10, osc, s1_T)  ← 覆盖用户点名的两天
    基础信号（≤THR）：能把每一波回调的低点标出来
    A 级：再叠加"急跌(vol低) + 全场共振(reso≥3)"  ← 高准确率档
    B 级：共振 2
    C 级：其余

本脚本验证各档的命中率与频率，并确认目标日被覆盖。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C
from d80 import build_full
from d86 import build as build_breadth
from d92 import prepare, hits_for

W0 = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2026-09-18")
COOL = 7
BASE = ["cnl10", "z10", "osc", "s1_T"]


def events(D, cond_idx, thr, h, extra=None, cool=COOL, seg=(W0, END)):
    """extra: (列名, 阈值) 的附加确认层；返回 [(board, date, hit, nboard)]"""
    names = D[C.BOARD_ORDER[0]]["names"]
    idx = D[C.BOARD_ORDER[0]]["index"]
    masks = {}
    for b in C.BOARD_ORDER:
        M = D[b]["cols"].to_numpy(dtype=float)[:, cond_idx]
        m = np.all(M <= thr, axis=1) & ~np.isnan(M).any(axis=1)
        if extra:
            col = names.index(extra[0])
            m = m & (D[b]["cols"].to_numpy(dtype=float)[:, col] <= extra[1])
        masks[b] = m
    mk = pd.DataFrame({b: pd.Series(masks[b], index=D[b]["index"]) for b in C.BOARD_ORDER})
    nb_all = mk.reindex(idx).fillna(0).sum(axis=1).to_numpy(dtype=int)
    base_mk = pd.DataFrame({b: pd.Series(
        np.all(D[b]["cols"].to_numpy(dtype=float)[:, cond_idx] <= thr, axis=1)
        & ~np.isnan(D[b]["cols"].to_numpy(dtype=float)[:, cond_idx]).any(axis=1),
        index=D[b]["index"]) for b in C.BOARD_ORDER})
    nb_base = base_mk.reindex(idx).fillna(0).sum(axis=1).to_numpy(dtype=int)
    out = []
    for b in C.BOARD_ORDER:
        hit = D[b][f"hit{h}"]
        last = -10 ** 9
        for i in np.flatnonzero(masks[b]):
            if i < 1 or i >= len(hit) - h or i - last < cool:
                continue
            last = i
            if seg and not (seg[0] <= idx[i] <= seg[1]):
                continue
            out.append((b, str(idx[i].date()), bool(hit[i]),
                        int(nb_base[i]), int(nb_all[i])))
    return out


def main():
    panels = build_full()
    wide = build_breadth()
    D = prepare(panels, wide)
    names = D[C.BOARD_ORDER[0]]["names"]
    ix = [names.index(c) for c in BASE]
    D = hits_for(D, 7)
    D = hits_for(D, 5)

    # 科创板分值，确认目标日覆盖
    sv = pd.Series(np.nanmax(D["STAR"]["cols"].to_numpy(dtype=float)[:, ix], axis=1),
                   index=D["STAR"]["index"])
    print("科创板 SWING 分值（max(cnl10,z10,osc,s1_T)）近段:")
    print(sv.loc["2026-07-29":].round(1).to_string())
    print(f"\n08-03 = {sv['2026-08-03']:.1f}   09-14 = {sv['2026-09-14']:.1f}")

    for thr in (12, 15, 18, 20, 22, 25, 30):
        for h in (5, 7):
            evs = events(D, ix, thr, h)
            if not evs:
                continue
            n = len(evs); ok = sum(1 for e in evs if e[2])
            years = (END - W0).days / 365.25
            # 分层
            A = [e for e in evs if e[4] >= 3]
            B = [e for e in evs if e[4] == 2]
            Cl = [e for e in evs if e[4] <= 1]
            def st(x):
                return (f"{sum(1 for e in x if e[2])}/{len(x)}"
                        f"({sum(1 for e in x if e[2])/len(x)*100:.0f}%)" if x else "—")
            print(f"  thr≤{thr:<3} h={h}  全部 {ok}/{n}({ok/n*100:.1f}%)"
                  f" 每板块/年 {n/(6*years):4.1f}   A(reso≥3) {st(A)}  B(reso2) {st(B)}  C {st(Cl)}")

    # A 级再加急跌确认
    print("\n--- A 级再加「急跌 vol≤25」---")
    for thr in (18, 20, 22, 25, 30):
        for h in (5, 7):
            evs = events(D, ix, thr, h, extra=("vol", 25))
            A = [e for e in evs if e[4] >= 3]
            B2 = [e for e in evs if e[4] >= 2]
            def st(x):
                return (f"{sum(1 for e in x if e[2])}/{len(x)}"
                        f"({sum(1 for e in x if e[2])/len(x)*100:.0f}%)" if x else "—")
            print(f"  thr≤{thr:<3} h={h}  vol≤25 全部 {st(evs)}"
                  f"   reso≥3 {st(A)}   reso≥2 {st(B2)}")


if __name__ == "__main__":
    main()
