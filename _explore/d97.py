"""d97 —— 最终方案定型：基础信号 + 分层确认层。

d96 的教训：cnl(创新低) 与 vol(高波动) **互相抵消** ——
  "创新低 + 高波动"是崩盘延续（h=7 命中只有 17%），不是反弹。
  所以基础信号里不能同时塞这两类。

改用 d95 里表现最好的基础信号（能覆盖目标日）：
  SWING = max(pos, s1_T, s1_L) ≤ 30
再在它上面做分层，看高层能不能拉到 85%。
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
CANDS = {
    "pos+s1_T+s1_L": ["pos", "s1_T", "s1_L"],
    "pos+s1_T": ["pos", "s1_T"],
    "pos+s1_T+s1_score": ["pos", "s1_T", "s1_score"],
    "z20+pos+s1_L": ["z20", "pos", "s1_L"],
}


def run(D, names, cols, thr, h, layers, cool=COOL):
    ix = [names.index(c) for c in cols]
    idx = D[C.BOARD_ORDER[0]]["names"] and D[C.BOARD_ORDER[0]]["index"]
    masks, vals = {}, {}
    for b in C.BOARD_ORDER:
        M = D[b]["cols"].to_numpy(dtype=float)[:, ix]
        masks[b] = np.all(M <= thr, axis=1) & ~np.isnan(M).any(axis=1)
        vals[b] = {k: D[b]["cols"][k].to_numpy(dtype=float) for k in layers}
    mk = pd.DataFrame({b: pd.Series(masks[b], index=D[b]["index"]) for b in C.BOARD_ORDER})
    nb = mk.reindex(idx).fillna(0).sum(axis=1).to_numpy(dtype=int)
    evs = []
    for b in C.BOARD_ORDER:
        hit = D[b][f"hit{h}"]
        last = -10 ** 9
        for i in np.flatnonzero(masks[b]):
            if i < 1 or i >= len(hit) - h or i - last < cool:
                continue
            last = i
            if not (W0 <= idx[i] <= END):
                continue
            rec = {"b": b, "d": str(idx[i].date()), "hit": bool(hit[i]), "reso": int(nb[i])}
            for k in layers:
                rec[k] = float(vals[b][k][i])
            evs.append(rec)
    return evs


def st(x):
    if not x:
        return "—"
    o = sum(1 for e in x if e["hit"])
    return f"{o}/{len(x)}({o/len(x)*100:.0f}%)"


def main():
    panels = build_full()
    wide = build_breadth()
    D = prepare(panels, wide)
    names = D[C.BOARD_ORDER[0]]["names"]
    for h in (5, 7):
        D = hits_for(D, h)

    print("科创板 SWING 分值（各候选）在目标日:")
    for tag, cols in CANDS.items():
        ix = [names.index(c) for c in cols]
        sv = pd.Series(np.nanmax(D["STAR"]["cols"].to_numpy(dtype=float)[:, ix], axis=1),
                       index=D["STAR"]["index"])
        print(f"  {tag:<20} 08-03={sv['2026-08-03']:6.1f}  09-14={sv['2026-09-14']:6.1f}")

    for h in (7, 5):
        print(f"\n########## h={h} ##########")
        for tag, cols in CANDS.items():
            for thr in (25, 30, 35):
                evs = run(D, names, cols, thr, h, ["vol", "dn1", "b5", "dn3"])
                if not evs:
                    continue
                years = (END - W0).days / 365.25
                A = [e for e in evs if e["vol"] <= 30 and e["reso"] >= 3]
                A2 = [e for e in evs if e["vol"] <= 30 and e["reso"] >= 2]
                B = [e for e in evs if e["reso"] >= 2]
                Bv = [e for e in evs if e["vol"] <= 30]
                print(f"  {tag:<20} ≤{thr:<3} 全部 {st(evs)}"
                      f" ({len(evs)/ (6*years):.1f}/板/年)"
                      f"  vol≤30&reso≥3 {st(A)}  vol≤30&reso≥2 {st(A2)}"
                      f"  reso≥2 {st(B)}  vol≤30 {st(Bv)}")


if __name__ == "__main__":
    main()
