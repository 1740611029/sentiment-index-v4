"""d92 —— 决定性一轮：AND(max) 分值 × 共振 × 多持有期，大样本 + 稳健性。

结论先行（前面几轮的实测）：
  · 加权平均（65.2%）不如 AND/max（75.0%）—— 平均会把极值抹平，
    而低点恰恰要求"每个维度都到位"。
  · 共振（同一天别的板块也超卖）是最有效的确认层，能 +5~10 个点。
  · 命中率-样本量前沿是硬的：想要更多信号就得接受更低命中率。

分值定义：SWING = max(各分量)，信号 = SWING ≤ 阈值。
  （AND 与 max 完全等价，但 max 能画成连续曲线，用户要的是指标线不是开关）
"""
from __future__ import annotations
import os, sys, itertools, json
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
COOL = 7
THR_GRID = [8, 10, 12, 15, 20, 25, 30]


def prepare(panels, wide):
    D = {}
    for b in C.BOARD_ORDER:
        p = panels[b]
        nm = norm(short_factors(p))
        g = groups_mean(nm)
        zn = norm(z_factors(p))
        bb = board_breadth(b, wide).reindex(p.index)
        bb_nm = norm(bb, extra_inv=("dn1", "dn3"))
        s1 = p[["L", "T", "U", "score"]].add_prefix("s1_")
        cols = pd.concat([g, nm[EX_IDX], zn, bb_nm, s1], axis=1)
        D[b] = {"cols": cols, "names": list(cols.columns), "index": cols.index,
                "close": p["close"].reindex(cols.index)}
    return D


def hits_for(D, h):
    for b in C.BOARD_ORDER:
        st = fwd_stats(D[b]["close"], h)
        D[b][f"hit{h}"] = ((st["end"] > 0) & (st["mdd"] >= -0.03)).to_numpy(dtype=bool)
    return D


def sim(D, cond_idx, thr, h, reso_min=1, cool=COOL, seg=None):
    idx = D[C.BOARD_ORDER[0]]["index"]
    masks = {}
    for b in C.BOARD_ORDER:
        M = D[b]["cols"].to_numpy(dtype=float)[:, cond_idx]
        masks[b] = np.all(M <= thr, axis=1) & ~np.isnan(M).any(axis=1)
    mk = pd.DataFrame({b: pd.Series(masks[b], index=D[b]["index"]) for b in C.BOARD_ORDER})
    cnt_all = mk.fillna(0).sum(axis=1)          # 按日期对齐后的共振数（公共日期轴）
    # 各板块历史长度不同（科创板 2019-10 才起），共振数必须按**该板块自己的日期轴**取，
    # 否则 i 会错位到别的日期上，科创板的共振数会被读成 0、信号全被过滤掉。
    nb_by = {b: cnt_all.reindex(D[b]["index"]).fillna(0).to_numpy(dtype=int)
             for b in C.BOARD_ORDER}
    n = ok = 0
    evs = []
    for b in C.BOARD_ORDER:
        hit = D[b][f"hit{h}"]
        nb = nb_by[b]
        last = -10 ** 9
        for i in np.flatnonzero(masks[b]):
            if i < 1 or i >= len(hit) - h or i - last < cool:
                continue
            if nb[i] < reso_min:
                continue
            last = i
            if seg and not (seg[0] <= idx[i] <= seg[1]):
                continue
            n += 1; ok += int(hit[i])
            evs.append((b, str(idx[i].date()), bool(hit[i])))
    return n, ok, evs


def main():
    panels = build_full()
    wide = build_breadth()
    D = prepare(panels, wide)
    names = D[C.BOARD_ORDER[0]]["names"]
    for h in (5, 7):
        D = hits_for(D, h)
        allhit = pd.concat([pd.Series(D[b][f"hit{h}"], index=D[b]["index"])
                            for b in C.BOARD_ORDER])
        base = allhit[allhit.index >= W0].mean() * 100
        print(f"\n########## h={h}　窗口 {W0.date()} 起　基线 {base:.1f}% ##########")
        rows = []
        for k in (1, 2, 3):
            for combo in itertools.combinations(range(len(names)), k):
                ci = list(combo)
                for thr in THR_GRID:
                    for rm in (1, 2, 3):
                        n, ok, _ = sim(D, ci, thr, h, reso_min=rm,
                                       seg=(W0, pd.Timestamp("2026-09-18")))
                        if n < 25:
                            continue
                        rows.append({"combo": ci, "thr": thr, "reso": rm,
                                     "n": n, "hit": ok / n * 100})
        df = pd.DataFrame(rows)
        df["nm"] = df["combo"].apply(lambda c: "+".join(names[i] for i in c))
        df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"d92_h{h}.csv"), index=False)
        print(f"候选 {len(df)} 个（n≥25）")
        for lo, hi in [(25, 40), (40, 70), (70, 110), (110, 200), (200, 10 ** 9)]:
            sub = df[(df.n >= lo) & (df.n < hi)]
            if sub.empty:
                continue
            r = sub.loc[sub["hit"].idxmax()]
            t5 = sub.sort_values("hit", ascending=False).head(5)
            print(f"  n∈[{lo},{hi})  最佳 {r['hit']:.1f}%  n={int(r['n']):<4}"
                  f" top5均值 {t5['hit'].mean():.1f}%  {r['nm']} ≤{r['thr']} reso≥{r['reso']}")
        print("  --- top10 ---")
        for _, r in df.sort_values("hit", ascending=False).head(10).iterrows():
            print(f"    {r['nm']:<26} ≤{r['thr']:<3} reso≥{r['reso']}  "
                  f"{r['hit']:5.1f}%  n={int(r['n'])}")


if __name__ == "__main__":
    main()
