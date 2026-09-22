"""d90 —— 扩大样本 + 加共振层，做一次能站得住的检验。

前面几轮的问题：
  1) 只有 726 天 × 6 板块，挑出来的 top 全是 n≈30 的小样本，
     训练/测试能差 35 个百分点（65% vs 100%），明显是噪声。
  2) 只看了近 3 年，而这 3 年是牛市。
这里把统计窗口扩到 **2021-01 起**（面板 2019 起，锚点 750 天预热到 2021 才算齐），
含 2021-2022 熊市，样本约 8280 个，再挑就挑不出纯噪声了。

共振层：SENTI-1 里已验证过最强的确认层（reso≥2 时 T+60 盈利 26/26）。
  含义是"同一天有几个板块也超卖"——系统性超卖 vs 单个板块自己的问题。
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
from d89 import z_factors, Z_COLS, EX_IDX

W0 = pd.Timestamp("2021-01-01")          # 锚点预热完成后的起点
H = 7
COOL = 7
THR_GRID = [8, 10, 12, 15, 20, 25, 30]


def prepare(panels, wide, h=H):
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
        st = fwd_stats(p["close"], h)
        hit = ((st["end"] > 0) & (st["mdd"] >= -0.03)).reindex(cols.index).fillna(False)
        D[b] = {"cols": cols, "hit": hit.to_numpy(dtype=bool),
                "names": list(cols.columns), "index": cols.index}
    return D


def raw_mask(D, cond_idx, thr):
    """每个板块每日是否满足条件（不看冷却）。"""
    out = {}
    for b in C.BOARD_ORDER:
        d = D[b]
        M = d["cols"].to_numpy(dtype=float)[:, cond_idx]
        out[b] = np.all(M <= thr, axis=1) & ~np.isnan(M).any(axis=1)
    return out


def sim(D, cond_idx, thr, reso_min=1, cool=COOL, seg=None, h=H):
    """逐日模拟，可加共振过滤：当日至少 reso_min 个板块同时满足条件。"""
    masks = raw_mask(D, cond_idx, thr)
    idx = D[C.BOARD_ORDER[0]]["index"]
    # 各板块指数历史长度不同（科创板 2019-10 才有），必须按日期对齐后再求和
    mk = pd.DataFrame({b: pd.Series(masks[b], index=D[b]["index"]) for b in C.BOARD_ORDER})
    nboard = mk.reindex(idx).fillna(0).sum(axis=1).to_numpy(dtype=int)
    n = ok = 0
    evs = []
    for b in C.BOARD_ORDER:
        d = D[b]
        hit = d["hit"]
        last = -10 ** 9
        for i in np.flatnonzero(masks[b]):
            if i < 1 or i >= len(hit) - h or i - last < cool:
                continue
            if nboard[i] < reso_min:
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
    allhit = pd.concat([pd.Series(d["hit"], index=d["index"]) for d in D.values()])
    base = allhit[allhit.index >= W0].mean() * 100
    print(f"窗口 {W0.date()} 起　h={H}　基线 {base:.1f}%")
    print(f"候选 {len(names)} 个\n")

    def w0_mask():
        return D[C.BOARD_ORDER[0]]["index"] >= W0

    rows = []
    for k in (1, 2, 3):
        for combo in itertools.combinations(range(len(names)), k):
            for thr in THR_GRID:
                n, ok, _ = sim(D, list(combo), thr, seg=(W0, pd.Timestamp("2026-09-18")))
                if n < 40:
                    continue
                rows.append({"combo": combo, "thr": thr, "n": n, "hit": ok / n * 100})
    df = pd.DataFrame(rows)
    df["nm"] = df["combo"].apply(lambda c: "+".join(names[i] for i in c))
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d90_grid.csv"), index=False)
    print(f"候选 {len(df)} 个（n≥40）\n--- 前沿 ---")
    for lo, hi in [(40, 70), (70, 110), (110, 180), (180, 300), (300, 10 ** 9)]:
        sub = df[(df.n >= lo) & (df.n < hi)]
        if sub.empty:
            continue
        r = sub.loc[sub["hit"].idxmax()]
        t5 = sub.sort_values("hit", ascending=False).head(5)
        print(f"  n∈[{lo},{hi})  最佳 {r['hit']:.1f}%  n={int(r['n']):<4}"
              f" top5均值 {t5['hit'].mean():.1f}%  {r['nm']} ≤{r['thr']}")

    print("\n--- 共振层叠加（对 top 候选加 reso≥2 / ≥3）---")
    for _, r in df.sort_values("hit", ascending=False).head(25).iterrows():
        line = f"  {r['nm']:<26} ≤{r['thr']:<3} 无过滤 {r['hit']:5.1f}%({int(r['n'])})"
        for rm in (2, 3):
            n2, ok2, _ = sim(D, list(r["combo"]), r["thr"], reso_min=rm,
                             seg=(W0, pd.Timestamp("2026-09-18")))
            if n2 >= 15:
                line += f"   reso≥{rm} {ok2/n2*100:5.1f}%({n2})"
        print(line)


if __name__ == "__main__":
    main()
