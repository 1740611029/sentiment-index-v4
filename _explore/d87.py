"""d87 —— 全因子搜索（指数短期因子 + 个股短期广度 + SENTI-1 分量）。

相比 d85 补了两块：
  1) 个股短期广度 dn1/dn3/b5/b10（d86）—— 项目里已验证广度是信息量最大的一类
  2) SENTI-1 的 L/T/U/score 分量 —— 虽然它是大波段模型，但 L（广度情绪）
     的低位对小波段反弹同样可能有效，值得一并检验

模拟加速：冷却期的贪心选择只在 mask=True 的位置上迭代，
不再对每个交易日循环（快 ~50 倍，才能搜得动 k=3）。
"""
from __future__ import annotations
import os, sys, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C
from d80 import build_full, hit_mask, H, TOL
from d81 import short_factors, norm
from d84 import groups_mean
from d86 import build as build_breadth, board_breadth

W0 = pd.Timestamp("2023-09-20")
TRAIN_END = pd.Timestamp("2025-06-30")
TEST0 = pd.Timestamp("2025-07-01")
COOL = 7
THR_GRID = [8, 10, 12, 15, 20, 25, 30]

EX_IDX = ["cnl5", "cnl10", "cnl20", "cnl40", "panic", "accel",
          "vslow5", "dnrun", "clpos", "b20", "amt_pct", "r5", "body", "lwsh", "oppos"]


def prepare(panels, wide):
    D = {}
    for b in C.BOARD_ORDER:
        p = panels[b]
        nm = norm(short_factors(p))
        g = groups_mean(nm)
        bb = board_breadth(b, wide).reindex(p.index)
        bb_nm = norm(bb, extra_inv=("dn1", "dn3"))
        s1 = p[["L", "T", "U", "score"]].add_prefix("s1_")
        cols = pd.concat([g, nm[EX_IDX], bb_nm, s1], axis=1).loc[p.index >= W0]
        hit = hit_mask(p["close"]).reindex(cols.index).fillna(False).to_numpy(dtype=bool)
        D[b] = {"cols": cols.to_numpy(dtype=float), "hit": hit,
                "names": list(cols.columns), "index": cols.index}
    return D


def sim(D, cond_idx, thr, cool=COOL, seg=None):
    n = ok = 0
    evs = []
    for b in C.BOARD_ORDER:
        d = D[b]
        M = d["cols"][:, cond_idx]
        valid = ~np.isnan(M).any(axis=1)
        mask = np.all(M <= thr, axis=1) & valid
        hit, idx = d["hit"], d["index"]
        last = -10 ** 9
        for i in np.flatnonzero(mask):
            if i < 1 or i >= len(hit) - H or i - last < cool:
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
    print(f"候选 {len(names)} 个: {names}\n")

    rows = []
    for k in (1, 2, 3):
        for combo in itertools.combinations(range(len(names)), k):
            for thr in THR_GRID:
                n, ok, _ = sim(D, list(combo), thr)
                if n < 30:
                    continue
                rows.append({"combo": combo, "thr": thr, "n": n, "hit": ok / n * 100})
    df = pd.DataFrame(rows)
    df["nm"] = df["combo"].apply(lambda c: "+".join(names[i] for i in c))
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d87_grid.csv"), index=False)
    print(f"候选 {len(df)} 个（n≥30）\n--- 全段 top20 ---")
    for _, r in df.sort_values("hit", ascending=False).head(20).iterrows():
        print(f"  {r['nm']:<34} ≤{r['thr']:<3} n={int(r['n']):<4} {r['hit']:.1f}%")

    print("\n--- 时段切分：训练 ≤2025-06-30 → 测试 ≥2025-07-01 ---")
    res = []
    for _, r in df.sort_values("hit", ascending=False).head(120).iterrows():
        ntr, oktr, _ = sim(D, list(r["combo"]), r["thr"], seg=(W0, TRAIN_END))
        nte, okte, _ = sim(D, list(r["combo"]), r["thr"], seg=(TEST0, pd.Timestamp("2026-09-18")))
        if ntr < 15 or nte < 12:
            continue
        res.append({"nm": r["nm"], "thr": r["thr"], "n": r["n"], "all": r["hit"],
                    "train": oktr / ntr * 100, "ntr": ntr,
                    "test": okte / nte * 100, "nte": nte})
    R = pd.DataFrame(res).sort_values("test", ascending=False)
    for _, r in R.head(15).iterrows():
        print(f"  {r['nm']:<30} ≤{r['thr']:<3} 训练 {r['train']:5.1f}%({int(r['ntr'])})"
              f"  测试 {r['test']:5.1f}%({int(r['nte'])})  全段 {r['all']:5.1f}%({int(r['n'])})")


if __name__ == "__main__":
    main()
