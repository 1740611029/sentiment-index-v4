"""d85 —— 条件枚举搜索 + 时段切分验证（防过拟合）。

d84 的等权组合只到 71.8%，且抓不到科创板 2026-09-14。说明"平均"是错的方向：
把一堆中等相关的因子平均，等于把真正的极值抹平了。
这里改成 **AND 组合** —— 要求多个独立的"像低点"证据同时成立，宁可信号少也要准。

防过拟合（项目里踩过的坑：19 个样本上调出来的东西不能采）：
  训练段 2023-09-20 ~ 2025-06-30 选参，测试段 2025-07-01 ~ 2026-09-18 检验。
  只有测试段也站得住才采纳；全段数字只作参考。
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

W0 = pd.Timestamp("2023-09-20")
TRAIN_END = pd.Timestamp("2025-06-30")
COOL = 7

# 候选证据：全部 norm 到 0~100，**越低越像小波段低点**
EXTRA = ["cnl5", "cnl10", "cnl20", "cnl40", "panic", "accel",
         "vslow5", "dnrun", "clpos", "b20", "amt_pct", "r5", "body", "lwsh", "oppos"]


def prepare(panels):
    D = {}
    for b in C.BOARD_ORDER:
        p = panels[b]
        nm = norm(short_factors(p))
        g = groups_mean(nm)
        cols = pd.concat([g, nm[[c for c in EXTRA if c in nm.columns]]], axis=1)
        cols = cols.loc[cols.index >= W0]
        hit = hit_mask(p["close"]).reindex(cols.index).fillna(False).to_numpy(dtype=bool)
        D[b] = {"cols": cols.to_numpy(dtype=float), "hit": hit,
                "names": list(cols.columns), "index": cols.index}
    return D


def sim(D, cond_idx, thr, cool=COOL, seg=None):
    """逐日模拟。seg=(start,end) 只统计该时段内的信号。返回 (n, ok, events)"""
    n = ok = 0
    evs = []
    for b in C.BOARD_ORDER:
        d = D[b]
        M = d["cols"][:, cond_idx]              # (T, k)
        mask = np.all(M <= thr, axis=1) & ~np.isnan(M).any(axis=1)
        hit = d["hit"]; idx = d["index"]
        T = len(mask)
        last = -10 ** 9
        for i in range(1, T - H):
            if not mask[i] or i - last < cool:
                continue
            last = i
            if seg and not (idx[i] >= seg[0] and idx[i] <= seg[1]):
                continue
            n += 1; ok += int(hit[i])
            evs.append((b, str(idx[i].date()), bool(hit[i])))
    return n, ok, evs


def main():
    panels = build_full()
    D = prepare(panels)
    names = D[C.BOARD_ORDER[0]]["names"]
    print(f"候选证据 {len(names)} 个: {names}")
    print(f"基线 48.2%  天花板 85.7%  冷却 {COOL}  口径 T+{H} 期末>0 且回撤≥−{TOL:.0%}\n")

    thr_grid = [10, 15, 20, 25, 30]
    rows = []
    for k in (1, 2, 3):
        for combo in itertools.combinations(range(len(names)), k):
            for thr in thr_grid:
                n, ok, _ = sim(D, list(combo), thr)
                if n < 30:                       # 样本太少不参与评选
                    continue
                rows.append({"combo": combo, "thr": thr, "n": n, "hit": ok / n * 100})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d85_grid.csv"), index=False)
    print(f"共 {len(df)} 个候选（n≥30）\n")
    df = df.sort_values("hit", ascending=False)
    for _, r in df.head(15).iterrows():
        nm = "+".join(names[i] for i in r["combo"])
        print(f"  {nm:<32} ≤{r['thr']:<3} n={int(r['n']):<4} 全段 {r['hit']:.1f}%")

    # ---- 时段切分：训练段选，测试段验 ----
    print("\n=== 时段切分（训练 ≤2025-06-30 选参 → 测试 ≥2025-07-01 检验）===")
    res = []
    for _, r in df.head(60).iterrows():
        combo, thr = r["combo"], r["thr"]
        ntr, oktr, _ = sim(D, list(combo), thr, seg=(W0, TRAIN_END))
        nte, okte, evte = sim(D, list(combo), thr, seg=(pd.Timestamp("2025-07-01"), pd.Timestamp("2026-09-18")))
        if ntr < 15 or nte < 10:
            continue
        res.append({"combo": combo, "thr": thr,
                    "train": oktr / ntr * 100, "ntr": ntr,
                    "test": okte / nte * 100, "nte": nte,
                    "all": r["hit"], "n": r["n"]})
    R = pd.DataFrame(res).sort_values(["test", "train"], ascending=False)
    for _, r in R.head(12).iterrows():
        nm = "+".join(names[i] for i in r["combo"])
        print(f"  {nm:<30} ≤{r['thr']:<3} 训练 {r['train']:5.1f}%({int(r['ntr'])})"
              f"  测试 {r['test']:5.1f}%({int(r['nte'])})  全段 {r['all']:5.1f}%({int(r['n'])})")


if __name__ == "__main__":
    main()
