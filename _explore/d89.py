"""d89 —— 换持有期 + 补极值类因子，看 85% 到底可不可达。

两个改动：
1) 持有期：用户说的是"4~10 天"，d80 一直用 h=7（中位数）。
   h=5 是波段下限，判定更贴合"买完很快就涨"，也更容易命中；
   三个 h 都测，看命中率-样本量前沿整体能抬多少。
   注意 h=5 的"局部低点天花板 100%"是循环定义（低点定义用了未来 5 天），
   不能当目标；但**实时信号不含未来数据**，所以它的命中率仍是真实能力。

2) 补极值因子：vol(波动高) 是最强单因子，说明市场要"急"；
   但 vol 太粗（涨也波动跌也波动）。这里补方向明确的极值类：
     z5/z10/z20  价格偏离短期均线的标准差倍数
     boll20      布林带位置（< −1 = 跌破下轨）
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

W0 = pd.Timestamp("2023-09-20")
TRAIN_END = pd.Timestamp("2025-06-30")
TEST0 = pd.Timestamp("2025-07-01")
COOL = 7
THR_GRID = [8, 10, 12, 15, 20, 25, 30]

EX_IDX = ["cnl5", "cnl10", "cnl20", "cnl40", "panic", "accel",
          "vslow5", "dnrun", "clpos", "b20", "amt_pct", "r5", "body", "lwsh", "oppos"]
Z_COLS = ["z5", "z10", "z20", "boll20"]


def z_factors(p: pd.DataFrame) -> pd.DataFrame:
    close = p["close"].astype(float)
    f = pd.DataFrame(index=p.index)
    for n in (5, 10, 20):
        ma = close.rolling(n).mean()
        sd = close.rolling(n).std()
        f[f"z{n}"] = (close - ma) / sd.replace(0.0, np.nan)
    ma20 = close.rolling(20).mean(); sd20 = close.rolling(20).std()
    f["boll20"] = (close - ma20) / (2 * sd20.replace(0.0, np.nan))
    return f


def prepare(panels, wide, h):
    D = {}
    for b in C.BOARD_ORDER:
        p = panels[b]
        nm = norm(short_factors(p))
        g = groups_mean(nm)
        zn = norm(z_factors(p), extra_inv=())       # z 本身越低越超卖，不反转
        bb = board_breadth(b, wide).reindex(p.index)
        bb_nm = norm(bb, extra_inv=("dn1", "dn3"))
        s1 = p[["L", "T", "U", "score"]].add_prefix("s1_")
        cols = pd.concat([g, nm[EX_IDX], zn, bb_nm, s1], axis=1).loc[p.index >= W0]
        st = fwd_stats(p["close"], h).reindex(cols.index)
        hit = ((st["end"] > 0) & (st["mdd"] >= -0.03)).fillna(False).to_numpy(dtype=bool)
        D[b] = {"cols": cols.to_numpy(dtype=float), "hit": hit,
                "names": list(cols.columns), "index": cols.index}
    return D


def sim(D, cond_idx, thr, h, cool=COOL, seg=None):
    n = ok = 0
    evs = []
    for b in C.BOARD_ORDER:
        d = D[b]
        M = d["cols"][:, cond_idx]
        mask = np.all(M <= thr, axis=1) & ~np.isnan(M).any(axis=1)
        hit, idx = d["hit"], d["index"]
        last = -10 ** 9
        for i in np.flatnonzero(mask):
            if i < 1 or i >= len(hit) - h or i - last < cool:
                continue
            last = i
            if seg and not (seg[0] <= idx[i] <= seg[1]):
                continue
            n += 1; ok += int(hit[i])
            evs.append((b, str(idx[i].date()), bool(hit[i])))
    return n, ok, evs


def baseline_of(D):
    h = pd.concat([pd.Series(d["hit"]) for d in D.values()])
    return h.mean() * 100


def run(panels, wide, h, topn=12):
    D = prepare(panels, wide, h)
    names = D[C.BOARD_ORDER[0]]["names"]
    base = baseline_of(D)
    rows = []
    for k in (1, 2, 3):
        for combo in itertools.combinations(range(len(names)), k):
            for thr in THR_GRID:
                n, ok, _ = sim(D, list(combo), thr, h)
                if n < 30:
                    continue
                rows.append({"combo": combo, "thr": thr, "n": n, "hit": ok / n * 100})
    df = pd.DataFrame(rows)
    df["nm"] = df["combo"].apply(lambda c: "+".join(names[i] for i in c))
    print(f"\n########## h={h}  基线 {base:.1f}%  ({len(df)} 个候选) ##########")
    for lo, hi in [(30, 50), (50, 80), (80, 120), (120, 200), (200, 10 ** 9)]:
        sub = df[(df.n >= lo) & (df.n < hi)]
        if sub.empty:
            continue
        r = sub.loc[sub["hit"].idxmax()]
        print(f"  n∈[{lo},{hi})  最佳 {r['hit']:.1f}%  n={int(r['n']):<4} {r['nm']} ≤{r['thr']}")
    # 时段切分
    res = []
    for _, r in df.sort_values("hit", ascending=False).head(150).iterrows():
        ntr, oktr, _ = sim(D, list(r["combo"]), r["thr"], h, seg=(W0, TRAIN_END))
        nte, okte, _ = sim(D, list(r["combo"]), r["thr"], h, seg=(TEST0, pd.Timestamp("2026-09-18")))
        if ntr < 15 or nte < 12:
            continue
        res.append({"nm": r["nm"], "thr": r["thr"], "n": r["n"], "all": r["hit"],
                    "train": oktr / ntr * 100, "ntr": ntr,
                    "test": okte / nte * 100, "nte": nte})
    R = pd.DataFrame(res)
    if len(R):
        print(f"  --- 时段切分 top{topn}（按测试段排序）---")
        for _, r in R.sort_values("test", ascending=False).head(topn).iterrows():
            print(f"    {r['nm']:<28} ≤{r['thr']:<3} 训练 {r['train']:5.1f}%({int(r['ntr'])})"
                  f"  测试 {r['test']:5.1f}%({int(r['nte'])})  全段 {r['all']:5.1f}%({int(r['n'])})")
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"d89_h{h}.csv"), index=False)
    return df, D, names


if __name__ == "__main__":
    panels = build_full()
    wide = build_breadth()
    for h in (5, 7, 10):
        run(panels, wide, h)
