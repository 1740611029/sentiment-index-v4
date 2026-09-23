"""s27 —— 第十九轮：第一枪当下可得的特征，能否提前识别「会长成长期阴跌」

现状（s23~s26）
------------
· 市场级口径（跨板块 ±5 日聚类）：并集 34 件 / 50.0%；SENTI-1 6 件 / 83.3%
· 簇形态：短簇（span ≤7 天）长历史 68~77%，长簇（span ≥8 天）19~36%
  —— 8 个参数组合（gap 3/5/7/10 × 间隔 0/7）**全部一致**，两样本都成立，非常稳。
· 但 s26 证明它**不可当过滤器**：
  - 长簇的失败发生在**第一枪**（14~22%），而 span 在第一枪时不可知；
  - 所有实时替代（跨板块新鲜度 N=1~20、单板块冷却）都让结果变差。
  → 目前只能当「事后诊断标签」。

本轮把这条路走到底：既然「事后 span」不可用，就找**第一枪当下就能算**的代理量。
候选（全部只用当日及历史，无未来函数）：
  1. reso        第一枪当天有几个板块出信号（横截面集中度）
  2. score       第一枪的分值本身（有多极端）
  3. dd20        距 20 日高点的回撤深度
  4. dd60        距 60 日高点的回撤深度
  5. vpct        20 日已实现波动率的因果锚分位（是不是高波动环境）
  6. dsince_hi20 距最近一次 20 日新高的交易日数（已经跌了多久）
  7. dsince_hi60 距最近一次 60 日新高的交易日数

判据（与 s17 同一套纪律）
  ① 该特征在中位数两侧的「长簇比例」要有明显差异；
  ② 用它做过滤后，并集命中率要**两个样本都提升**；
  ③ 参数邻域要平坦；
  ④ 过了才谈得上收编，否则如实记为「不可实时识别」。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s8
import s9                                        # noqa: F401
import s10                                       # noqa: F401
import s12
import s17
import s18
import s19
import s23
import s24
import s25
from s8 import wilson, W_ALL, BOARDS             # noqa: E402


def feats(df, i, s):
    """第一枪当下可得的特征。df 该板块行情，i 为位置，s 为该板块分值序列。"""
    c = df["close"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    if i < 65:
        return None
    d20 = c[i] / hi[max(0, i - 20):i + 1].max() - 1
    d60 = c[i] / hi[max(0, i - 60):i + 1].max() - 1
    r = np.diff(c[i - 20:i + 1]) / c[i - 20:i]
    v20 = r.std()
    j20 = None
    for j in range(i, max(0, i - 60), -1):
        if hi[j] >= hi[max(0, j - 20):j + 1].max():
            j20 = i - j
            break
    j60 = None
    for j in range(i, max(0, i - 120), -1):
        if hi[j] >= hi[max(0, j - 60):j + 1].max():
            j60 = i - j
            break
    return {"dd20": d20, "dd60": d60, "v20": v20,
            "dsince_hi20": j20 if j20 is not None else 60,
            "dsince_hi60": j60 if j60 is not None else 120}


def main():
    from senti import config as C
    from senti import store, swing, swing2, swing3

    sp, s2p, s3p = store.swing_panels(), store.swing2_panels(), store.swing3_panels()
    sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
    panels = store.load()
    cal = list(panels["SH"].index)

    u_ev = [(b, e) for b in C.BOARD_ORDER
            for e in store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r)]
    cl = s25.prep(s24.space_out(s23.cluster_market(u_ev, 5), cal, 7))

    print("=" * 100)
    print("A. 第一枪的实时特征（6 板块，市场级 28 件）")
    print("=" * 100)
    rows = []
    for c in cl:
        b, e = c[0], c[1]
        p = panels[b]
        i = p.index.get_loc(pd.Timestamp(e["date"]))
        f = feats(p, i, None)
        if f is None:
            continue
        rows.append({"b": b, "date": e["date"], "ok": e["ok"],
                     "span": c[3]["span"], "long": c[3]["span"] >= 8,
                     "n_sig": c[3]["n_sig"], "same_max": c[3]["same_max"], **f})
    R = pd.DataFrame(rows)
    print(f"  可用样本 {len(R)} 件　长簇（span≥8）{int(R['long'].sum())} 件"
          f"　命中 {int(R['ok'].sum())}/{len(R)} = {100*R['ok'].mean():.0f}%\n")
    print(f"  {'特征':<14}{'低半长簇率':>11}{'高半长簇率':>11}{'低半命中':>10}{'高半命中':>10}"
          f"{'低半件数':>9}{'高半件数':>9}")
    for f in ("dd20", "dd60", "v20", "dsince_hi20", "dsince_hi60", "same_max", "n_sig"):
        med = R[f].median()
        lo, hi = R[R[f] <= med], R[R[f] > med]
        if len(lo) == 0 or len(hi) == 0:
            continue
        print(f"  {f:<14}{100*lo['long'].mean():>10.0f}%{100*hi['long'].mean():>10.0f}%"
              f"{100*lo['ok'].mean():>9.0f}%{100*hi['ok'].mean():>9.0f}%"
              f"{len(lo):>9}{len(hi):>9}")

    print("\n" + "=" * 100)
    print("B. 用「第一枪特征」过滤并集（实时版）—— 两个样本对照")
    print("=" * 100)
    # 6 板块：对并集的**每个信号**（不是簇）应用特征过滤
    print(f"  {'过滤规则':<26}{'6板块件数':>10}{'6板块命中':>10}{'9宽基件数':>10}{'9宽基命中':>10}")

    E = s8.Evaluator()
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])
    lev = [(b, {"date": dt, "ok": ok}) for b, d in inc.items()
           for dt, (ok, _) in d.items()]

    def filt6(rule):
        out = []
        for b, e in u_ev:
            p = panels[b]
            i = p.index.get_loc(pd.Timestamp(e["date"]))
            f = feats(p, i, None)
            if f is None:
                continue
            if rule(f):
                out.append((b, e))
        return out

    def filt9(rule):
        out = []
        for b, e in lev:
            df = E.D[b]
            i = df.index.get_loc(pd.Timestamp(e["date"]))
            f = feats(df, i, None)
            if f is None:
                continue
            if rule(f):
                out.append((b, e))
        return out

    n0, k0, r0, _ = s24.cstat([e for _, e in u_ev])
    n90, k90, r90, _ = s24.cstat([e for _, e in lev])
    print(f"  {'无过滤':<26}{n0:>10}{r0:>9.1f}%{n90:>10}{r90:>9.1f}%")
    rules = {
        "dd60 ≥ −20%（跌得浅）": lambda f: f["dd60"] >= -0.20,
        "dd60 ≥ −15%": lambda f: f["dd60"] >= -0.15,
        "dd60 ≤ −20%（跌得深）": lambda f: f["dd60"] <= -0.20,
        "dsince_hi20 ≥ 30 天": lambda f: f["dsince_hi20"] >= 30,
        "dsince_hi20 ≤ 15 天": lambda f: f["dsince_hi20"] <= 15,
        "v20 低半（波动小）": lambda f: f["v20"] <= 0.018,
        "dd20 ≤ −10%（短期超跌）": lambda f: f["dd20"] <= -0.10,
    }
    for tag, rule in rules.items():
        a = filt6(rule)
        b_ = filt9(rule)
        na, ka, ra, _ = s24.cstat([e for _, e in a])
        nb, kb, rb, _ = s24.cstat([e for _, e in b_])
        print(f"  {tag:<26}{na:>10}{ra:>8.1f}%{nb:>10}{rb:>8.1f}%")

    print("\n" + "=" * 100)
    print("C. 结论")
    print("=" * 100)
    print("  见上方：若没有任何规则的「两个样本同时提升」，则长簇问题**无法实时识别**，")
    print("  只能作为事后诊断标签写进文档，不进页面、不做过滤。")


if __name__ == "__main__":
    main()
