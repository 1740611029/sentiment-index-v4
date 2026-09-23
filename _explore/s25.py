"""s25 —— 第十七轮：市场级簇的「形状」——一次性投降 vs 陆续阴跌

s24 的 D 段出现一个很强的结构（并集，市场级，间隔 7 交易日）：
  簇内 1 个信号（单板块局部危机）  11 件 / 45.5%
  簇内 2~5 个信号                  8 件 / 87.5%
  簇内 6+ 个信号（系统性下跌）      9 件 / **22.2%**
"全市场一起恐慌"反而最差 —— 与页面既有的**同日共振**分档（S 级 87%）方向相反。

两者其实是不同的东西：
  · 页面 reso = **同一天**有几个板块出信号（横截面集中度）
  · 簇规模   = **±5 日内**累计有几个信号（含跨日陆续触发）
一个 6+ 簇可能是「6 个板块同一天一起投降」，也可能是「2 个板块第 1 天、
2 个第 3 天、2 个第 5 天」—— 后者是**陆续阴跌**。

假说 H：**同日集中投降（好）vs 跨日陆续阴跌（差）**。
这和 SENTI-1 的老结论「磨底不是底」（深度回撤 + 高度同步 = 还在慢慢跌）是同一个道理。

本脚本
A. 6 板块口径：把簇按「同日最大板块数 same_max」和「跨度 span」两个维度切，看命中率
B. 9 宽基长历史（2013-2026）复验同一结构 —— 这是本项目唯一的历史否决关
C. 参数稳健性：gap / 间隔 / 分界点邻域扰动
D. 结论：能否做成一个「形状标签」（注意：标签不做过滤，与 reso 同一个道理）
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
from s8 import wilson, W_ALL, W_57, W_3, BOARDS  # noqa: E402


def shape(cl):
    """一个市场级簇的形状指标。"""
    evs = cl[2]
    ds = [pd.Timestamp(e["date"]) for _, e in evs]
    by_day = {}
    for b, e in evs:
        by_day.setdefault(e["date"], set()).add(b)
    return {
        "n_sig": len(evs),
        "n_boards": len({b for b, _ in evs}),
        "same_max": max(len(v) for v in by_day.values()),
        "span": (max(ds) - min(ds)).days,
        "ndays": len(by_day),
    }


def bucketed(clusters, key, edges, labels, tag):
    print(f"  ── 按 {tag}")
    print(f"  {'分组':<20}{'件数':>7}{'命中':>9}{'下界':>8}")
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        part = [c for c in clusters if lo <= c[3][key] < hi]
        if not part:
            continue
        n_, k_, r_, w_ = s24.cstat([c[1] for c in part])
        print(f"  {lab:<20}{n_:>7}{r_:>8.1f}%{w_:>7.1f}%")


def prep(clusters):
    return [c + (shape(c),) for c in clusters]


def main():
    from senti import config as C
    from senti import store, swing, swing2, swing3

    sp, s2p, s3p = store.swing_panels(), store.swing2_panels(), store.swing3_panels()
    sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
    panels = store.load()
    cal = list(panels["SH"].index)
    u_ev = [(b, e) for b in C.BOARD_ORDER
            for e in store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r)]
    cl = prep(s24.space_out(s23.cluster_market(u_ev), cal, 7))

    print("=" * 104)
    print("A. 6 板块口径（并集，市场级，间隔 7 交易日）")
    print("=" * 104)
    print(f"  随机基线 {float(np.mean([swing2.baseline(s2p[b]) for b in C.BOARD_ORDER])):.1f}%"
          f"　样本 {len(cl)} 件\n")
    bucketed(cl, "same_max", [1, 2, 3, 5, 99],
             ["1 个板块", "2 个板块", "3~4 个板块", "5+ 个板块"], "同日最大板块数 same_max")
    print()
    bucketed(cl, "span", [0, 1, 3, 8, 999],
             ["0 天（同一天）", "1~2 天", "3~7 天", "8 天以上"], "簇跨度 span")
    print()
    bucketed(cl, "ndays", [1, 2, 3, 99],
             ["1 天", "2 天", "3 天以上"], "出信号的天数 ndays")

    print("\n" + "=" * 104)
    print("B. 9 宽基长历史复验（2013-2026，三模型代理并集）")
    print("=" * 104)
    E = s8.Evaluator()
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])
    lev = [(b, {"date": dt, "ok": ok}) for b, d in inc.items() for dt, (ok, _) in d.items()]
    # 用各宽基自己的交易日轴做间隔判定：这里用大盘
    cal9 = list(E.CL["SH"].index)
    for gap, gap_td in ((5, 7), (5, 0), (7, 7)):
        clm = prep(s24.space_out(s23.cluster_market(lev, gap), cal9, gap_td))
        n_, k_, r_, w_ = s24.cstat([c[1] for c in clm])
        print(f"  gap={gap} 间隔={gap_td} 交易日：{n_} 件 / {r_:.1f}%（下界 {w_:.1f}%）")
        bucketed(clm, "span", [0, 1, 3, 8, 999],
                 ["0 天（同一天）", "1~2 天", "3~7 天", "8 天以上"], "簇跨度 span")
        print()

    print("=" * 104)
    print("C. 稳健性：分界点邻域（6 板块，跨度 0~2 天 vs 3+ 天）")
    print("=" * 104)
    print(f"  {'gap':>4}{'间隔':>5}{'紧凑(≤2天)':>22}{'分散(3天+)':>22}")
    for gap in (3, 5, 7, 10):
        for gap_td in (0, 5, 7, 10):
            c2 = prep(s24.space_out(s23.cluster_market(u_ev, gap), cal, gap_td))
            a = [c for c in c2 if c[3]["span"] <= 2]
            b_ = [c for c in c2 if c[3]["span"] >= 3]
            na, ka, ra, wa = s24.cstat([c[1] for c in a])
            nb, kb, rb, wb = s24.cstat([c[1] for c in b_])
            print(f"  {gap:>4}{gap_td:>5}"
                  f"{f'{ka}/{na} = {ra:.0f}%':>22}{f'{kb}/{nb} = {rb:.0f}%':>22}")

    print("\n" + "=" * 104)
    print("D. 逐笔明细（6 板块，gap=5 间隔=7）")
    print("=" * 104)
    print(f"  {'首枪日':<13}{'板块':<9}{'信号':>5}{'板块数':>7}{'同日最大':>9}{'跨度':>6}"
          f"{'天数':>6}{'命中':>7}")
    for c in cl:
        sh = c[3]
        print(f"  {c[1]['date']:<13}{C.BOARDS[c[0]]['name']:<9}{sh['n_sig']:>5}"
              f"{sh['n_boards']:>7}{sh['same_max']:>9}{sh['span']:>6}{sh['ndays']:>6}"
              f"{'✔' if c[1]['ok'] else '✘':>7}")


if __name__ == "__main__":
    main()
