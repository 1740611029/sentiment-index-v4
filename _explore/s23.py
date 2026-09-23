"""s23 —— 第十五轮：市场级聚类（跨板块）与 SENTI-1 的事件级口径

上一轮做到「同板块 ±4 自然日算一件事」，但 136 件事里仍有相当一部分是
**同一波下跌里不同板块各自的低点**（6 个板块高度相关，同一天或隔天一起出信号）。
所以还要往上做一层：**任意板块、±5 自然日内的信号 = 一次市场级事件**。

这一层会暴露两个数字的差别：
  · 事件级（同板块）136 件 —— 「这个板块自己的入场机会」
  · 市场级    ?  件 —— 「全市场一共几波下跌值得动手」
后者才是「这套系统一年用几次」的答案。

同时补 SENTI-1：AGENTS.md 第 5 节早就写着「底部命中率按 78% 说，因为近 3 年
17 次信号高度重合、实际只对应 5 次市场级事件」，但这个数字一直只在文档里，
没进代码。本轮把它算出来并落进 summary()。

风险（先说清，防自欺）
------------------
单链聚类（chaining）会把连续阴跌串成一串超长簇 —— 2024-01 那种连跌 20 天的行情
可能被并成「一次事件」。所以必须：
  ① 报簇的时长分布，看有没有异常长簇；
  ② 给簇加最大跨度上限（span_cap），超过就强制断开；
  ③ 对 span_cap 做敏感性扫描（不是挑一个好看的数）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s19
from s8 import wilson                                # noqa: E402


def cluster_market(events, gap_days=5, span_cap=20):
    """跨板块聚类：任意板块、与簇内前一个信号相隔 ≤gap_days 自然日 → 同一次事件。

    events: [(board, event_dict), ...]
    返回 [(first_board, first_event, cluster_list), ...]
    """
    ev = sorted(events, key=lambda x: (x[1]["date"], x[0]))
    out = []
    cur = []
    last_d = None
    start_d = None
    for b, e in ev:
        d = pd.Timestamp(e["date"])
        if not cur:
            cur = [(b, e)]
            last_d = start_d = d
            continue
        if (d - last_d).days <= gap_days and (d - start_d).days <= span_cap:
            cur.append((b, e))
            last_d = d
        else:
            out.append(cur)
            cur = [(b, e)]
            last_d = start_d = d
    if cur:
        out.append(cur)
    return [(c[0][0], c[0][1], c) for c in out]


def cstat(evs):
    done = [e for e in evs if e.get("ok") is not None]
    if not done:
        return 0, 0, float("nan"), float("nan")
    k = sum(1 for e in done if e["ok"])
    return len(done), k, 100 * k / len(done), wilson(k, len(done))


def main():
    from senti import config as C
    from senti import store, swing, swing2, swing3

    sp, s2p, s3p = store.swing_panels(), store.swing2_panels(), store.swing3_panels()
    sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
    panels = store.load()
    reso = store.resonance(panels)
    bna = store.bna_series()

    u_ev = [(b, e) for b in C.BOARD_ORDER
            for e in store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r)]
    b_ev = [(b, e) for b in C.BOARD_ORDER
            for e in store.events(panels[b], reso, bna)]

    print("=" * 108)
    print("A. 小波段三模型并集：三层口径对照")
    print("=" * 108)
    base = float(np.mean([swing2.baseline(s2p[b]) for b in C.BOARD_ORDER]))
    print(f"  随机基线 {base:.1f}%\n")
    print(f"  {'口径':<26}{'件数':>7}{'命中':>9}{'下界':>8}{'年化':>8}   说明")
    sn, sk, sr_, _ = cstat([e for _, e in u_ev])
    print(f"  {'信号级':<26}{sn:>7}{sr_:>8.1f}%{'':>8}{sn/3:>8.1f}   页面画的点数")
    cl_b = s19.cluster(u_ev)
    cn, ck, cr, cwl = cstat(cl_b)
    print(f"  {'事件级（同板块 ±4 日）':<24}{cn:>7}{cr:>8.1f}%{cwl:>7.1f}%{cn/3:>8.1f}   单板块的入场机会")
    cl_m = cluster_market(u_ev)
    mn, mk, mr, mwl = cstat([c[1] for c in cl_m])
    print(f"  {'市场级（跨板块 ±5 日）':<24}{mn:>7}{mr:>8.1f}%{mwl:>7.1f}%{mn/3:>8.1f}   全市场几波下跌")

    print("\n  簇规模分布（一次事件里涉及几个板块的信号）")
    sizes = [len(c[2]) for c in cl_m]
    for lo, hi in ((1, 1), (2, 3), (4, 6), (7, 12), (13, 99)):
        k = sum(1 for s in sizes if lo <= s <= hi)
        if k:
            print(f"    {lo:>3}~{hi if hi < 99 else '+':<3} 个信号：{k:>3} 次事件")
    spans = [(pd.Timestamp(c[2][-1][1]["date"]) - pd.Timestamp(c[2][0][1]["date"])).days
             for c in cl_m]
    print(f"  簇跨度：中位 {int(np.median(spans))} 天，最长 {max(spans)} 天")

    print("\n" + "=" * 108)
    print("B. 市场级聚类的参数敏感性（gap_days × span_cap）—— 是平台还是刀刃？")
    print("=" * 108)
    print(f"  {'gap':>4}{'span':>6}{'件数':>7}{'命中':>9}{'下界':>8}{'年化':>8}")
    for gap in (3, 5, 7, 10):
        for cap in (10, 15, 20, 30, 9999):
            cl = cluster_market(u_ev, gap, cap)
            n_, k_, r_, w_ = cstat([c[1] for c in cl])
            print(f"  {gap:>4}{('无' if cap > 999 else cap):>6}{n_:>7}{r_:>8.1f}%{w_:>7.1f}%{n_/3:>8.1f}")

    print("\n" + "=" * 108)
    print("C. SENTI-1（主模型）的三层口径 —— 把 AGENTS.md 里的「5 次市场级事件」算实")
    print("=" * 108)
    print(f"  {'口径':<26}{'件数':>7}{'命中':>9}{'下界':>8}   说明")
    bn, bk, br, bwl = cstat([e for _, e in b_ev])
    print(f"  {'信号级':<26}{bn:>7}{br:>8.1f}%{bwl:>7.1f}%   逐板块信号")
    cl1 = s19.cluster(b_ev)
    cn1, ck1, cr1, cwl1 = cstat(cl1)
    print(f"  {'事件级（同板块 ±4 日）':<24}{cn1:>7}{cr1:>8.1f}%{cwl1:>7.1f}%   单板块入场机会")
    clm1 = cluster_market(b_ev)
    mn1, mk1, mr1, mwl1 = cstat([c[1] for c in clm1])
    print(f"  {'市场级（跨板块 ±5 日）':<24}{mn1:>7}{mr1:>8.1f}%{mwl1:>7.1f}%   全市场几波下跌")

    print("\n  市场级事件逐笔（SENTI-1，判定 T+20 期末>0 且期间回撤 ≥ −3%）")
    print(f"  {'首次信号日':<14}{'板块':<10}{'涉及板块数':>10}{'该笔命中':>9}")
    for b, e, c in clm1:
        boards = len({x[0] for x in c})
        print(f"  {e['date']:<14}{C.BOARDS[b]['name']:<10}{boards:>10}"
              f"{'✔' if e['ok'] else '✘':>9}")

    print("\n" + "=" * 108)
    print("D. 结论")
    print("=" * 108)
    print(f"  小波段并集：信号 {sn} → 事件 {cn} → 市场级 {mn} 次")
    print(f"    命中率 {sr_:.1f}% → {cr:.1f}% → {mr:.1f}%　（市场级下界 {mwl:.1f}%）")
    print(f"  SENTI-1：信号 {bn} → 事件 {cn1} → 市场级 {mn1} 次")
    print(f"    命中率 {br:.1f}% → {cr1:.1f}% → {mr1:.1f}%　（市场级下界 {mwl1:.1f}%）")
    print("\n  含义：市场级件数才是「这套系统一年动手几次」的答案；")
    print("  但它把「某板块单独崩」的信息抹掉了（那正是 SWING 定位的局部流动性危机），")
    print("  所以市场级只作为**报告口径**，不作为过滤条件（与 reso 共振标签同一个道理）。")


if __name__ == "__main__":
    main()
