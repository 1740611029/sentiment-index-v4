"""s24 —— 第十六轮：市场级口径必须配「持有期感知的间隔」，否则数字没意义

s23 的结果
--------
小波段并集：信号 189 → 事件级 136（67.6%）→ 市场级 34 件（**50.0%**，下界 34.1%）
命中率掉到基线附近，看着像「并集在市场级完全无效」。

但先别下结论 —— 这里有个方法学问题：
**持有期是 T+7，而市场级聚类只要求簇间间隔 >5 自然日。**
于是同一个下跌过程会被拆成连续几个「独立事件」，第二、三个事件的"第一枪"
其实是在你已经持仓的窗口里开的 —— 那不是新的底部，是**同一笔交易的延续**。
把这种重叠交易当成独立样本统计，必然把命中率拉到基线。

正确做法：市场级事件之间加**持有期感知的间隔**（≥H 个交易日才认作新机会）。
本脚本做四件事
A. 三种簇内聚合规则对照（第一枪 / 任意命中 / 簇内均值）—— 看数字有多依赖规则
B. 加持有期间隔后的市场级口径（这才是「一年动手几次、效果如何」的答案）
C. gap / span_cap / 间隔的敏感性
D. 按簇规模分层（单板块局部危机 vs 全市场系统性下跌）
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s19
import s23
from s8 import wilson                                # noqa: E402


def cstat(evs):
    done = [e for e in evs if e.get("ok") is not None]
    if not done:
        return 0, 0, float("nan"), float("nan")
    k = sum(1 for e in done if e["ok"])
    return len(done), k, 100 * k / len(done), wilson(k, len(done))


def space_out(clusters, cal, min_gap_td):
    """按首个信号的交易日位置，要求相邻被接受事件间隔 ≥ min_gap_td 个交易日。"""
    pos = {d: i for i, d in enumerate(cal)}
    out = []
    last = None
    for c in clusters:
        d = pd.Timestamp(c[1]["date"])
        if d not in pos:
            continue
        i = pos[d]
        if last is not None and i - last < min_gap_td:
            continue
        out.append(c)
        last = i
    return out


def main():
    from senti import config as C
    from senti import store, swing, swing2, swing3

    sp, s2p, s3p = store.swing_panels(), store.swing2_panels(), store.swing3_panels()
    sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
    panels = store.load()
    reso = store.resonance(panels)
    bna = store.bna_series()
    cal = list(panels["SH"].index)          # 用大盘做交易日历

    u_ev = [(b, e) for b in C.BOARD_ORDER
            for e in store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r)]
    b_ev = [(b, e) for b in C.BOARD_ORDER
            for e in store.events(panels[b], reso, bna)]
    base = float(np.mean([swing2.baseline(s2p[b]) for b in C.BOARD_ORDER]))

    print("=" * 106)
    print("A. 簇内聚合规则对照（gap=5，span_cap=20）—— 同一个 34 簇，三种读法")
    print("=" * 106)
    print(f"  随机基线 {base:.1f}%\n")
    for tag, evs in (("小波段三模型并集", u_ev), ("SENTI-1 底部", b_ev)):
        cl = s23.cluster_market(evs)
        print(f"  【{tag}】市场级簇 {len(cl)} 个")
        first = [c[1] for c in cl]
        n_, k_, r_, w_ = cstat(first)
        print(f"    ① 只算第一枪           {n_:>4} 件  {r_:>5.1f}%  下界 {w_:>4.1f}%")
        done = [c for c in cl if all(x[1].get("ok") is not None for x in c[2])]
        any_hit = [1 for c in done if any(x[1]["ok"] for x in c[2])]
        print(f"    ② 簇内任意一个命中       {len(done):>4} 件  "
              f"{100*len(any_hit)/max(len(done),1):>5.1f}%")
        all_hit = [1 for c in done if all(x[1]["ok"] for x in c[2])]
        print(f"    ③ 簇内全部命中           {len(done):>4} 件  "
              f"{100*len(all_hit)/max(len(done),1):>5.1f}%")
        rates = [np.mean([1 if x[1]["ok"] else 0 for x in c[2]]) for c in done]
        print(f"    ④ 簇内平均命中率         {len(done):>4} 件  {100*np.mean(rates):>5.1f}%")
        print()

    print("=" * 106)
    print("B. 加「持有期感知间隔」后的市场级口径（H=7 交易日）")
    print("=" * 106)
    print(f"  {'间隔(交易日)':<14}{'件数':>7}{'命中':>9}{'下界':>8}{'年化':>8}")
    for tag, evs in (("并集", u_ev), ("SENTI-1", b_ev)):
        print(f"  ── {tag}")
        cl = s23.cluster_market(evs)
        for gap_td in (0, 3, 5, 7, 10, 15):
            sub = space_out(cl, cal, gap_td) if gap_td else cl
            n_, k_, r_, w_ = cstat([c[1] for c in sub])
            print(f"  {gap_td:<14}{n_:>7}{r_:>8.1f}%{w_:>7.1f}%{n_/3:>8.1f}")
        print()

    print("=" * 106)
    print("C. 敏感性：gap_days × 间隔（并集，间隔固定 7 交易日）")
    print("=" * 106)
    print(f"  {'gap':>4}{'span':>7}{'件数':>7}{'命中':>9}{'下界':>8}")
    for gap in (3, 5, 7, 10):
        for cap in (10, 20, 9999):
            cl = s23.cluster_market(u_ev, gap, cap)
            sub = space_out(cl, cal, 7)
            n_, k_, r_, w_ = cstat([c[1] for c in sub])
            print(f"  {gap:>4}{('无' if cap > 999 else cap):>7}{n_:>7}{r_:>8.1f}%{w_:>7.1f}%")

    print("\n" + "=" * 106)
    print("D. 按簇规模分层（并集，gap=5，间隔 7 交易日）")
    print("=" * 106)
    cl = s23.cluster_market(u_ev)
    sub = space_out(cl, cal, 7)
    print(f"  {'簇内信号数':<14}{'件数':>7}{'命中':>9}{'下界':>8}   语义")
    for lo, hi, sem in ((1, 1, "单板块局部危机"), (2, 5, "2~5 板块"),
                        (6, 999, "6+ 板块·系统性下跌")):
        part = [c for c in sub if lo <= len(c[2]) <= hi]
        if not part:
            continue
        n_, k_, r_, w_ = cstat([c[1] for c in part])
        print(f"  {lo}~{hi if hi < 999 else '+':<11}{n_:>7}{r_:>8.1f}%{w_:>7.1f}%   {sem}")

    print("\n" + "=" * 106)
    print("E. 结论")
    print("=" * 106)
    cl = s23.cluster_market(u_ev)
    for gap_td in (0, 7):
        sub = space_out(cl, cal, gap_td) if gap_td else cl
        n_, k_, r_, w_ = cstat([c[1] for c in sub])
        print(f"  并集 · 市场级 · 间隔 {gap_td} 交易日：{n_} 件 / {r_:.1f}%（下界 {w_:.1f}%）")
    cl1 = s23.cluster_market(b_ev)
    for gap_td in (0, 7):
        sub = space_out(cl1, cal, gap_td) if gap_td else cl1
        n_, k_, r_, w_ = cstat([c[1] for c in sub])
        print(f"  SENTI-1 · 市场级 · 间隔 {gap_td} 交易日：{n_} 件 / {r_:.1f}%（下界 {w_:.1f}%）")


if __name__ == "__main__":
    main()
