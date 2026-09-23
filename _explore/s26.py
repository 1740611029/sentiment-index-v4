"""s26 —— 第十八轮：长簇（span ≥ 8 天）为什么差，以及能不能实时用

s25 的结构（两个样本都复现）
--------------------------
  span 0 天    26 件 / 73.1%      6 板块 12 件 / 58% 量级
  span 1~2 天  10 件 / 70.0%
  span 3~7 天  20 件 / 75.0%
  span 8 天+   19 件 / **21.1%**（下界 8.5%）
s25 的 C 段把分界点错设成 2 天，所以看不出稳健性。真正的分界点在 ~8 天。

必须先回答一个致命问题：这会不会是同义反复？
------------------------------------------
如果一个簇跨 18 天，说明这 18 天里市场**一直在创新低**（信号不断触发）。
那么第一枪的 T+7 窗口（≈10 自然日）整段都落在继续下跌的过程里 → 当然亏。
**若如此，「span」就是一个用未来信息定义的标签，拿它当过滤器等于铁律二（事后挑点）。**

所以本脚本第一件事是**判定它是不是同义反复**：
  对每个簇，检查「簇内最后一个信号日」是否落在「第一枪的 T+7 窗口」之内。
  若长簇几乎全部落在窗口内 → 同义反复，只能当**事后诊断标签**，不能当过滤器。
  若长簇里有相当一部分是"跌完了才陆续出信号"（最后一个信号在 T+7 窗口之后）
  → 说明它是真实的「阴跌 vs 投降」形态差异，有独立信息。

然后测一个**实时可用**的版本
--------------------------
「市场新鲜度」过滤：只有当**过去 N 个交易日内任何板块都没出过信号**时才接受本信号。
（与 s22 测过的**单板块内**冷却不同，这是**跨板块**的；与 s7 的信号级冷却也不同。）
注意它是顺序依赖的（接受后才影响后续），必须逐日模拟。

A. 同义反复判定
B. span ≥8 分界点的稳健性（gap × 间隔 × 两个样本）
C. 市场新鲜度过滤（实时）扫描
D. 结论
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


def cstat(evs):
    return s24.cstat(evs)


def tautology(clusters, cal, H=7):
    """检查：簇内最后一个信号日是否落在第一枪的 T+7 窗口内（= 还在继续跌）。"""
    pos = {d: i for i, d in enumerate(cal)}
    inside = outside = 0
    for c in clusters:
        d0 = pd.Timestamp(c[1]["date"])
        dl = pd.Timestamp(c[2][-1][1]["date"])
        if d0 not in pos or dl not in pos:
            continue
        if pos[dl] - pos[d0] <= H:
            inside += 1
        else:
            outside += 1
    return inside, outside


def freshness(events, cal, N):
    """跨板块新鲜度：过去 N 个交易日内任何板块出过信号 → 跳过本信号（顺序依赖）。"""
    pos = {d: i for i, d in enumerate(cal)}
    ev = sorted(events, key=lambda x: (x[1]["date"], x[0]))
    last_all = None
    out = []
    for b, e in ev:
        d = pd.Timestamp(e["date"])
        if d not in pos:
            continue
        i = pos[d]
        if last_all is not None and i - last_all <= N:
            continue
        out.append((b, e))
        last_all = i
    return out


def main():
    from senti import config as C
    from senti import store, swing, swing2, swing3

    sp, s2p, s3p = store.swing_panels(), store.swing2_panels(), store.swing3_panels()
    sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
    panels = store.load()
    cal = list(panels["SH"].index)
    u_ev = [(b, e) for b in C.BOARD_ORDER
            for e in store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r)]
    base = float(np.mean([swing2.baseline(s2p[b]) for b in C.BOARD_ORDER]))

    print("=" * 100)
    print("A. 同义反复判定：长簇的第一枪是不是「买在继续跌的中途」")
    print("=" * 100)
    for gap, gtd in ((5, 7), (5, 0)):
        c2 = s25.prep(s24.space_out(s23.cluster_market(u_ev, gap), cal, gtd))
        short = [c for c in c2 if c[3]["span"] <= 7]
        long_ = [c for c in c2 if c[3]["span"] >= 8]
        print(f"  gap={gap} 间隔={gtd}")
        for tag, grp in (("短簇(≤7天)", short), ("长簇(≥8天)", long_)):
            ins, outs = tautology(grp, cal)
            n_, k_, r_, w_ = cstat([c[1] for c in grp])
            tot = max(ins + outs, 1)
            print(f"    {tag:<12}{n_:>3} 件 命中 {r_:>5.1f}%"
                  f"　最后一个信号落在 T+7 窗口内 {ins}/{tot}"
                  f"（{100*ins/tot:.0f}%）")
        print()

    print("=" * 100)
    print("B. span ≥8 分界点的稳健性（6 板块 / 9 宽基长历史）")
    print("=" * 100)
    print(f"  6 板块（基线 {base:.1f}%）")
    print(f"  {'gap':>4}{'间隔':>5}{'短簇件数':>9}{'短簇命中':>9}{'长簇件数':>9}{'长簇命中':>9}")
    for gap in (3, 5, 7, 10):
        for gtd in (0, 5, 7, 10):
            c2 = s25.prep(s24.space_out(s23.cluster_market(u_ev, gap), cal, gtd))
            s_ = [c for c in c2 if c[3]["span"] <= 7]
            l_ = [c for c in c2 if c[3]["span"] >= 8]
            ns, ks, rs, _ = cstat([c[1] for c in s_])
            nl, kl, rl, _ = cstat([c[1] for c in l_])
            print(f"  {gap:>4}{gtd:>5}{ns:>9}{rs:>8.1f}%{nl:>9}{rl:>8.1f}%")

    E = s8.Evaluator()
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])
    lev = [(b, {"date": dt, "ok": ok}) for b, d in inc.items()
           for dt, (ok, _) in d.items()]
    cal9 = list(E.CL["SH"].index)
    print(f"\n  9 宽基长历史（基线 {E.base(*W_ALL):.1f}%）")
    print(f"  {'gap':>4}{'间隔':>5}{'短簇件数':>9}{'短簇命中':>9}{'长簇件数':>9}{'长簇命中':>9}")
    for gap in (3, 5, 7, 10):
        for gtd in (0, 7):
            c2 = s25.prep(s24.space_out(s23.cluster_market(lev, gap), cal9, gtd))
            s_ = [c for c in c2 if c[3]["span"] <= 7]
            l_ = [c for c in c2 if c[3]["span"] >= 8]
            ns, ks, rs, _ = cstat([c[1] for c in s_])
            nl, kl, rl, _ = cstat([c[1] for c in l_])
            print(f"  {gap:>4}{gtd:>5}{ns:>9}{rs:>8.1f}%{nl:>9}{rl:>8.1f}%")

    print("\n" + "=" * 100)
    print("C. 市场新鲜度过滤（实时可用，跨板块）：过去 N 交易日内任何板块出过信号就跳过")
    print("=" * 100)
    print(f"  {'N(交易日)':<12}{'件数':>8}{'命中':>9}{'下界':>8}{'年化':>8}   相对无过滤")
    n0, k0, r0, w0 = cstat([e for _, e in u_ev])
    print(f"  {'无过滤':<12}{n0:>8}{r0:>8.1f}%{w0:>7.1f}%{n0/3:>8.1f}   —")
    for N in (1, 2, 3, 5, 7, 8, 10, 12, 15, 20):
        f = freshness(u_ev, cal, N)
        n_, k_, r_, w_ = cstat([e for _, e in f])
        print(f"  {N:<12}{n_:>8}{r_:>8.1f}%{w_:>7.1f}%{n_/3:>8.1f}   {r_-r0:+.1f}pp")

    print("\n" + "=" * 100)
    print("D. 结论")
    print("=" * 100)
    c2 = s25.prep(s24.space_out(s23.cluster_market(u_ev, 5), cal, 7))
    s_ = [c for c in c2 if c[3]["span"] <= 7]
    l_ = [c for c in c2 if c[3]["span"] >= 8]
    ins, outs = tautology(l_, cal)
    print(f"  长簇（≥8 天）：{len(l_)} 件，其中 {ins} 件的最后一个信号仍在第一枪的 T+7 窗口内")
    print(f"  → 若该比例接近 100%，span 就是「用未来定义」的标签，只能做事后诊断；")
    print(f"    市场新鲜度过滤（实时版）的效果见 C 段。")


if __name__ == "__main__":
    main()
