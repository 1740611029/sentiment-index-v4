"""s33 —— 第二十五轮：QVIX（隐含波动率）能不能给 7 天波段加信息？

数据现实（s32）
------------
只有 **50ETF QVIX** 有完整长历史（2015-02-09 起），其余序列的起点是各自期权上市日
（300ETF 2019-12、中证1000 2022-07、500ETF/创业板 2022-09、上证50指数 2023-01）。
所以主口径只能是 **50ETF QVIX 作为市场级恐慌读数**（对所有板块用同一条序列），
板块专属序列只能在近 3~6 年窗口做二次确认。

口径选择
------
用 **QVIX 绝对水平**，不用因果锚分位：隐含波动率本身就是年化百分比的绝对刻度
（类似 VIX>30 即恐慌），跨年代可比；而且 50ETF QVIX 从 2015-02 起，
若用 rolling(750, min_periods=500) 做锚，前 500 个有效值全 NaN →
**2015-06 与 2016-01 两次崩盘会看不见**（铁律四那个坑的翻版）。

本脚本
A. QVIX 水平的分年分布（先搞清阈值语义）
B. 作为**独立第 4 模型**：高 IV 抄底 vs 低 IV 抄底，阈值扫描（9 宽基，2015-02 起）
C. 与现役三模型并集：邻域合并后真正新增几个信号？并集增益多少？
D. 作为**确认层**：只在 QVIX 达阈值时才买（过滤现有并集）
E. 分半 + 参数平台
F. 6 板块页面口径对照
G. 结论
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
import s24
import s32
from s8 import wilson, BOARDS                    # noqa: E402

W_Q = (pd.Timestamp("2015-02-09"), pd.Timestamp("2026-09-18"))
W_57 = (pd.Timestamp("2021-01-01"), pd.Timestamp("2026-09-18"))
W_3 = (pd.Timestamp("2023-09-20"), pd.Timestamp("2026-09-18"))
WINS = (("2015起", W_Q), ("近5.7年", W_57), ("近3年", W_3))
H1 = (pd.Timestamp("2015-02-09"), pd.Timestamp("2020-12-31"))
H2 = (pd.Timestamp("2021-01-01"), pd.Timestamp("2026-09-18"))


def qvix_scores(E, W, code="SH50ETF", mode="high"):
    out = {}
    for b in BOARDS:
        v = W[code].reindex(E.CL[b].index).ffill()
        if mode == "high":
            out[b] = -v
        elif mode == "low":
            out[b] = v
        elif mode == "chg":
            out[b] = -(v / v.shift(5) - 1)
        elif mode == "spike":
            out[b] = -(v / v.rolling(20).mean() - 1)
    return out


def st(ev, w0, w1):
    sub = [r for r in ev if w0 <= pd.Timestamp(r[1]) <= w1]
    n = len(sub)
    k = sum(1 for r in sub if r[2])
    return n, k, (100 * k / n if n else float("nan")), wilson(k, n)


def yr(ev, y):
    sub = [r for r in ev if r[1][:4] == y]
    return f"{sum(1 for r in sub if r[2])}/{len(sub)}" if sub else "—"


def main():
    E = s8.Evaluator()
    W = s32.wide(pd.read_parquet(s32.CACHE))
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])

    print("=" * 104)
    print("A. 50ETF QVIX 的分年分布（先搞清阈值语义）")
    print("=" * 104)
    q = W["SH50ETF"].dropna()
    print(f"  {'年份':<8}{'中位':>8}{'P75':>8}{'P90':>8}{'最大':>8}{'>25 天数':>10}"
          f"{'>30 天数':>10}{'>35 天数':>10}")
    for y in range(2015, 2027):
        v = q[q.index.year == y]
        if len(v) < 20:
            continue
        print(f"  {y:<8}{v.median():>8.1f}{v.quantile(.75):>8.1f}{v.quantile(.9):>8.1f}"
              f"{v.max():>8.1f}{int((v > 25).sum()):>10}{int((v > 30).sum()):>10}"
              f"{int((v > 35).sum()):>10}")

    print("\n" + "=" * 104)
    print("B. 作为独立第 4 模型：9 宽基口径（2015-02 起）")
    print("=" * 104)
    print(f"  基线：2015起 {E.base(*W_Q):.1f}%　近5.7年 {E.base(*W_57):.1f}%"
          f"　近3年 {E.base(*W_3):.1f}%")
    for mode, lab, thrs in (("high", "高 IV（恐慌）→ 买", [-20, -22, -25, -28, -30, -35]),
                            ("low", "低 IV（平静）→ 买", [15, 18, 20, 22, 25]),
                            ("chg", "IV 5日急升 → 买", [-0.30, -0.20, -0.15, -0.10]),
                            ("spike", "IV 相对MA20飙升 → 买", [-0.25, -0.15, -0.10, -0.05])):
        print(f"\n  ── {lab}")
        print(f"  {'阈值':>7}{'n':>7}{'命中':>8}{'下界':>7}{'2015':>10}{'2016':>10}"
              f"{'近5.7年':>16}{'近3年':>16}")
        S = qvix_scores(E, W, mode=mode)
        for t in thrs:
            ev = E.run(S, t, 4, *W_Q)
            n, k, r, wl = st(ev, *W_Q)
            if n < 15:
                continue
            n57, k57, r57, _ = st(ev, *W_57)
            n3, k3, r3, _ = st(ev, *W_3)
            print(f"  {t:>7g}{n:>7}{r:>7.1f}%{wl:>7.1f}{yr(ev, '2015'):>10}"
                  f"{yr(ev, '2016'):>10}{f'{r57:.1f}%({n57})':>16}"
                  f"{f'{r3:.1f}%({n3})':>16}")

    print("\n" + "=" * 104)
    print("C. 与现役三模型并集：邻域合并后真正新增多少？并集增益多少？")
    print("=" * 104)
    print(f"  {'候选':<26}{'原始信号':>9}{'新增':>7}{'并集n':>8}{'并集命中':>10}"
          f"{'基准':>8}{'增益':>9}")
    ref = {}
    for lab, (w0, w1) in WINS:
        n_, k_, r_, _ = st([(b, dt, ok, rt) for b, d in inc.items()
                            for dt, (ok, rt) in d.items()], w0, w1)
        ref[lab] = r_
    print(f"  {'【基准】并集(3模型)':<26}{'':>9}{'':>7}{'':>8}{'':>10}{'':>8}{'':>9}"
          f"　全程 {ref['2015起']:.1f}% / 5.7年 {ref['近5.7年']:.1f}% / 3年 {ref['近3年']:.1f}%")
    for mode, t, lab in (("high", -25, "QVIX≥25"), ("high", -30, "QVIX≥30"),
                         ("high", -35, "QVIX≥35"), ("chg", -0.20, "IV 5日涨≥20%"),
                         ("spike", -0.15, "IV 高于MA20 15%")):
        S = qvix_scores(E, W, mode=mode)
        ev = E.run(S, t, 4, *W_Q)
        cand = {}
        for b, dt, ok, rt in ev:
            cand.setdefault(b, {})[dt] = (ok, rt)
        add = s18.dedup_near(E, inc, cand)
        nadd = sum(len(v) for v in add.values())
        nraw = sum(len(v) for v in cand.values())
        uni = s18.merge_all(inc, add)
        cells = []
        gain = []
        for lab2, (w0, w1) in WINS:
            n_, k_, r_, _ = st([(b, dt, ok, rt) for b, d in uni.items()
                                for dt, (ok, rt) in d.items()], w0, w1)
            cells.append(f"{r_:.1f}%")
            gain.append(r_ - ref[lab2])
        print(f"  {lab:<26}{nraw:>9}{nadd:>7}{'':>8}"
              f"{cells[0]:>9}　{gain[0]:+.1f}pp　5.7年 {cells[1]} ({gain[1]:+.1f})"
              f"　3年 {cells[2]} ({gain[2]:+.1f})")

    print("\n" + "=" * 104)
    print("D. 作为确认层：只在 QVIX 达阈值时才买（过滤现有并集）")
    print("=" * 104)
    print(f"  {'过滤条件':<24}{'保留信号':>9}{'命中':>9}{'下界':>8}{'2015':>10}{'2016':>10}"
          f"{'近5.7年':>14}{'近3年':>14}")
    lev = [(b, dt, ok, rt) for b, d in inc.items() for dt, (ok, rt) in d.items()]
    n0, k0, r0, w0_ = st(lev, *W_Q)
    print(f"  {'无过滤':<24}{n0:>9}{r0:>8.1f}%{w0_:>7.1f}{yr(lev, '2015'):>10}"
          f"{yr(lev, '2016'):>10}"
          f"{f'{st(lev, *W_57)[2]:.1f}%({st(lev, *W_57)[0]})':>14}"
          f"{f'{st(lev, *W_3)[2]:.1f}%({st(lev, *W_3)[0]})':>14}")
    for cap in (20, 22, 25, 28, 30):
        keep = []
        for b, dt, ok, rt in lev:
            v = W["SH50ETF"].reindex(E.CL[b].index).ffill()
            i = E.CL[b].index.get_loc(pd.Timestamp(dt))
            x = v.iloc[i]
            if not np.isnan(x) and x >= cap:
                keep.append((b, dt, ok, rt))
        n_, k_, r_, wl = st(keep, *W_Q)
        n57, k57, r57, _ = st(keep, *W_57)
        n3, k3, r3, _ = st(keep, *W_3)
        print(f"  {'只买 QVIX ≥ ' + str(cap):<24}{n_:>9}{r_:>8.1f}%{wl:>7.1f}"
              f"{yr(keep, '2015'):>10}{yr(keep, '2016'):>10}"
              f"{f'{r57:.1f}%({n57})':>14}{f'{r3:.1f}%({n3})':>14}")

    print("\n" + "=" * 104)
    print("E. 分半检验（选在 A 半、验在 B 半）")
    print("=" * 104)
    print(f"  {'口径':<24}{'前半 2015-2020':>18}{'后半 2021-2026':>18}")
    print(f"  {'基准并集':<24}{f'{st(lev, *H1)[2]:.1f}% ({st(lev, *H1)[0]})':>18}"
          f"{f'{st(lev, *H2)[2]:.1f}% ({st(lev, *H2)[0]})':>18}")
    for cap in (20, 25, 30):
        keep = []
        for b, dt, ok, rt in lev:
            v = W["SH50ETF"].reindex(E.CL[b].index).ffill()
            i = E.CL[b].index.get_loc(pd.Timestamp(dt))
            x = v.iloc[i]
            if not np.isnan(x) and x >= cap:
                keep.append((b, dt, ok, rt))
        print(f"  {'QVIX ≥ ' + str(cap):<24}"
              f"{f'{st(keep, *H1)[2]:.1f}% ({st(keep, *H1)[0]})':>18}"
              f"{f'{st(keep, *H2)[2]:.1f}% ({st(keep, *H2)[0]})':>18}")


if __name__ == "__main__":
    main()
