"""s35 —— 第二十七轮：代理变量检验（决定性）

s34 出现了两个「看起来有效」的外部过滤器：
    10Y 国债 ≤ 3.1        → 603 信号 66.7%（基准 59.0%），前半 57.8%、后半 70.0%
    10Y 低于 MA20 5bp     → 250 信号 66.4%（基准 59.0%），前半 53.1%、后半 84.8%
    全A中位 PE ≤ P25      → 231 信号 68.4%（基准 59.0%），前半 69.9%、后半 67.4%
它们都「过了」分半检验。但 s34 的 D 段露出一处马脚：
    **2015 列全是「—」** —— 这些过滤器把 2015 年的信号几乎全剔掉了，
    而 2015-06 与 2016-01 正是本项目已知的**失败簇**。
这就是 s30 里那个「波动过滤」的同款陷阱：**过滤条件在全样本有效，
可能只是「避开某段坏行情」的代理变量。**

本脚本用四个正面检验来判定（全部可复现）：
A. 对照诊断：被剔掉的信号成色如何？剔掉的人里 2015+2016 占比 vs 全样本占比
B. 逐年明细：9 年里过滤器有几年真的比该年基准好？（s30 用的是这一条）
C. 排除 2015-2016 后重测：增益还剩多少？（代理变量的核心判据）
D. 去趋势版本：把绝对阈值换成**因果分位**，看是否还有效
E. 参数邻域：是平台还是刀刃
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
import s12                                       # noqa: F401
import s17
import s18
import s19
import s24                                       # noqa: F401
import s32                                       # noqa: F401
import s34
from s8 import wilson, BOARDS                    # noqa: E402

CACHE_DIR = s8.C.CACHE_DIR
W_Q = s34.W_Q
W_57 = s34.W_57
W_3 = s34.W_3
H1 = s34.H1
H2 = s34.H2
st = s34.st
yr = s34.yr
mk_series = s34.mk_series

# 2015-2016 是本项目已知的失败簇（见 AGENTS.md §6 长历史结论）
EXCL = (pd.Timestamp("2015-01-01"), pd.Timestamp("2016-12-31"))
NO_EXCL = (pd.Timestamp("2015-01-01"), pd.Timestamp("1900-01-01"))   # 空区间


def split(lev, ser, cond):
    keep, drop = [], []
    for r in lev:
        v = ser[r[0]].loc[pd.Timestamp(r[1])]
        (keep if (not np.isnan(v) and cond(v)) else drop).append(r)
    return keep, drop


def rate(ev):
    n = len(ev)
    k = sum(1 for r in ev if r[2])
    return n, k, (100 * k / n if n else float("nan")), wilson(k, n)


def in_win(r, w):
    return w[0] <= pd.Timestamp(r[1]) <= w[1]


def main():
    E = s8.Evaluator()
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])
    lev = [(b, dt, ok, rt) for b, d in inc.items() for dt, (ok, rt) in d.items()]

    # 数据
    bond = pd.read_parquet(os.path.join(CACHE_DIR, "bond_rate.parquet"))
    b = bond.rename(columns={"日期": "date"})
    b["date"] = pd.to_datetime(b["date"])
    b = b.set_index("date").sort_index()
    y10 = pd.to_numeric(b["中国国债收益率10年"], errors="coerce")

    val = pd.read_parquet(os.path.join(CACHE_DIR, "a_valuation.parquet"))
    v = val.rename(columns={"date": "date"})
    v["date"] = pd.to_datetime(v["date"])
    v = v.set_index("date").sort_index()
    pe = pd.to_numeric(v["middlePETTM"], errors="coerce")

    S10 = mk_series(E, y10)
    SMA = mk_series(E, y10 - y10.rolling(20).mean())
    SPE = mk_series(E, pe)

    CAND = [
        ("10Y ≤ 3.1", S10, lambda x: x <= 3.1),
        ("10Y 低于MA20 5bp", SMA, lambda x: x <= -0.05),
        ("全A PE ≤ P25", SPE, lambda x: x <= 21.0),
    ]

    print("=" * 116)
    print("A. 对照诊断：被剔掉的信号成色如何？（代理变量的第一处马脚）")
    print("=" * 116)
    n0, k0, r0, _ = rate(lev)
    p_all = np.mean([in_win(r, EXCL) for r in lev])
    print(f"  全样本：{n0} 个信号，命中 {r0:.1f}%；其中落在 2015-2016 的占 {p_all:.1%}"
          f"（{int(p_all*n0)} 个）")
    print(f"\n  {'过滤器':<20}{'保留n':>7}{'保留命中':>9}{'剔除n':>7}{'剔除命中':>9}"
          f"{'对比差':>8}{'剔除里15-16占比':>16}{'全样本占比':>11}")
    for lab, ser, cond in CAND:
        keep, drop = split(lev, ser, cond)
        nk, kk, rk, _ = rate(keep)
        nd, kd, rd, _ = rate(drop)
        pd_ = np.mean([in_win(r, EXCL) for r in drop]) if drop else float("nan")
        print(f"  {lab:<20}{nk:>7}{rk:>8.1f}%{nd:>7}{rd:>8.1f}%{rk-rd:>+8.1f}"
              f"{pd_:>15.1%}{p_all:>11.1%}")
    print("\n  判读：若「剔除命中」明显低于「保留命中」，说明过滤器在剔除坏信号；")
    print("        但若「剔除里 15-16 占比」远高于全样本，它可能只是 2015-2016 的代理。")

    print("\n" + "=" * 116)
    print("B. 逐年明细：9 年里过滤器有几年真的比该年基准好？")
    print("=" * 116)
    years = [str(y) for y in range(2015, 2027)]
    for lab, ser, cond in CAND:
        keep, _ = split(lev, ser, cond)
        print(f"\n  ── {lab}")
        print(f"  {'年份':<7}{'全样本':>14}{'过滤后':>14}{'差':>9}{'该年保留率':>11}")
        win = 0
        tot = 0
        for y in years:
            a = [r for r in lev if r[1][:4] == y]
            c = [r for r in keep if r[1][:4] == y]
            if not a:
                continue
            ra = 100 * sum(1 for r in a if r[2]) / len(a)
            if not c:
                print(f"  {y:<7}{f'{ra:.0f}%({len(a)})':>14}{'全剔':>14}{'':>9}{'0%':>11}")
                tot += 1
                continue
            rc = 100 * sum(1 for r in c if r[2]) / len(c)
            tot += 1
            win += 1 if rc > ra else 0
            print(f"  {y:<7}{f'{ra:.0f}%({len(a)})':>14}{f'{rc:.0f}%({len(c)})':>14}"
                  f"{rc-ra:>+9.1f}{len(c)/len(a):>11.0%}")
        print(f"  → 有效年份 {win}/{tot} 变好")

    print("\n" + "=" * 116)
    print("C. 排除 2015-2016 后重测（代理变量的核心判据，s30 同款）")
    print("=" * 116)
    base_all = [r for r in lev if not in_win(r, EXCL)]
    base_1516 = [r for r in lev if in_win(r, EXCL)]
    print(f"  基准：全样本 {rate(lev)[2]:.1f}%({len(lev)})　"
          f"排除15-16后 {rate(base_all)[2]:.1f}%({len(base_all)})　"
          f"仅15-16 {rate(base_1516)[2]:.1f}%({len(base_1516)})")
    print(f"\n  {'过滤器':<20}{'全程增益':>10}{'排除15-16后增益':>17}{'增益留存':>10}"
          f"{'仅15-16保留':>14}")
    for lab, ser, cond in CAND:
        keep, _ = split(lev, ser, cond)
        g_all = rate(keep)[2] - rate(lev)[2]
        ka = [r for r in keep if not in_win(r, EXCL)]
        g_sub = rate(ka)[2] - rate(base_all)[2]
        kk = [r for r in keep if in_win(r, EXCL)]
        print(f"  {lab:<20}{g_all:>+9.1f}pp{g_sub:>+16.1f}pp"
              f"{g_sub/g_all if g_all else float('nan'):>10.0%}"
              f"{f'{len(kk)}/{len(base_1516)}':>14}")
    print("\n  判读：若排除 15-16 后增益大幅缩水（留存 < 40%），说明它主要在「避开崩盘年」。")

    print("\n" + "=" * 116)
    print("D. 去趋势版本：把绝对阈值换成因果分位（去掉利率长期下行/估值中枢漂移）")
    print("=" * 116)
    print("  ⚠️ 10Y 从 2013 的 4.5% 一路降到 2025 的 1.7%，固定绝对阈值 = 「只在近几年买」，")
    print("     这是**制度代理**。正确做法是滚动因果分位（去掉趋势，只留相对高低）。")
    for win_, mn in ((250, 125), (500, 250)):
        pct = y10.rolling(win_, min_periods=mn).rank(pct=True).shift(1)
        Sp = mk_series(E, pct)
        print(f"\n  ── 10Y 因果分位 rolling({win_}, min_periods={mn})")
        print(f"  {'条件':<22}{'保留':>7}{'命中':>9}{'下界':>8}{'全程增益':>10}"
              f"{'排除15-16增益':>15}{'近5.7年':>14}{'近3年':>14}")
        for t in (0.2, 0.3, 0.5, 0.7):
            keep, _ = split(lev, Sp, lambda x, t=t: x <= t)
            if len(keep) < 30:
                continue
            ka = [r for r in keep if not in_win(r, EXCL)]
            n_, k_, r_, wl = rate(keep)
            print(f"  {'分位 ≤ ' + str(t):<22}{n_:>7}{r_:>8.1f}%{wl:>8.1f}"
                  f"{r_-rate(lev)[2]:>+9.1f}pp"
                  f"{rate(ka)[2]-rate(base_all)[2]:>+14.1f}pp"
                  f"{f'{st(keep, *W_57)[2]:.1f}%({st(keep, *W_57)[0]})':>14}"
                  f"{f'{st(keep, *W_3)[2]:.1f}%({st(keep, *W_3)[0]})':>14}")
    # PE 因果分位（月频，60 个月 ≈ 5 年）
    ppct = pe.rolling(60, min_periods=36).rank(pct=True).shift(1)
    Sp2 = mk_series(E, ppct)
    print(f"\n  ── 全A中位PE 因果分位 rolling(60 月)")
    print(f"  {'条件':<22}{'保留':>7}{'命中':>9}{'下界':>8}{'全程增益':>10}"
          f"{'排除15-16增益':>15}{'近5.7年':>14}{'近3年':>14}")
    for t in (0.2, 0.3, 0.5):
        keep, _ = split(lev, Sp2, lambda x, t=t: x <= t)
        if len(keep) < 30:
            continue
        ka = [r for r in keep if not in_win(r, EXCL)]
        n_, k_, r_, wl = rate(keep)
        print(f"  {'PE 分位 ≤ ' + str(t):<22}{n_:>7}{r_:>8.1f}%{wl:>8.1f}"
              f"{r_-rate(lev)[2]:>+9.1f}pp"
              f"{rate(ka)[2]-rate(base_all)[2]:>+14.1f}pp"
              f"{f'{st(keep, *W_57)[2]:.1f}%({st(keep, *W_57)[0]})':>14}"
              f"{f'{st(keep, *W_3)[2]:.1f}%({st(keep, *W_3)[0]})':>14}")

    print("\n" + "=" * 116)
    print("E. 参数邻域：是平台还是刀刃？")
    print("=" * 116)
    print(f"  {'MA窗口':>8}{'阈值':>8}{'保留':>7}{'命中':>9}{'全程增益':>10}"
          f"{'排除15-16增益':>15}{'前半':>14}{'后半':>14}")
    for mw in (10, 20, 40, 60):
        for t in (-0.02, -0.05, -0.10, -0.15):
            s = mk_series(E, y10 - y10.rolling(mw).mean())
            keep, _ = split(lev, s, lambda x, t=t: x <= t)
            if len(keep) < 30:
                continue
            ka = [r for r in keep if not in_win(r, EXCL)]
            n_, k_, r_, _ = rate(keep)
            print(f"  {mw:>8}{t:>8.2f}{n_:>7}{r_:>8.1f}%{r_-rate(lev)[2]:>+9.1f}pp"
                  f"{rate(ka)[2]-rate(base_all)[2]:>+14.1f}pp"
                  f"{f'{st(keep, *H1)[2]:.1f}%({st(keep, *H1)[0]})':>14}"
                  f"{f'{st(keep, *H2)[2]:.1f}%({st(keep, *H2)[0]})':>14}")

    print("\n" + "=" * 116)
    print("F. 结论")
    print("=" * 116)
    print("  见 DELIVERY.md 第 16 节。")


if __name__ == "__main__":
    main()
