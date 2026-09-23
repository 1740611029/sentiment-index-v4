"""s36 —— 第二十八轮：唯一存活候选的收口复验（10Y 收益率快速下行）

s35 的判决
--------
  10Y ≤ 3.1（绝对水平）   → 剔除组里 52.1% 是 2015-2016（全样本只 23.7%）→ **制度/崩盘年代理**
  全A PE ≤ P25            → 去趋势后「排除 15-16 增益」归零（+6.2pp → −0.0pp）→ **代理**
  10Y 低于 MA20 5bp       → 剔除组里 15-16 占比 21.1% **低于**全样本 23.7%，
                            排除 15-16 后增益反而从 +6.6pp 升到 +12.7pp → **不是代理，真的在做事的**
                            **但** E 段参数邻域是刀刃：同一 MA20 下
                            −0.02 → +10.0pp、−0.05 → +6.6pp、−0.10 → −11.4pp；MA40/60 全负。

所以只剩它需要收口。本脚本用三重检验钉死：
A. 二维网格（MA 窗口 × 阈值）—— 是连续平台还是孤立岛？数「正格 / 总格」与连通性
B. 事件级口径（铁律五）：±3 交易日邻域合并后，增益还在不在？
C. 6 板块页面口径（2021 起）+ 换冷却参数
D. 逐年 + 分半 + 2015 保留
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
import s35
from s8 import wilson                            # noqa: E402

CACHE_DIR = s8.C.CACHE_DIR
W_Q, W_57, W_3, H1, H2 = s34.W_Q, s34.W_57, s34.W_3, s34.H1, s34.H2
st, mk_series = s34.st, s34.mk_series
rate, split, in_win = s35.rate, s35.split, s35.in_win
EXCL = s35.EXCL


def main():
    E = s8.Evaluator()
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])
    lev = [(b, dt, ok, rt) for b, d in inc.items() for dt, (ok, rt) in d.items()]
    base_all = [r for r in lev if not in_win(r, EXCL)]
    r_base, r_base_sub = rate(lev)[2], rate(base_all)[2]

    bond = pd.read_parquet(os.path.join(CACHE_DIR, "bond_rate.parquet"))
    b = bond.rename(columns={"日期": "date"})
    b["date"] = pd.to_datetime(b["date"])
    b = b.set_index("date").sort_index()
    y10 = pd.to_numeric(b["中国国债收益率10年"], errors="coerce")

    print("=" * 118)
    print("A. 二维网格：MA 窗口 × 阈值（正格数 + 连通性）")
    print("=" * 118)
    MAS = [5, 10, 15, 20, 25, 30, 40, 60]
    THRS = [-0.01, -0.02, -0.03, -0.05, -0.08, -0.10, -0.15]
    grid = {}
    for mw in MAS:
        s = mk_series(E, y10 - y10.rolling(mw).mean())
        for t in THRS:
            keep, _ = split(lev, s, lambda x, t=t: x <= t)
            if len(keep) < 40:
                grid[(mw, t)] = None
                continue
            ka = [r for r in keep if not in_win(r, EXCL)]
            grid[(mw, t)] = (len(keep), rate(keep)[2], rate(ka)[2] - r_base_sub)
    print(f"  {'MA\\阈值':<8}" + "".join(f"{t:>11.2f}" for t in THRS))
    for mw in MAS:
        row = f"  {mw:<8}"
        for t in THRS:
            g = grid[(mw, t)]
            row += f"{'—':>11}" if g is None else f"{g[2]:>+10.1f}p"
        print(row)
    print(f"\n  单元格 = 排除 2015-2016 后的增益（pp）；— 表示保留信号 < 40 个")
    vals = [v for v in grid.values() if v]
    pos = [v for v in vals if v[2] > 0]
    print(f"  有效格 {len(vals)} 个，其中正格 {len(pos)} 个（{len(pos)/len(vals):.0%}）")
    print(f"  正格明细：" + "　".join(
        f"MA{mw}/thr{t:+.2f}→{grid[(mw,t)][2]:+.1f}pp(n={grid[(mw,t)][0]})"
        for (mw, t), v in grid.items() if v and v[2] > 0))

    print("\n" + "=" * 118)
    print("B. 事件级口径（铁律五：±3 交易日邻域合并，只留第一枪）")
    print("=" * 118)
    def ev_level(rows, gap_days=4):
        """同板块内 ≤gap_days 自然日算一件事，只留第一枪（铁律五）。保留 board 以便查过滤器。"""
        by_b = {}
        for b_, dt, ok, rt in rows:
            by_b.setdefault(b_, []).append((dt, ok, rt))
        out = []
        for b_, ev in by_b.items():
            ev.sort()
            last = None
            for dt, ok, rt in ev:
                d = pd.Timestamp(dt)
                if last is None or (d - last).days > gap_days:
                    out.append((b_, dt, ok, rt))
                    last = d
        return out

    s = mk_series(E, y10 - y10.rolling(20).mean())
    keep, _ = split(lev, s, lambda x: x <= -0.05)
    lev_e, keep_e = ev_level(lev), ev_level(keep)
    print(f"  {'口径':<26}{'n':>7}{'命中':>9}{'下界':>8}{'基准':>9}{'增益':>9}")
    print(f"  {'信号级 · 无过滤':<26}{rate(lev)[0]:>7}{rate(lev)[2]:>8.1f}%"
          f"{rate(lev)[3]:>8.1f}{r_base:>8.1f}%{0.0:>+8.1f}pp")
    print(f"  {'信号级 · 10Y下降过滤':<26}{rate(keep)[0]:>7}{rate(keep)[2]:>8.1f}%"
          f"{rate(keep)[3]:>8.1f}{r_base:>8.1f}%{rate(keep)[2]-r_base:>+8.1f}pp")
    print(f"  {'事件级 · 无过滤':<26}{rate(lev_e)[0]:>7}{rate(lev_e)[2]:>8.1f}%"
          f"{rate(lev_e)[3]:>8.1f}{'—':>9}{'—':>9}")
    print(f"  {'事件级 · 10Y下降过滤':<26}{rate(keep_e)[0]:>7}{rate(keep_e)[2]:>8.1f}%"
          f"{rate(keep_e)[3]:>8.1f}{rate(lev_e)[2]:>8.1f}%"
          f"{rate(keep_e)[2]-rate(lev_e)[2]:>+8.1f}pp")
    print(f"  保留率：信号级 {len(keep)/len(lev):.0%}　事件级 {len(keep_e)/len(lev_e):.0%}")
    print("  判读：事件级是交付口径。若事件级增益明显小于信号级，说明增益靠重复计数堆出来的。")

    print("\n" + "=" * 118)
    print("C. 6 板块页面口径（2021 起）+ 换冷却参数")
    print("=" * 118)
    print(f"  {'冷却':<6}{'无过滤 n/命中':>18}{'过滤后 n/命中':>18}{'增益':>9}"
          f"{'近3年无过滤':>16}{'近3年过滤后':>16}")
    for cool in (2, 4, 7, 10):
        inc2 = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, cool, s17.COND_FN[cn])
                               for nm, thr, cn in s17.INC])
        lv = [(b_, dt, ok, rt) for b_, d in inc2.items() for dt, (ok, rt) in d.items()]
        kp, _ = split(lv, s, lambda x: x <= -0.05)
        r0 = rate([r for r in lv if W_57[0] <= pd.Timestamp(r[1]) <= W_57[1]])[2]
        r1 = rate([r for r in kp if W_57[0] <= pd.Timestamp(r[1]) <= W_57[1]])[2]
        r0_3 = rate([r for r in lv if W_3[0] <= pd.Timestamp(r[1]) <= W_3[1]])[2]
        r1_3 = rate([r for r in kp if W_3[0] <= pd.Timestamp(r[1]) <= W_3[1]])[2]
        print(f"  {cool:<6}{f'{rate(lv)[2]:.1f}%({len(lv)})':>18}"
              f"{f'{rate(kp)[2]:.1f}%({len(kp)})':>18}{rate(kp)[2]-rate(lv)[2]:>+8.1f}pp"
              f"{f'{r0_3:.1f}%':>16}{f'{r1_3:.1f}%':>16}")
        if cool == 4:
            print(f"         近5.7年：无过滤 {r0:.1f}%({len([r for r in lv if W_57[0] <= pd.Timestamp(r[1]) <= W_57[1]])})"
                  f" → 过滤后 {r1:.1f}%({len([r for r in kp if W_57[0] <= pd.Timestamp(r[1]) <= W_57[1]])})")

    print("\n" + "=" * 118)
    print("D. 逐年 + 分半 + 2015 保留（MA20 / thr −0.05 基准口径）")
    print("=" * 118)
    print(f"  {'年份':<7}{'全样本':>14}{'过滤后':>14}{'差':>9}{'保留率':>9}")
    win = tot = 0
    for y in [str(v) for v in range(2015, 2027)]:
        a = [r for r in lev if r[1][:4] == y]
        c = [r for r in keep if r[1][:4] == y]
        if not a:
            continue
        ra = rate(a)[2]
        tot += 1
        if not c:
            print(f"  {y:<7}{f'{ra:.0f}%({len(a)})':>14}{'全剔':>14}{'':>9}{'0%':>9}")
            continue
        rc = rate(c)[2]
        win += 1 if rc > ra else 0
        print(f"  {y:<7}{f'{ra:.0f}%({len(a)})':>14}{f'{rc:.0f}%({len(c)})':>14}"
              f"{rc-ra:>+9.1f}{len(c)/len(a):>9.0%}")
    print(f"  → 有效年份 {win}/{tot} 变好")
    print(f"\n  分半：前半 {rate([r for r in keep if in_win(r, H1)])[2]:.1f}%"
          f"({rate([r for r in keep if in_win(r, H1)])[0]})　"
          f"后半 {rate([r for r in keep if in_win(r, H2)])[2]:.1f}%"
          f"({rate([r for r in keep if in_win(r, H2)])[0]})")
    print(f"  基准：前半 {rate([r for r in lev if in_win(r, H1)])[2]:.1f}%　"
          f"后半 {rate([r for r in lev if in_win(r, H2)])[2]:.1f}%")

    print("\n" + "=" * 118)
    print("E. 机制质疑：『10Y 快速下行』会不会只是『股市正在跌』的代理？")
    print("=" * 118)
    print("  逻辑上：股债跷跷板 —— 股市急跌时避险资金涌入债市，收益率自然下行。")
    print("  若如此，这个过滤器只是「跌得更深」的代理，而「跌得深」价量因子已经知道。")
    print("  检验法：把信号按**价量状态**分桶，看桶内 10Y 下降组是否仍然更好（同环境基线）。")
    P = E.score("pos")
    m5 = {}
    m20 = {}
    for b_ in s8.BOARDS:
        c = E.CL[b_]
        m5[b_] = c / c.shift(5) - 1
        m20[b_] = c / c.shift(20) - 1
    print(f"\n  {'状态量':<10}{'保留组中位':>13}{'剔除组中位':>13}{'差':>10}{'判读':>28}")
    for lab, get in (("pos 分值", lambda b_, dt: P[b_].loc[pd.Timestamp(dt)]),
                     ("5日跌幅", lambda b_, dt: m5[b_].loc[pd.Timestamp(dt)]),
                     ("20日跌幅", lambda b_, dt: m20[b_].loc[pd.Timestamp(dt)])):
        kv = np.array([get(b_, dt) for b_, dt, _, _ in keep], float)
        dv = np.array([get(b_, dt) for b_, dt, _, _ in split(lev, s, lambda x: x <= -0.05)[1]], float)
        kv, dv = kv[~np.isnan(kv)], dv[~np.isnan(dv)]
        flag = "⚠️ 两组状态不同 → 有代理嫌疑" if abs(np.median(kv) - np.median(dv)) > (
            1.0 if lab == "pos 分值" else 0.004) else "两组接近"
        print(f"  {lab:<10}{np.median(kv):>13.3f}{np.median(dv):>13.3f}"
              f"{np.median(kv)-np.median(dv):>+10.3f}{flag:>28}")

    print(f"\n  桶内对照（按 pos 分值十分位分桶，桶内比较保留/剔除）")
    pv = np.array([P[b_].loc[pd.Timestamp(dt)] for b_, dt, _, _ in lev], float)
    m = ~np.isnan(pv)
    idx = np.where(m)[0]
    edges = np.quantile(pv[idx], np.linspace(0, 1, 11))
    print(f"  {'pos 桶':<14}{'桶内 n':>8}{'保留 n':>8}{'保留命中':>10}{'剔除命中':>10}"
          f"{'差':>9}")
    wins = tot = 0
    for j in range(10):
        lo, hi = edges[j], edges[j + 1]
        sel = [i for i in idx if (pv[i] >= lo and (pv[i] <= hi if j == 9 else pv[i] < hi))]
        if len(sel) < 20:
            continue
        kk = [lev[i] for i in sel if s[lev[i][0]].loc[pd.Timestamp(lev[i][1])] <= -0.05]
        dd = [lev[i] for i in sel if not (s[lev[i][0]].loc[pd.Timestamp(lev[i][1])] <= -0.05)]
        if len(kk) < 5 or len(dd) < 5:
            print(f"  {f'{lo:.0f}-{hi:.0f}':<14}{len(sel):>8}{len(kk):>8}{'样本少':>10}{'':>10}")
            continue
        rk, rd = rate(kk)[2], rate(dd)[2]
        tot += 1
        wins += 1 if rk > rd else 0
        print(f"  {f'{lo:.0f}-{hi:.0f}':<14}{len(sel):>8}{len(kk):>8}{rk:>9.1f}%"
              f"{rd:>9.1f}%{rk-rd:>+9.1f}")
    print(f"  → {wins}/{tot} 个桶内保留组更好")

    print("\n" + "=" * 118)
    print("F. 结论")
    print("=" * 118)
    print("  见 DELIVERY.md 第 16 节。")


if __name__ == "__main__":
    main()
