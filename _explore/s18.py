"""s18 —— 第十轮（关键复核）：s17 的 35 组「过关者」是真增益还是重复计数？

s17 的问题
----------
s17 判据只做了「同日去重」。但 s17 里过关的 35 组候选（roc10+up1、gap+up1、
vol_ratio+up1、ppo+vconv、slope_combo+vconv …）全是 ma20_slope 的同族近亲：
都是「动量转弱 + 止跌确认」。它们触发的日子和 SWING-3 常常只差 1~2 天。

同一波下跌里，9 月 10 日触发 SWING-3、9 月 12 日触发 roc10，**是同一件事**。
只按同日去重会把它算成两个信号 → 信号数虚增、命中率被重复样本拉高。
这正是本项目铁律二（不许重复计同一个事件）要防的东西。

本脚本做三件事
------------
A. 把「±MERGE 个交易日内的信号」合并成同一事件，重算 s17 的过关者
B. 分半稳定检验：2013-2019 选、2020-2026 验（以及反向），增益必须两半都不为负
C. 顺带体检交付口径：6 板块并集 189 次里有多少是相邻日重复
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
import s17
from s8 import wilson, W_ALL, W_57, W_3, BOARDS  # noqa: E402

WINS = (("全程", W_ALL), ("近5.7年", W_57), ("近3年", W_3))
H1 = (pd.Timestamp("2013-01-01"), pd.Timestamp("2019-12-31"))
H2 = (pd.Timestamp("2020-01-01"), pd.Timestamp("2026-09-18"))
MERGE = 3          # ±3 个交易日视为同一事件


def pos_of(E, b, dt):
    return E.CL[b].index.get_loc(pd.Timestamp(dt))


def dedup_near(E, base, add, merge=MERGE):
    """把 add 里落在 base 的 ±merge 交易日邻域内的信号丢掉（同一事件）。"""
    out = {}
    for b in BOARDS:
        idx = E.CL[b].index
        bp = sorted(pos_of(E, b, d) for d in base.get(b, {}))
        keep = {}
        for dt, v in add.get(b, {}).items():
            i = pos_of(E, b, dt)
            if any(abs(i - j) <= merge for j in bp):
                continue
            keep[dt] = v
        out[b] = keep
    return out


def merge_all(*sets):
    out = {}
    for S in sets:
        for b, d in S.items():
            out.setdefault(b, {}).update(d)
    return out


def rate(uni, w0, w1):
    n = k = 0
    for b, d in uni.items():
        for dt, (ok, _) in d.items():
            if w0 <= pd.Timestamp(dt) <= w1:
                n += 1
                k += int(ok)
    return n, k, (100 * k / n if n else float("nan"))


def show(tag, uni, extra=""):
    cells = []
    for lab, (w0, w1) in WINS:
        n, k, r = rate(uni, w0, w1)
        cells.append(f"{r:5.1f}%({n:>3})")
    print(f"  {tag:<28}{cells[0]:>14}{cells[1]:>14}{cells[2]:>14}{extra}")


def main():
    E = s8.Evaluator()
    inc = merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                      for nm, thr, cn in s17.INC])

    print("=" * 116)
    print("A. 邻域合并（±3 交易日 = 同一事件）对 s17 过关者的影响")
    print("=" * 116)
    print(f"  {'候选':<28}{'全程':>14}{'近5.7年':>14}{'近3年':>14}  新增/被并")
    show("【基准】并集(3 模型)", inc)

    names = [n for n in s8.FACTORS if n not in ("pos", "macd", "ma20_slope")]
    survivors = []
    for nm in names:
        S = E.score(nm)
        for cn, cond in s17.COND_FN.items():
            for thr in s17.THRS:
                cand = s17.sig_dates(E, S, thr, 4, cond)
                uni = merge_all(inc, cand)
                # s17 判据
                rs = {lab: rate(uni, w0, w1)[2] for lab, (w0, w1) in WINS}
                ref = {lab: rate(inc, w0, w1)[2] for lab, (w0, w1) in WINS}
                if not all(rs[l] >= ref[l] - 1e-9 for l, _ in WINS):
                    continue
                if not any(rs[l] > ref[l] + 1e-9 for l, _ in WINS):
                    continue
                # 邻域合并后重算
                add = dedup_near(E, inc, cand)
                nadd = sum(len(v) for v in add.values())
                nraw = sum(len(v) for v in cand.values())
                uni2 = merge_all(inc, add)
                r2 = {lab: rate(uni2, w0, w1)[2] for lab, (w0, w1) in WINS}
                keep = all(r2[l] >= ref[l] - 1e-9 for l, _ in WINS) and \
                    any(r2[l] > ref[l] + 1e-9 for l, _ in WINS)
                tag = f"{nm}≤{thr:g}" + (f"+{cn}" if cn != "无" else "")
                show(("✔ " if keep else "✘ ") + tag, uni2,
                     f"  {nraw - nadd:>3}/{nraw:<3}被并")
                if keep:
                    survivors.append((nm, thr, cn, add))
    print(f"\n  邻域合并后仍过关：{len(survivors)} 组")

    print("\n" + "=" * 116)
    print("B. 分半稳定检验：增益必须在 2013-2019 与 2020-2026 两半都不为负")
    print("=" * 116)
    print(f"  {'候选':<28}{'前半 基准→加后':>20}{'后半 基准→加后':>20}   判定")
    stable = []
    for nm, thr, cn, add in survivors:
        uni2 = merge_all(inc, add)
        r1b = rate(inc, *H1)[2]
        r1a = rate(uni2, *H1)[2]
        r2b = rate(inc, *H2)[2]
        r2a = rate(uni2, *H2)[2]
        ok = (r1a >= r1b - 1e-9) and (r2a >= r2b - 1e-9)
        print(f"  {nm + '≤' + f'{thr:g}' + ('+' + cn if cn != '无' else ''):<28}"
              f"{f'{r1b:.1f}%→{r1a:.1f}%':>20}{f'{r2b:.1f}%→{r2a:.1f}%':>20}"
              f"   {'★稳定' if ok else '✘ 不稳'}")
        if ok:
            stable.append((nm, thr, cn, add))
    print(f"\n  两半都不为负：{len(stable)} 组")

    if stable:
        print("\n" + "=" * 116)
        print("C. 稳定组逐个 + 全体并集")
        print("=" * 116)
        show("【基准】并集(3 模型)", inc)
        alladd = {}
        for nm, thr, cn, add in stable:
            uni2 = merge_all(inc, add)
            show(f"{nm}≤{thr:g}" + (f"+{cn}" if cn != "无" else ""), uni2)
            alladd = merge_all(alladd, add)
        show("并集(3 模型) ∪ 全部稳定组", merge_all(inc, alladd))

    print("\n" + "=" * 116)
    print("D. 交付口径体检：6 板块并集 189 次里有多少是相邻日重复")
    print("=" * 116)
    from senti import config as C
    from senti import store, swing, swing2, swing3
    sp, s2p, s3p = store.swing_panels(), store.swing2_panels(), store.swing3_panels()
    sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
    uev = {b: store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r)
           for b in C.BOARD_ORDER}
    tot = dup = 0
    for b in C.BOARD_ORDER:
        ds = [pd.Timestamp(e["date"]) for e in uev[b]]
        for i, d in enumerate(ds):
            if not (W_3[0] <= d <= W_3[1]):
                continue
            tot += 1
            if any(abs((d - o).days) <= 4 and o != d for o in ds):
                dup += 1
    print(f"  近 3 年并集信号 {tot} 次，其中 {dup} 次与同板块另一信号相隔 ≤4 自然日"
          f"（{100*dup/tot:.0f}%）")
    print("  → 这些是同一波下跌的重复计数，页面按信号次数展示时会显得比实际事件多。")


if __name__ == "__main__":
    main()
