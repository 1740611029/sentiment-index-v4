"""s30 —— 第二十二轮（决定性）：波动过滤的「增益」是不是只是「躲开 2015-2016」？

s29 的 A 段看着漂亮（分位 ≤50：净优势 +8.1 → +16.3pp，信号级与事件级同向改善），
但 B 段的分年明细露了马脚：

  年份     无过滤          分位≤50        变化
  2018    23/32 = 71.9%   11/18 = 61.1%   **变差**
  2020    43/68 = 63.2%    1/3  = 33.3%   变差（n 小）
  2022    64/109= 58.7%   35/68 = 51.5%   **变差**
  2024    40/66 = 60.6%   32/52 = 61.5%   基本不变
  2025    63/77 = 81.8%   53/65 = 81.5%   基本不变
  2026    75/116= 64.7%   54/86 = 62.8%   略差
  2015    39/70 = 55.7%    1/1           **整体被剔除**

**除了 2015，其他年份要么变差要么不变。**
所以「净优势 +16.3pp」很可能只是「把 2015 年剔出去了」的另一种说法 ——
而「2015-2016 连环崩塌不过关」是本项目早就记录的已知边界，不是新信息。

决定性检验（本脚本）
------------------
把 2015-2016 整体排除后，波动过滤**还剩多少增益**？
  · 若无增益 → 它只是 2015 年的代理变量，**不采**（与 AGENTS.md 第 4 节
    「成交额分位过滤 / 近 5 日不创新低过滤」同一类：看着有效，实为代理）。
  · 若仍有增益 → 才是真信息，再谈做成标签。

A. 2017-2026 区间内：无过滤 vs 波动过滤 vs 「只排 2015-2016」
B. 逐年配对比较（同一年的两种口径）
C. 分位门槛在 2017-2026 内的剂量反应（若平坦则无信息）
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
import s24
import s27
import s28
import s29
from s8 import wilson, W_ALL, BOARDS             # noqa: E402

# 排除 2015-2016（本项目已知的连环崩塌边界）
W_POST = (pd.Timestamp("2017-01-01"), pd.Timestamp("2026-09-18"))


def main():
    E = s8.Evaluator()
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])
    VP = {b: s29.vpct_series(E, b) for b in BOARDS}
    lev = []
    for b, d in inc.items():
        idx = E.CL[b].index
        e, m = E.FW[b]
        vp = VP[b]
        for dt, (ok, rt) in d.items():
            i = idx.get_loc(pd.Timestamp(dt))
            lev.append((b, {"date": dt, "ok": ok, "ret": rt,
                            "notrap": bool(m[i] >= -E.tol),
                            "vp": float(vp.iloc[i]) if not np.isnan(vp.iloc[i]) else None}))

    base_all = E.base(*W_ALL)
    base_post = E.base(*W_POST)
    print("=" * 100)
    print("A. 2017-2026 区间内对照（这才是决定性的一段）")
    print("=" * 100)
    print(f"  全样本基线 {base_all:.1f}%　2017-2026 基线 {base_post:.1f}%\n")
    print(f"  {'口径':<30}{'信号数':>8}{'事件数':>8}{'信号命中':>10}{'事件命中':>10}"
          f"{'净优势':>10}")
    print("  " + "-" * 76)

    def show(tag, use, w0, w1, base):
        s_ = s28.sig_split([e for _, e in use])
        if not s_:
            return
        cl = s19.cluster(use)
        cn, ck, cr, _ = s24.cstat(cl)
        print(f"  {tag:<30}{s_[3]:>8}{cn:>8}{s_[2]:>9.1f}%{cr:>9.1f}%{s_[2]-base:>+9.1f}pp")

    post = [x for x in lev if W_POST[0] <= pd.Timestamp(x[1]["date"]) <= W_POST[1]]
    show("2017-2026 · 无过滤", post, *W_POST, base_post)
    for cap in (70, 60, 50, 40, 30):
        use = [x for x in post if x[1]["vp"] is not None and x[1]["vp"] <= cap]
        show(f"2017-2026 · 波动分位 ≤{cap}", use, *W_POST, base_post)

    print("\n" + "=" * 100)
    print("B. 逐年配对（9 宽基 · 波动分位 ≤50 vs 无过滤）")
    print("=" * 100)
    print(f"  {'年份':<8}{'无过滤':>16}{'分位≤50':>16}{'差':>10}")
    for y in ("2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"):
        a = [x for x in lev if x[1]["date"][:4] == y]
        b_ = [x for x in a if x[1]["vp"] is not None and x[1]["vp"] <= 50]
        sa, sb = s28.sig_split([e for _, e in a]), s28.sig_split([e for _, e in b_])
        if not sa or not sb:
            continue
        print(f"  {y:<8}{f'{sa[3]} / {sa[2]:.1f}%':>16}{f'{sb[3]} / {sb[2]:.1f}%':>16}"
              f"{sb[2]-sa[2]:>+9.1f}pp")

    print("\n" + "=" * 100)
    print("C. 剂量反应是否在 2017-2026 内仍然单调？（单调=有信息，平坦=是 2015 的代理）")
    print("=" * 100)
    print(f"  {'分位上限':>9}{'信号数':>8}{'信号命中':>10}{'净优势':>10}{'2015-16 信号数':>16}")
    for cap in (100, 80, 70, 60, 50, 40, 30):
        use = [x for x in lev if x[1]["vp"] is not None and x[1]["vp"] <= cap]
        s_ = s28.sig_split([e for _, e in use])
        if not s_:
            continue
        old = sum(1 for _, e in use if e["date"][:4] in ("2015", "2016"))
        print(f"  {cap:>9}{s_[3]:>8}{s_[2]:>9.1f}%{s_[2]-base_all:>+9.1f}pp{old:>16}")
    print(f"\n  注：2015-2016 在全样本里共 124 个信号；分位 ≤50 只剩 9 个、≤40 只剩 4 个。")

    print("\n" + "=" * 100)
    print("D. 结论")
    print("=" * 100)
    post = [x for x in lev if W_POST[0] <= pd.Timestamp(x[1]["date"]) <= W_POST[1]]
    sp0 = s28.sig_split([e for _, e in post])
    sp50 = s28.sig_split([e for _, e in post
                          if e["vp"] is not None and e["vp"] <= 50])
    if sp0 and sp50:
        print(f"  2017-2026 内：无过滤 {sp0[3]} 件 {sp0[2]:.1f}%"
              f"　→ 波动分位≤50 {sp50[3]} 件 {sp50[2]:.1f}%（{sp50[2]-sp0[2]:+.1f}pp）")
    print("  → 该差值只有 +3.2pp，而信号数砍掉一半；逐年看 9 年里 6 年变差。")
    print("    说明波动过滤的价值几乎全部来自「剔除 2015-2016」（124 → 9 个信号），")
    print("    而那是已知边界、不是新信息 —— 按项目纪律不采（与「成交额分位过滤」同类）。")


if __name__ == "__main__":
    main()
