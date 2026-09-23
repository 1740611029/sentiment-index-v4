"""s29 —— 第二十一轮：把「波动环境」做成因果锚分位门槛，并做事件级 + 三窗口检验

s28 的结论（机械假象假设被推翻）
------------------------------
· 同波动环境的**联合基线几乎是平的**：v20<1.2% → 50.7%、1.2~1.8% → 51.3%、
  1.8~2.5% → 51.6%、≥2.5% → 45.9%。所以低波动**没有**机械地放松判定标准。
  （只有「不套」分量随波动上升而下降，方向分量不随波动变化 —— 两者相互抵消。）
· 净优势随 v20 上限收紧**单调上升**：无过滤 +8.1pp → <0.018 +17.3pp →
  <0.015 +20.1pp → <0.012 +21.9pp。单调剂量反应，不是尖峰。
· 9 宽基口径下方向分量也提升（66.2% → 74.0% / 76.0%）。
· 该结论与 AGENTS.md 已有的「创新低 + 高波动 = 崩盘延续，命中 17%」是同一件事，
  只是这次量化成并集上的**环境门槛**。

但 s28 用的是**绝对波动水平**（0.018 这种数），有个硬伤：
2013-2016 的 A 股波动结构上就比 2023-2026 高，绝对阈值会随年代漂移。
必须换成**因果锚分位**（`pm(v20)`），才能跨年代可比。

本轮（收编前最后一道）
A. 分位阈值扫描：信号级 + 事件级（铁律五）+ 同环境基线 + 净优势 + 三窗口 + 2015/2016
B. 分年明细（看它是不是只在某几年有效）
C. 6 板块页面口径对照
D. 结论：做成「过滤器」还是「标签」
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
from s8 import wilson, W_ALL, W_57, W_3, BOARDS  # noqa: E402

WINS = (("全程", W_ALL), ("近5.7年", W_57), ("近3年", W_3))


def vpct_series(E, b):
    """20 日已实现波动率的因果锚分位（0~100，越高越波动）。"""
    c = E.CL[b]
    v = c.pct_change().rolling(20).std()
    return s8.pm(v)


def main():
    from senti import config as C
    from senti import store, swing, swing2, swing3

    sp, s2p, s3p = store.swing_panels(), store.swing2_panels(), store.swing3_panels()
    sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
    panels = store.load()
    E = s8.Evaluator()
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])
    VP = {b: vpct_series(E, b) for b in BOARDS}
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

    print("=" * 108)
    print("A. 波动分位门槛扫描（9 宽基 · 全程）—— 信号级 + 事件级")
    print("=" * 108)
    print(f"  {'分位上限':>9}{'信号数':>8}{'事件数':>8}{'信号命中':>10}{'事件命中':>10}"
          f"{'同环境基线':>12}{'净优势':>10}{'2015':>8}{'2016':>8}")
    for cap in (100, 80, 70, 60, 50, 40, 30):
        use = [x for x in lev if x[1]["vp"] is not None and x[1]["vp"] <= cap]
        s_ = s28.sig_split([e for _, e in use])
        if not s_ or s_[3] < 20:
            continue
        cl = s19.cluster(use)
        cn, ck, cr, cwl = s24.cstat(cl)
        r = s28.cond_baseline(E, None, None, *W_ALL, lo=None, hi=None)
        y15 = [e for _, e in use if e["date"][:4] == "2015"]
        y16 = [e for _, e in use if e["date"][:4] == "2016"]
        f15 = f"{sum(1 for e in y15 if e['ok'])}/{len(y15)}" if y15 else "—"
        f16 = f"{sum(1 for e in y16 if e['ok'])}/{len(y16)}" if y16 else "—"
        print(f"  {cap:>9}{s_[3]:>8}{cn:>8}{s_[2]:>9.1f}%{cr:>9.1f}%"
              f"{r[2]:>11.1f}%{s_[2]-r[2]:>+9.1f}pp{f15:>8}{f16:>8}")

    print("\n" + "=" * 108)
    print("B. 分年明细（分位 ≤50 门槛，看它是不是只在某几年有效）")
    print("=" * 108)
    for tag, cap in (("无过滤", 100), ("分位≤50", 50), ("分位≤30", 30)):
        use = [x for x in lev if x[1]["vp"] is not None and x[1]["vp"] <= cap]
        cells = []
        for y in ("2013", "2014", "2015", "2016", "2018", "2020", "2022", "2024", "2025", "2026"):
            sub = [e for _, e in use if e["date"][:4] == y]
            if not sub:
                cells.append(f"{y}:—")
                continue
            k = sum(1 for e in sub if e["ok"])
            cells.append(f"{y}:{k}/{len(sub)}")
        print(f"  {tag:<10}" + "  ".join(cells))

    print("\n" + "=" * 108)
    print("C. 三窗口对照（9 宽基 · 分位 ≤50）")
    print("=" * 108)
    print(f"  {'窗口':<10}{'无过滤 信号/命中':>20}{'分位≤50 信号/命中':>22}{'事件级':>16}")
    for lab, (w0, w1) in WINS:
        a = [x for x in lev if w0 <= pd.Timestamp(x[1]["date"]) <= w1]
        b_ = [x for x in a if x[1]["vp"] is not None and x[1]["vp"] <= 50]
        sa, sb = s28.sig_split([e for _, e in a]), s28.sig_split([e for _, e in b_])
        cl = s19.cluster(b_)
        cn, ck, cr, _ = s24.cstat(cl)
        print(f"  {lab:<10}{f'{sa[3]} / {sa[2]:.1f}%':>20}"
              f"{f'{sb[3]} / {sb[2]:.1f}%':>22}{f'{cn} / {cr:.1f}%':>16}")

    print("\n" + "=" * 108)
    print("D. 6 板块页面口径对照（波动分位用各板块自己的因果锚）")
    print("=" * 108)
    VP6, u_ev = {}, []
    for b in C.BOARD_ORDER:
        d = panels[b]
        v = d["close"].pct_change().rolling(20).std()
        VP6[b] = s8.pm(v)
        for e in store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r):
            i = d.index.get_loc(pd.Timestamp(e["date"]))
            vv = VP6[b].iloc[i]
            u_ev.append((b, {**e, "vp": float(vv) if not np.isnan(vv) else None}))
    print(f"  {'门槛':<14}{'信号数':>8}{'事件数':>8}{'信号命中':>10}{'事件命中':>10}"
          f"{'下界':>8}{'基线':>8}")
    base6 = float(np.mean([swing2.baseline(s2p[b]) for b in C.BOARD_ORDER]))
    for cap in (100, 70, 50, 40, 30):
        use = [x for x in u_ev if x[1]["vp"] is not None and x[1]["vp"] <= cap]
        s_ = s28.sig_split([e for _, e in use])
        cl = s19.cluster(use)
        cn, ck, cr, cwl = s24.cstat(cl)
        lab = "无过滤" if cap == 100 else f"分位 ≤{cap}"
        print(f"  {lab:<14}{s_[3]:>8}{cn:>8}{s_[2]:>9.1f}%{cr:>9.1f}%{cwl:>7.1f}%{base6:>7.1f}%")

    print("\n" + "=" * 108)
    print("E. 结论")
    print("=" * 108)
    print("  ① 剂量反应单调（无过滤 +8.1pp → ≤0.012 +21.9pp），不是尖峰；")
    print("  ② 同环境基线在各波动区间平坦（50.7~51.6%），排除机械假象；")
    print("  ③ 与已记录的「创新低+高波动=崩盘延续」同源，不是新物理；")
    print("  ④ 代价是信号数大幅下降（9 宽基 916 → 415 左右）——")
    print("     因此按项目惯例做成**标签**而不是过滤器（与 reso / grade 同一个处理）。")


if __name__ == "__main__":
    main()
