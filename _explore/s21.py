"""s21 —— 第十三轮：并集层面的参数稳健性（决定性）

s20 的两个矛盾信号
----------------
  支持收编：9 宽基长历史里，把 rsi_slope≤2+vconv 加进三模型并集，
            全程 58.7→59.3%、近5.7年 67.9→68.1%、近3年 71.1→71.8%，
            且 2020/2022/2024/2025 四个年度都提升；6 板块近3年
            信号级 70.9→72.1%、事件级 67.6→69.1%。
  反对收编：它**单独**用的 89.5% 是刀刃不是平台 —— vconv 窗口
            (5,20)→89.5%，(5,30)→77.8%，(10,30)→47.4%（掉到基线以下）。

本项目纪律（d23/d26/d33）：**plateau 不是尖峰**。所以必须把并集层面的
增益也做同样的窗口扰动。若并集增益随 vconv 窗口崩塌，就是过拟合，不能收编；
若并集增益在窗口族上平坦，则收编（但**不宣传**它单独用的高命中率）。

注意：6 板块面板只从 BACKTEST_START(2023-09-20) 起（swing*.py 里做了截断），
所以页面口径只能比近 3 年；长历史必须在 9 宽基口径上做。
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
from s8 import wilson, W_ALL, W_57, W_3, BOARDS  # noqa: E402

WINS = (("全程", W_ALL), ("近5.7年", W_57), ("近3年", W_3))
VARIANTS = [("5/20", 5, 20), ("5/30", 5, 30), ("8/25", 8, 25),
            ("10/30", 10, 30), ("3/20", 3, 20), ("5/60", 5, 60)]


def main():
    E = s8.Evaluator()
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])
    ref = {lab: s19.rate(inc, w0, w1)[2] for lab, (w0, w1) in WINS}
    print("9 宽基口径（长历史）")
    print(f"  {'vconv窗':>9}{'阈值':>5}{'全程':>16}{'近5.7年':>16}{'近3年':>16}{'2015':>9}")

    def fy(u, y):
        n = k = 0
        for b, d in u.items():
            for dt, (ok, _) in d.items():
                if dt[:4] == y:
                    n += 1
                    k += int(ok)
        return f"{k}/{n}"

    print(f"  {'【基准】':>9}{'':>5}"
          + "".join(f"{f'{ref[lab]:.1f}%':>16}" for lab, _ in WINS)
          + f"{fy(inc, '2015'):>9}")
    rows = {}
    for tag, ks, kl in VARIANTS:
        for t in (1, 2, 3, 4):
            S = E.score("rsi_slope")
            cand = s17.sig_dates(E, S, t, 4, s12.mk_vconv(ks, kl))
            u4 = s18.merge_all(inc, cand)
            cells = []
            delta = []
            for lab, (w0, w1) in WINS:
                r = s19.rate(u4, w0, w1)[2]
                cells.append(f"{r:5.1f}%")
                delta.append(r - ref[lab])
            print(f"  {tag:>9}{t:>5}" + "".join(f"{c:>16}" for c in cells)
                  + f"{fy(u4, '2015'):>9}")
            rows[(tag, t)] = (min(delta), np.mean(delta))

    print("\n  稳健性汇总（相对基准的增益 pp）")
    ok = [k for k, v in rows.items() if v[0] >= -0.2]
    print(f"  三个窗口增益都 ≥ −0.2pp 的变体：{len(ok)}/{len(rows)}")
    for k in sorted(rows, key=lambda x: -rows[x][1])[:8]:
        print(f"    {k[0]:>6} 阈值{k[1]}　最小增益 {rows[k][0]:+.1f}pp"
              f"　平均 {rows[k][1]:+.1f}pp")

    print("\n" + "=" * 110)
    print("6 板块页面口径（近 3 年，只能比这一窗）")
    print("=" * 110)
    from senti import config as C
    from senti import store, swing, swing2, swing3
    sp, s2p, s3p = store.swing_panels(), store.swing2_panels(), store.swing3_panels()
    sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
    E6 = s12.Ev6()
    u3 = [(b, e) for b in C.BOARD_ORDER
          for e in store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r)]
    sn3, _, sr3, _ = s19.cstat([e for _, e in u3])
    cl3 = s19.cluster(u3)
    cn3, _, cr3, cwl3 = s19.cstat(cl3)
    print(f"  【基准】并集(3 模型)　信号 {sn3} 次 {sr3:.1f}%　"
          f"事件 {cn3} 件 {cr3:.1f}%（下界 {cwl3:.1f}%）")
    print(f"  {'vconv窗':>9}{'阈值':>5}{'信号数':>8}{'事件数':>8}{'信号命中':>10}{'事件命中':>10}")
    for tag, ks, kl in VARIANTS:
        for t in (1, 2, 3):
            ev = E6.run(E6.score("rsi_slope"), t, 4, *W_3,
                        cond=s12.mk_vconv(ks, kl))
            c4 = [(b, {"date": d, "ok": ok}) for b, d, ok, _ in ev]
            seen = {(b, e["date"]) for b, e in u3}
            u4 = u3 + [x for x in c4 if (x[0], x[1]["date"]) not in seen]
            sn, _, srr, _ = s19.cstat([e for _, e in u4])
            cl = s19.cluster(u4)
            cnn, _, crr, _ = s19.cstat(cl)
            print(f"  {tag:>9}{t:>5}{sn:>8}{cnn:>8}{srr:>9.1f}%{crr:>9.1f}%")


if __name__ == "__main__":
    main()
