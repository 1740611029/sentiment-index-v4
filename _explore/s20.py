"""s20 —— 第十二轮：rsi_slope≤2 + vconv 到底该不该作为第 4 个模型收编？

s17~s19 的推理链
--------------
s17  1288 组候选里 35 组「过关」（三窗口都不劣、至少一窗更高）→ 看似该加
s18  ±3 日邻域合并后只剩 15 组，分半检验后 12 组，其中 9 组增益只是加了 1~4 个信号
     （58.7% → 58.8%）→ 判定为**噪声**，是同一波下跌的重复计数
s19  唯一有真实量级的是 rsi_slope + vconv：加进 9 宽基并集后
       全程 58.7 → 60.3%（+1.6pp）　近5.7年 67.9 → 68.9%　近3年 71.1 → 71.8%
     而且把邻域窗口放宽到 ±10 日，新增信号只从 102 掉到 89
     → **不是重复计数，是真的新事件**。分半检验两半都为正。

本轮做收编前的最后四道关
--------------------
A. 2015 / 2016 那一关（本项目唯一的历史否决关）
B. 6 板块页面口径：信号级 + **事件级**（同板块 ±4 自然日只算第一枪）
C. 参数平台（阈值 × vconv 窗口 × 冷却）
D. 结论：收编 / 否决
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
CAND = ("rsi_slope", 2, "vconv")     # s19 显示 thr 1 与 2 结果相同，取 2（平台中点）


def byyear(uni, y):
    n = k = 0
    for b, d in uni.items():
        for dt, (ok, _) in d.items():
            if dt[:4] == y:
                n += 1
                k += int(ok)
    return k, n


def main():
    E = s8.Evaluator()
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])
    nm, thr, cn = CAND
    cand = s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
    uni4 = s18.merge_all(inc, cand)

    print("=" * 110)
    print("A. 9 宽基长历史：2015 / 2016 关 + 分年（全程窗口）")
    print("=" * 110)
    print(f"  {'口径':<26}{'2015':>10}{'2016':>10}{'2020':>10}{'2022':>10}{'2024':>10}{'2025':>10}")
    for tag, u in (("并集(3 模型)", inc), ("并集(3) + rsi_slope+vconv", uni4)):
        cells = [f"{byyear(u, y)[0]}/{byyear(u, y)[1]}" for y in
                 ("2015", "2016", "2020", "2022", "2024", "2025")]
        print(f"  {tag:<26}" + "".join(f"{c:>10}" for c in cells))
    print()
    print(f"  {'口径':<26}{'全程':>14}{'近5.7年':>14}{'近3年':>14}")
    for tag, u in (("并集(3 模型)", inc), ("并集(3) + rsi_slope+vconv", uni4)):
        cells = []
        for lab, (w0, w1) in WINS:
            n, k, r = s19.rate(u, w0, w1)
            cells.append(f"{r:5.1f}%({n:>3})")
        print(f"  {tag:<26}" + "".join(f"{c:>14}" for c in cells))

    print("\n" + "=" * 110)
    print("B. 6 板块页面口径（信号级 vs 事件级）")
    print("=" * 110)
    from senti import config as C
    from senti import store, swing, swing2, swing3
    sp, s2p, s3p = store.swing_panels(), store.swing2_panels(), store.swing3_panels()
    sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
    base = float(np.mean([swing2.baseline(s2p[b]) for b in C.BOARD_ORDER]))

    E6 = s12.Ev6()
    for wtag, W in (("近3年", W_3), ("2019起全区间", (pd.Timestamp("2019-01-01"), W_3[1]))):
        print(f"\n  ── {wtag}（随机基线 {E6.base(*W):.1f}%）")
        print(f"  {'口径':<28}{'信号数':>8}{'事件数':>8}{'重复率':>8}"
              f"{'信号命中':>10}{'事件命中':>10}{'下界':>9}")
        print("  " + "-" * 88)
        for b in C.BOARD_ORDER:
            u = store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r)
            pass
        # 三模型并集
        u3 = [(b, e) for b in C.BOARD_ORDER
              for e in store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r)
              if W[0] <= pd.Timestamp(e["date"]) <= W[1]]
        # 第 4 模型
        c4 = E6.run(E6.score(CAND[0]), CAND[1], 4, *W, cond=s12.mk_vconv())
        c4 = [(b, {"date": d, "ok": ok}) for b, d, ok, _ in c4]
        seen = {(b, e["date"]) for b, e in u3}
        u4 = u3 + [x for x in c4 if (x[0], x[1]["date"]) not in seen]

        for tag, evs in (("并集(3 模型)", u3), ("并集(3) + rsi_slope+vconv", u4)):
            sn, sk, sr_, _ = s19.cstat([e for _, e in evs])
            cl = s19.cluster(evs)
            cnn, ck, cr, cwl = s19.cstat(cl)
            dup = 100 * (1 - cnn / sn) if sn else 0
            print(f"  {tag:<28}{sn:>8}{cnn:>8}{dup:>7.0f}%"
                  f"{sr_:>9.1f}%{cr:>9.1f}%{cwl:>8.1f}%")

    print("\n" + "=" * 110)
    print("C. 参数平台（6 板块 · 近 3 年 · 事件级命中率）")
    print("=" * 110)
    print(f"  {'阈值':>5}{'vconv窗':>9}{'冷却':>5}{'信号数':>8}{'事件数':>8}{'信号命中':>10}{'事件命中':>10}")
    for t in (1, 2, 3, 4):
        for ks, kl in ((5, 20), (5, 30), (10, 30)):
            ev = E6.run(E6.score("rsi_slope"), t, 4, *W_3,
                        cond=s12.mk_vconv(ks, kl))
            evd = [(b, {"date": d, "ok": ok}) for b, d, ok, _ in ev]
            sn, sk, sr_, _ = s19.cstat([e for _, e in evd])
            cl = s19.cluster(evd)
            cn, ck, cr, _ = s19.cstat(cl)
            print(f"  {t:>5}{f'{ks}/{kl}':>9}{4:>5}{sn:>8}{cn:>8}{sr_:>9.1f}%{cr:>9.1f}%")

    print("\n" + "=" * 110)
    print("D. 判定")
    print("=" * 110)
    k15i, n15i = byyear(inc, "2015")
    k15c, n15c = byyear(uni4, "2015")
    print(f"  ① 2015 关：并集(3) {k15i}/{n15i}　加入后 {k15c}/{n15c}"
          f"　{'✔ 未劣化' if k15c / max(n15c, 1) >= k15i / max(n15i, 1) - 1e-9 else '✘ 劣化'}")
    print("  ② 6 板块事件级是否提升：见 B 段")


if __name__ == "__main__":
    main()
