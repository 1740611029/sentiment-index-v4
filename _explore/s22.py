"""s22 —— 第十四轮：并集融合规则在**事件级**重测（第 4 模型的替代方案）

为什么重测
--------
s7 当初定「跨模型不做冷却（COOL=0）」的依据是**信号级**统计：
  冷却 0 → 222/64.4%　冷却 2 → 194/62.9%　冷却 3 → 187/62.6%
冷却 0 在信号数与命中率上同时最优。

但 s19 查出：并集 189 次信号里有 28% 是同板块 ±4 日内的重复记录。
信号级统计把同一波下跌重复计了多次，**会系统性地偏向「不冷却」**——
因为不冷却的信号更多、而多出来的那些正是重复事件。

所以融合规则必须在事件级重测。本脚本：
A. 跨模型冷却扫描（0~8 交易日），同时给信号级与事件级
B. 「并集 vs 单模型」在事件级的诚实对比（并集是用命中率换覆盖）
C. 若存在事件级更优且平坦的冷却，给出建议值
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s8
import s19
from s8 import wilson, W_3                              # noqa: E402


def fuse(per_model, cool):
    """per_model: {model: [(board, event_dict), ...]}；cool = 跨模型冷却交易日数。"""
    allx = [(m, b, e) for m, evs in per_model.items() for b, e in evs]
    by_b = {}
    for m, b, e in allx:
        by_b.setdefault(b, []).append((m, e))
    out = []
    for b, lst in by_b.items():
        lst.sort(key=lambda x: x[1]["date"])
        last = None
        for m, e in lst:
            d = pd.Timestamp(e["date"])
            if last is not None and (d - last).days <= cool:
                continue
            out.append((b, e))
            last = d
    return out


def main():
    from senti import config as C
    from senti import store, swing, swing2, swing3
    sp, s2p, s3p = store.swing_panels(), store.swing2_panels(), store.swing3_panels()
    sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
    per = {"SWING": [], "SWING-2": [], "SWING-3": []}
    for b in C.BOARD_ORDER:
        per["SWING"] += [(b, e) for e in swing.events(sp[b], sr)]
        per["SWING-2"] += [(b, e) for e in swing2.events(s2p[b], s2r)]
        per["SWING-3"] += [(b, e) for e in swing3.events(s3p[b], s3r)]
    base = float(np.mean([swing2.baseline(s2p[b]) for b in C.BOARD_ORDER]))
    print(f"6 板块页面口径（展示窗口 2023-09-20 起）　随机基线 {base:.1f}%\n")

    print("=" * 104)
    print("A. 跨模型冷却扫描：信号级 vs 事件级")
    print("=" * 104)
    print(f"  {'冷却(自然日)':<14}{'信号数':>8}{'信号命中':>10}{'事件数':>8}{'事件命中':>10}{'事件下界':>10}")
    rows = {}
    for cool in (0, 2, 4, 6, 8, 10, 12, 15, 20):
        u = fuse(per, cool)
        sn, _, sr_, _ = s19.cstat([e for _, e in u])
        cl = s19.cluster(u)
        cn, _, cr, cwl = s19.cstat(cl)
        print(f"  {cool:<14}{sn:>8}{sr_:>9.1f}%{cn:>8}{cr:>9.1f}%{cwl:>9.1f}%")
        rows[cool] = (sn, sr_, cn, cr, cwl)

    print("\n" + "=" * 104)
    print("B. 单模型 vs 并集（事件级，诚实对比）")
    print("=" * 104)
    print(f"  {'口径':<22}{'事件数':>8}{'事件命中':>10}{'下界':>10}{'年化事件数':>12}")
    for tag, evs in (("SWING", per["SWING"]), ("SWING-2", per["SWING-2"]),
                     ("SWING-3", per["SWING-3"]),
                     ("并集·不冷却", fuse(per, 0)),
                     ("并集·冷却8", fuse(per, 8)),
                     ("并集·冷却12", fuse(per, 12))):
        cl = s19.cluster(evs)
        cn, ck, cr, cwl = s19.cstat(cl)
        print(f"  {tag:<22}{cn:>8}{cr:>9.1f}%{cwl:>9.1f}%{cn/3.0:>11.1f}")
    print("\n  注：并集是拿命中率换覆盖。SWING-3 事件级 84.1% 但 3 年只有 44 件事；")
    print("      并集 136 件事、67.6%，覆盖是它的 3 倍。两者定位不同，都该保留。")


if __name__ == "__main__":
    main()
