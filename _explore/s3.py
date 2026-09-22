"""s3 —— 最终参数下的分层表现（阈值22 冷却4）。

看三档：
  全部信号        （覆盖率优先）
  共振 reso≥3     （系统性一起超卖）
  急跌 rush       （波动率处于自身高分位）
  两者同时        （最严）
并给出分年/分板，判断是不是只靠某一年或某一个板块撑着。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C, swing

if __name__ == "__main__":
    panels = swing.build_and_cache(force=True)
    reso = swing.resonance(panels)
    all_ev = []
    for b in C.BOARD_ORDER:
        for e in swing.events(panels[b], reso):
            e["b"] = b
            all_ev.append(e)
    done = [e for e in all_ev if e["ok"] is not None]
    pend = [e for e in all_ev if e["ok"] is None]

    def st(x):
        if not x:
            return "—"
        o = sum(1 for e in x if e["ok"])
        return f"{o}/{len(x)} = {o/len(x)*100:.1f}%"

    print(f"阈值 ≤{swing.THR}　冷却 {swing.COOL}　判定 T+{swing.H} 期末>0 且回撤≥−{swing.TOL:.0%}")
    print(f"  全部信号        {st(done)}   （待验证 {len(pend)}）")
    for k in (2, 3, 4, 5):
        sub = [e for e in done if (e["reso"] or 0) >= k]
        print(f"  共振 reso≥{k}      {st(sub)}")
    print(f"  急跌 rush        {st([e for e in done if e['rush']])}")
    print(f"  急跌+共振≥3      {st([e for e in done if e['rush'] and (e['reso'] or 0) >= 3])}")

    print("\n分年（全部信号）:")
    ys = {}
    for e in done:
        ys.setdefault(e["date"][:4], [0, 0])
        ys[e["date"][:4]][0] += 1; ys[e["date"][:4]][1] += int(e["ok"])
    for y, v in sorted(ys.items()):
        print(f"  {y}: {v[1]}/{v[0]} = {v[1]/v[0]*100:.1f}%")

    print("\n分板（全部信号）:")
    bs = {}
    for e in done:
        bs.setdefault(e["b"], [0, 0]); bs[e["b"]][0] += 1; bs[e["b"]][1] += int(e["ok"])
    for b, v in bs.items():
        print(f"  {C.BOARDS[b]['name']:<9} {v[1]}/{v[0]} = {v[1]/v[0]*100:.1f}%")

    print("\n科创板全部信号（近 6 个月）:")
    for e in sorted([x for x in all_ev if x["b"] == "STAR"], key=lambda x: x["date"]):
        if e["date"] >= "2026-03-01":
            mark = "✔" if e["ok"] else ("待验证" if e["ok"] is None else "✘")
            print(f"  {e['date']}  分值 {e['score']:>5}  急跌={str(e['rush']):<5}"
                  f" 共振={e['reso']}  T+7 {e['ret']}%  {mark}")
