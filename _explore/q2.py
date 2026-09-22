"""SWING 扩展窗口（2021-01 起）样本量复核。"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C, swing

C.BACKTEST_START = "2021-01-01"
panels = swing.build_all()
n0 = {b: len(p) for b, p in panels.items()}
print("面板(2021-01起):", min(p.index.min() for p in panels.values()).date(),
      "~", max(p.index.max() for p in panels.values()).date(), "行数", n0)

reso = swing.resonance(panels)
rows = []
for b in C.BOARD_ORDER:
    for e in swing.events(panels[b], reso):
        e["board"] = b; rows.append(e)
done = [e for e in rows if e["ok"] is not None]
pend = [e for e in rows if e["ok"] is None]

W3 = ("2023-09-20", "2026-09-18")
W5 = ("2021-01-01", "2026-09-18")
for tag, (w0, w1) in (("近3年", W3), ("近5.7年", W5)):
    sub = [e for e in done if w0 <= e["date"] <= w1]
    for lab, lo, hi in (("全部", 0, 9), ("共振>=4", 4, 9), ("共振>=5", 5, 9), ("共振=6", 6, 9)):
        s2 = [e for e in sub if lo <= (e["reso"] or 0) <= hi]
        if lab == "共振=6":
            s2 = [e for e in sub if (e["reso"] or 0) >= 6]
        if s2:
            print(f"{tag:<8}{lab:<8} n={len(s2):>3} 命中 {sum(1 for e in s2 if e['ok']):>3} "
                  f"= {sum(1 for e in s2 if e['ok'])/len(s2)*100:.1f}%")
    print()
