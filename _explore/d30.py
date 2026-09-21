"""定位 d28 顶部少算 2 个事件的原因。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from senti import config as C, model

panels = model.build_all()
GAP = 20

def runs(s, thr, hi):
    m = (s > thr) if hi else (s < thr)
    if not m.any(): return []
    grp = (m != m.shift()).cumsum()
    return sorted({(seg.idxmax() if hi else seg.idxmin()) for _, seg in s[m].groupby(grp[m])})

def cluster(picks, gap=GAP):
    out=[]
    for d in picks:
        if out and (d-out[-1][-1]).days <= gap: out[-1].append(d)
        else: out.append([d])
    return out

def pick_first(picks, s, hi):
    return [c[0] for c in cluster(picks)]

for b in C.BOARD_ORDER:
    p = panels[b]; s = p["score"].dropna()
    m = s > 100.0
    r = runs(s, 100.0, True)
    pf = pick_first(r, s, True)
    if len(pf) != len(r):
        print(f"  {C.BOARDS[b]['name']}: runs={len(r)} -> pick_first={len(pf)}  合并掉了")
    print(f"  {C.BOARDS[b]['name']:>8}  破100天数={int(m.sum()):>3}  runs={len(r)}  "
          f"pick_first={len(pf)}  " + ", ".join(f"{str(d.date())}({s.loc[d]:.1f})" for d in pf))

print("\n检查 2024-10-08 与 2024-11-11 之间是否还有别的破 100 日（会被 20 天 gap 合并）：")
for b in ("CSI2000", "CSI1000"):
    s = panels[b]["score"].dropna()
    seg = s[(s.index >= pd.Timestamp("2024-10-01")) & (s.index <= pd.Timestamp("2024-11-30"))]
    over = seg[seg > 100.0]
    print(f"  {b}: 破100的日期 = " + ", ".join(
        f"{str(d.date())}:{v:.1f}" for d, v in over.items()))
    print(f"     10-08 -> 11-11 相差 {(pd.Timestamp('2024-11-11') - pd.Timestamp('2024-10-08')).days} 天")
