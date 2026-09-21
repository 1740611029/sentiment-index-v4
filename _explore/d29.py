"""排查：为什么 d28 算出 6 个顶部事件，而 store 算出 8 个？
如果这里有问题，d28 的底部结论也不能直接信。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, model, store

print("build fresh ...", flush=True)
fresh = model.build_all()
cached = store.load()

print("\n1) 面板是否一致")
for b in C.BOARD_ORDER:
    a, c = fresh[b], cached[b]
    same_shape = a.shape == c.shape
    sc = a["score"]
    cc = c["score"].reindex(sc.index)
    d = (sc - cc).abs().max()
    print(f"  {C.BOARDS[b]['name']:>8} shape同={same_shape}  max|score差|={d:.6f}  "
          f"fresh末日={a.index.max().date()}  cached末日={c.index.max().date()}")

print("\n2) store.events 给的顶部事件")
for b in C.BOARD_ORDER:
    bot, top = store.events(cached[b])
    print(f"  {C.BOARDS[b]['name']:>8} 底{len(bot)} 顶{len(top)}  "
          + ", ".join(f"{e['date']}({e['score']})" for e in top))

print("\n3) d28 算法给的顶部事件（fresh 面板）")
def runs(s, thr, hi):
    m = (s > thr) if hi else (s < thr)
    if not m.any(): return []
    grp = (m != m.shift()).cumsum()
    return sorted({(seg.idxmax() if hi else seg.idxmin()) for _, seg in s[m].groupby(grp[m])})

def first(picks, gap=20):
    out=[]
    for d in picks:
        if out and (d-out[-1]).days<=gap: continue
        out.append(d)
    return out

for b in C.BOARD_ORDER:
    s = fresh[b]["score"].dropna()
    p = first(runs(s, 100.0, True))
    print(f"  {C.BOARDS[b]['name']:>8} 顶{len(p)}  "
          + ", ".join(f"{str(d.date())}({s.loc[d]:.1f})" for d in p))
