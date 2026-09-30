# -*- coding: utf-8 -*-
"""t8: 给 A 方案定稿——(a)确认「共振+分值深度」是平台不是刀刃；
(b)和页面现役分级(只看 reso)对比，证明有增量；(c)给出建议档位表。"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"d:\情绪指标4")
import numpy as np, pandas as pd
from senti import config as C, store, swing, swing2, swing3

boards = C.BOARD_ORDER
sp, s2p, s3p = swing.load(), swing2.load(), swing3.load()
sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
uev = {b: store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r) for b in boards}
ures = store.union_resonance(uev)
rows = []
for b in boards:
    for e in store.cluster_events({b: uev[b]}):
        if e["ok"] is None: continue
        rows.append(dict(date=e["date"], ok=bool(e["ok"]), reso=e.get("reso") or 0,
                         ures=ures.get(e["date"], 1), dep=e.get("score") or 99))
df = pd.DataFrame(rows); df["yr"] = pd.to_datetime(df.date).dt.year
N = len(df); BASE = df.ok.mean()*100
print(f"事件级 {N} 件 / 基线 {BASE:.1f}%\n")

print("=== 平台检验：reso×dep 网格（件数 / 命中率），找稳定 >=80% 的区域 ===")
print("        dep<=4  dep<=5  dep<=6  dep<=7  dep<=8  dep<=10")
for R in (2, 3, 4, 5):
    line = f"  reso>={R} "
    for D in (4, 5, 6, 7, 8, 10):
        s = df[(df.reso >= R) & (df.dep <= D)]
        line += f" {len(s):>2}/{(s.ok.mean()*100 if len(s) else 0):>4.0f}%"
    print(line)

print("\n=== 同样网格但用 ures（并集广度，页面当前没用） ===")
for R in (2, 3, 4, 5):
    line = f"  ures>={R} "
    for D in (5, 6, 7, 8, 10):
        s = df[(df.ures >= R) & (df.dep <= D)]
        line += f" {len(s):>2}/{(s.ok.mean()*100 if len(s) else 0):>4.0f}%"
    print(line)

print("\n=== 现役分级(只看 reso, 页面 S/A/B/C) vs 建议(reso+dep) 的覆盖-命中对比 ===")
def curve(mask_fn, label):
    pts = []
    # 逐步放宽
    for R in (6, 5, 4, 3, 2, 1):
        for D in (4, 6, 8, 12, 22, 101):
            s = df[mask_fn(R, D)]
            if len(s) >= 15:
                pts.append((len(s), s.ok.mean()*100)); break
    print(f"  {label}: " + "  ".join(f"{n}件/{r:.0f}%" for n, r in pts))
curve(lambda R, D: (df.reso >= R) & (df.dep <= 101), "现役 仅 reso")
curve(lambda R, D: (df.reso >= R) & (df.dep <= D), "建议 reso+dep")

# 直接对比同等覆盖下谁命中高
print("\n  等覆盖对比（都取 ~40 件）:")
a = df.sort_values(["reso", "dep"], ascending=[False, True]).head(40)
b = df[(df.reso >= 3) & (df.dep <= 6)]
b = b.sort_values(["reso", "dep"], ascending=[False, True]).head(40)
print(f"    仅按 reso 取前40:      {a.ok.mean()*100:.1f}%")
print(f"    reso+dep 联合取≤40:    {b.ok.mean()*100:.1f}%  ({len(b)}件)")

print("\n=== 2026 近端可用性（各建议档还剩几件、命中） ===")
for (R, D) in [(4, 101), (3, 6), (4, 6), (4, 8)]:
    s = df[(df.reso >= R) & (df.dep <= D)]
    s26 = s[s.yr == 2026]
    tag = "reso" if D > 100 else f"reso&dep<={D}"
    print(f"  {tag} >={R}: 全程{len(s)}件/{s.ok.mean()*100 if len(s) else 0:.0f}%  2026 {len(s26)}件/{s26.ok.mean()*100 if len(s26) else 0:.0f}%")

print("\n=== 建议档位表（事件级，可直接映射到页面 A/B/C 文案） ===")
T = [("S*", (df.reso >= 5)),
     ("A", (df.reso >= 4) & (df.dep <= 8)),
     ("B", ((df.reso >= 3) & (df.dep <= 6)) & ~((df.reso >= 4) & (df.dep <= 8))),
     ("C", ~((df.reso >= 3) & (df.dep <= 6)) & ~(df.reso >= 5))]
for name, m in T:
    s = df[m]; print(f"  {name:<3} {len(s):>3} 件  {s.ok.mean()*100:>5.1f}%")
