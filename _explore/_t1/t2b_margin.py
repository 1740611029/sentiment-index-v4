# -*- coding: utf-8 -*-
"""t2b: 对「融资分位低」过滤器做严格体检——分半 / 共线 / 次数代价 / 逐年。"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"d:\情绪指标4")
import numpy as np
import pandas as pd
from senti import config as C, store, swing, swing2, swing3

mg = pd.read_parquet(r"d:\情绪指标4\data\cache\market\margin.parquet")
mg["date"] = pd.to_datetime(mg["date"])
bal = mg.set_index("date").sort_index()["margin"].astype(float)
pct = bal.rolling(250, min_periods=120).rank(pct=True).shift(1)

sp, s2p, s3p = swing.load(), swing2.load(), swing3.load()
sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
uev = {b: store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r) for b in C.BOARD_ORDER}
sig = [(b, e) for b, evs in uev.items() for e in evs]

def val(e, s):
    v = s.get(pd.Timestamp(e["date"]), np.nan); return np.nan if v is None else v
def rt(rows):
    d = [(b, e) for b, e in rows if e.get("ok") is not None]
    n = len(d); k = sum(1 for _, e in d if e["ok"])
    return n, (round(k/n*100, 1) if n else None)

print("=== 逐年命中率：融资分位<20% vs 全部 ===")
years = sorted({pd.Timestamp(e["date"]).year for _, e in sig})
for y in years:
    ally = [(b, e) for b, e in sig if pd.Timestamp(e["date"]).year == y]
    low = [(b, e) for b, e in ally if val(e, pct) < 0.20]
    na, ra = rt(ally); nl, rl = rt(low)
    print(f"  {y}: 全部 {na:>3}/{ra}%   分位<20% {nl:>3}/{rl}%")

print("\n=== 分半(以2025-06-01) ===")
mid = pd.Timestamp("2025-06-01")
for pq in (0.20, 0.30):
    for tag, sel in (("前半", lambda d: d < mid), ("后半", lambda d: d >= mid)):
        base = [(b, e) for b, e in sig if sel(pd.Timestamp(e["date"]))]
        low = [(b, e) for b, e in base if val(e, pct) < pq]
        nb, rb = rt(base); nl, rl = rt(low)
        print(f"  分位<{int(pq*100)}% {tag}: 不过滤 {nb}/{rb}%   过滤后 {nl}/{rl}%")

print("\n=== 与「共振」共线检查：低融资分位是否只是挑出高共振信号 ===")
low_sig = [(b, e) for b, e in sig if val(e, pct) < 0.20]
def avgreso(rows):
    r = [e["reso"] for _, e in rows if e.get("reso") is not None]
    return round(float(np.mean(r)), 2) if r else None
print("  全部并集 平均共振:", avgreso(sig), " 融资分位<20% 平均共振:", avgreso(low_sig))

print("\n=== 次数代价(事件级) ===")
all_ev = store.cluster_events(uev)
low_by = {}
for b, evs in uev.items():
    low_by[b] = [e for e in evs if val(e, pct) < 0.20]
low_ev = store.cluster_events(low_by)
print(f"  事件级: 不过滤 {len(all_ev)} 件 → 过滤后 {len(low_ev)} 件  (保留 {len(low_ev)/len(all_ev)*100:.0f}%)")

# margin 近端覆盖
dts = pd.Timestamp("2026-09-17")
recent = [(b, e) for b, e in sig if pd.Timestamp(e["date"]) > dts]
print(f"\n  margin止于2026-09-17；之后并集还有 {len(recent)} 个信号会被判为 NaN→丢弃")
