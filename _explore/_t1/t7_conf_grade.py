# -*- coding: utf-8 -*-
"""t7: 置信分级前沿。全部并集信号保留展示，只贴「当下可得」的因果标签，
看哪种(些)轴能把高置信档的命中率抬到 75~80%+，且档内事件数别太少。

候选轴（都是信号日收盘即可算出，无未来函数）：
  ures  横截面广度：当日有多少个板块出现任一模型信号（store.union_resonance）
  reso  触发模型自身的横截面共振（几个板块同时 ≤ 该模型阈值）
  dep   触发分值深度（score 越低越极端）
  nmod  本板块 ±4 日内有几个不同模型参与（t6 已知平）
  mgn   融资余额 250 日因果分位（低=杠杆出清，但可能是时代代理）
"""
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

m1 = {b: {e["date"]: 1 for e in swing.events(sp[b], sr)} for b in boards}
m2 = {b: {e["date"]: 1 for e in swing2.events(s2p[b], s2r)} for b in boards}
m3 = {b: {e["date"]: 1 for e in swing3.events(s3p[b], s3r)} for b in boards}
def n_models(b, d):
    dd = pd.Timestamp(d); c = 0
    for mi in (m1, m2, m3):
        if any(abs((dd - pd.Timestamp(x)).days) <= 4 for x in mi[b]): c += 1
    return c

mg = pd.read_parquet(r"d:\情绪指标4\data\cache\market\margin.parquet")
mg["date"] = pd.to_datetime(mg["date"])
bal = mg.set_index("date").sort_index()["margin"].astype(float)
mpct = bal.rolling(250, min_periods=120).rank(pct=True).shift(1)

# 组装事件级样本（第一枪）
rows = []
for b in boards:
    for e in store.cluster_events({b: uev[b]}):
        if e["ok"] is None: continue
        d = pd.Timestamp(e["date"])
        rows.append(dict(board=b, date=e["date"], ok=bool(e["ok"]),
                         reso=e.get("reso"), ures=ures.get(e["date"], 1),
                         dep=e.get("score"), nmod=n_models(b, e["date"]),
                         mgn=mpct.get(d, np.nan)))
df = pd.DataFrame(rows)
N = len(df); BASE = df["ok"].mean()*100
print(f"并集事件级样本 {N} 件，命中基线 {BASE:.1f}%\n")

def grade(col, edges, labels=None):
    print(f"—— 按 {col} 分档 ——")
    g = pd.cut(df[col], edges, labels=labels, include_lowest=True) if edges[0] != "cat" else df[col]
    t = df.groupby(g, observed=True)["ok"].agg(["count", "mean"])
    for idx, r in t.iterrows():
        print(f"   {str(idx):<14} {int(r['count']):>3} 件  {r['mean']*100:>5.1f}%")
    print()

grade("reso", ["cat", range(0, 7)])
grade("ures", [0, 1, 2, 3, 4, 5, 6], ["1", "2", "3", "4", "5", "6"])
grade("nmod", ["cat", [1, 2, 3]])
grade("dep", [-1, 3, 6, 10, 15, 101], ["<=3", "3-6", "6-10", "10-15", ">15"])
print("—— 融资分位 mgn 分档（时代代理风险）——")
mm = df.dropna(subset=["mgn"])
t = mm.groupby(pd.cut(mm["mgn"], [0, .2, .4, .6, .8, 1.0]), observed=True)["ok"].agg(["count", "mean"])
for idx, r in t.iterrows(): print(f"   {str(idx):<14} {int(r['count']):>3} 件  {r['mean']*100:>5.1f}%")

print("\n=== 复合档位（找 >=75% 且件数可观 的组合）===")
def sub(mask, label):
    s = df[mask]; n = len(s)
    if n == 0: print(f"   {label:<40} 0 件"); return
    print(f"   {label:<40} {n:>3} 件  {s['ok'].mean()*100:>5.1f}%")
for R in (3, 4):
    for extra, lab in [(df.dep <= 6, "& 分值<=6"), (df.dep <= 3, "& 分值<=3")]:
        sub((df.reso >= R) & extra, f"reso>={R} {lab}")
sub((df.ures >= 4) & (df.dep <= 6), "ures>=4 & 分值<=6")
sub((df.reso >= 4) & (df.nmod >= 2), "reso>=4 & 多模型")
sub((df.reso >= 3) & (df.dep <= 5), "reso>=3 & 分值<=5")
sub((df.ures >= 5), "ures>=5(广共振)")
sub((df.ures >= 5) & (df.dep <= 6), "ures>=5 & 分值<=6")

print("\n=== 覆盖 vs 命中 前沿（按某评分排序取 top-k%）===")
# 简单合成分：reso 权重高、分值越深越好
df["z"] = df.reso.fillna(0) - df.dep/20
for pct in (20, 30, 40, 50, 60):
    k = int(N*pct/100); top = df.nlargest(k, "z")
    print(f"   top {pct}% ({k}件) by [reso − 分值/20]: {top['ok'].mean()*100:.1f}%")

print("\n=== 分半复核：ures>=4 & 分值<=6 与 reso>=4 ===")
mid = pd.Timestamp("2025-06-01")
dtp = pd.to_datetime(df.date)
for lab, mask in [("ures>=4&dep<=6", (df.ures >= 4) & (df.dep <= 6)),
                  ("reso>=4", df.reso >= 4)]:
    for tag, sel in (("前半", dtp < mid), ("后半", dtp >= mid)):
        s = df[mask & sel]; b = df[sel]
        print(f"   {lab} {tag}: {len(s)}件/{(s['ok'].mean()*100 if len(s) else None)}%  (该半区全部{len(b)}/{b['ok'].mean()*100:.1f}%)")

print("\n=== 2026 年（近端）各高置信档还剩多少 ===")
d26 = df[pd.to_datetime(df.date).dt.year == 2026]
print(f"   2026 全部 {len(d26)}件/{d26['ok'].mean()*100:.1f}%")
print(f"   2026 ures>=4&dep<=6: {len(d26[(d26.ures>=4)&(d26.dep<=6)])}件")
print(f"   2026 reso>=4: {len(d26[d26.reso>=4])}件  |  mgn<0.2:{len(d26[d26.mgn<0.2])}件(融资档在2026是否可用)")
