# -*- coding: utf-8 -*-
"""t6: 唯一还活着的杠杆——不新增触发器，而是给现有并集按「几个小波段模型在±4日内
同日/邻域共振」分置信档。看能否在**不减少展示次数**的前提下抬升有效命中率。"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"d:\情绪指标4")
import numpy as np, pandas as pd
from senti import config as C, store, swing, swing2, swing3
sp,s2p,s3p=swing.load(),swing2.load(),swing3.load()
sr,s2r,s3r=swing.resonance(sp),swing2.resonance(s2p),swing3.resonance(s3p)
boards=C.BOARD_ORDER
# 每板块：收集三模型各自信号日期 + ok
m1={b:{e["date"]:e["ok"] for e in swing.events(sp[b],sr)} for b in boards}
m2={b:{e["date"]:e["ok"] for e in swing2.events(s2p[b],s2r)} for b in boards}
m3={b:{e["date"]:e["ok"] for e in swing3.events(s3p[b],s3r)} for b in boards}
mods=[m1,m2,m3]
# 事件级第一枪（并集），再数该事件前后±4日内有几个不同模型参与
uev={b:store.union_events(b,sp[b],s2p[b],s3p[b],sr,s2r,s3r) for b in boards}
def d4(x): return pd.Timestamp(x)
buckets={}
for b in boards:
    ev=store.cluster_events({b:uev[b]})
    for e in ev:
        d=d4(e["date"]); nm=0
        for mi in mods:
            if any(abs((d-d4(dd)).days)<=4 and mi[b].get(dd) is not None for dd in mi[b]):
                nm+=1
        if e["ok"] is None: continue
        buckets.setdefault(nm,[]).append(e["ok"])
print("并集事件级：按「±4日内几个模型参与」分档")
alln=sum(len(v) for v in buckets.values()); allk=sum(sum(v) for v in buckets.values())
print(f"  全部事件 {alln} / {round(allk/alln*100,1)}%")
for nm in sorted(buckets):
    v=buckets[nm]; print(f"  {nm} 个模型共振: {len(v):>3} 件 / {round(sum(v)/len(v)*100,1)}%")
# 累计阈值：>=2 / >=3
for th in (2,3):
    v=[ok for nm,vs in buckets.items() if nm>=th for ok in vs]
    print(f"  >= {th} 模型: {len(v)} 件 / {round(sum(v)/len(v)*100,1)}%  (保留 {len(v)/alln*100:.0f}% 次数)")
# 与已有 reso 档对比（页面已用的横截面共振）
r={}
for b in boards:
    for e in store.cluster_events({b:uev[b]}):
        if e["ok"] is None: continue
        r.setdefault(e.get("reso",0),[]).append(e["ok"])
print("\n对照：现有横截面共振 reso 分档（页面 S/A/B/C 用的是这个）")
for k in sorted(r):
    if r[k]: print(f"  reso={k}: {len(r[k]):>3} 件 / {round(sum(r[k])/len(r[k])*100,1)}%")
