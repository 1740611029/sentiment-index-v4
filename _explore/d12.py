"""诊断 12：标定最终模型参数（确认项强度 K、增益 g），寻找「事件少 + 命中高」的组合。

最终模型 SENTI-1（单一，全板块统一）：
  L  情绪水平(广度口径) = Σ w·pct_map(因子)         0~100
  T  过热水平(价格口径) = 0.30·bias + 0.30·ret20 + 0.22·rsi + 0.18·amt
  U  = 0.5·L + 0.5·T
  cf_b 底部确认 = 超跌程度 × 放量程度      (0~1)
  cf_t 顶部确认 = 过热程度 × 收阴程度      (0~1)
  score = 50 + g·(U + K·cf_t − K·cf_b − 50)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors

big = data.build_stock_indicators()
RAW = {}
for b in C.BOARD_ORDER:
    r = factors.build_board_raw(b, big)
    r = r[r.index >= pd.Timestamp(C.BACKTEST_START)].copy()
    d = data.load_index(b).set_index("date")
    for col in ("open","high","low"): r[col] = d[col].reindex(r.index)
    RAW[b] = r

LEVEL_F=["b20","b60","r5","nh","lim","rsi","bias","ret20"]
W={"b20":.18,"b60":.15,"r5":.11,"nh":.13,"lim":.09,"rsi":.14,"bias":.12,"ret20":.08}
def pmap(v,lo_q=2,hi_q=98):
    lo=np.nanpercentile(v.values,lo_q); hi=np.nanpercentile(v.values,hi_q)
    return (100.0*(v-lo)/(hi-lo)).clip(-25,125)

for b in C.BOARD_ORDER:
    r=RAW[b]
    s=pd.Series(0.0,index=r.index); tw=0.0
    for k in LEVEL_F: s=s+pmap(r[k].astype(float))*W[k]; tw+=W[k]
    r["L"]=s/tw
    r["T"]=(pmap(r["bias"])*.30+pmap(r["ret20"])*.30+pmap(r["rsi"])*.22+r["amt_pct"]*100*.18)
    r["U"]=0.5*r["L"]+0.5*r["T"]
    r["c"]=r["close"]; r["o"]=r["open"]; r["h"]=r["high"]; r["l"]=r["low"]
    rng=(r["h"]-r["l"]).replace(0,np.nan)
    r["pos"]=(r["c"]-r["l"])/rng
    r["amt_q"]=r["amt_pct"]
    r["cf_b"]=((r["amt_q"]-0.60)/0.40).clip(0,1) * ((12-r["L"])/12).clip(0,1)
    r["cf_t"]=((r["T"]-88)/12).clip(0,1) * ((0.5-r["pos"])/0.5).clip(0,1)

def build_score(b, K, g):
    r=RAW[b]
    return 50 + g*((r["U"] + K*r["cf_t"] - K*r["cf_b"]) - 50)

def events(b, s, low=0.0, high=100.0, gap=20):
    s=s.dropna()
    out=[]
    for thr, how in ((low,"min"), (high,"max")):
        m = (s<thr) if how=="min" else (s>thr)
        if not m.any(): out.append([]); continue
        grp=(m!=m.shift()).cumsum()
        picks=[seg.idxmin() if how=="min" else seg.idxmax() for _,seg in s[m].groupby(grp[m])]
        picks=sorted(set(picks)); res=[]
        for d in picks:
            if res and (d-res[-1]).days<=gap: continue
            res.append(d)
        out.append(res)
    return out[0], out[1]

def evaluate(b, ds, kind, h=20, tol=0.03):
    r=RAW[b]; close=r["c"].values; idx=r.index
    pos={d:i for i,d in enumerate(idx)}; rows=[]
    for d in ds:
        i=pos.get(d)
        if i is None or i+h>=len(close): continue
        c0=close[i]; seg=close[i+1:i+h+1]
        rows.append((seg[-1]/c0-1, seg.min()/c0-1, seg.max()/c0-1))
    if not rows: return (0,0,np.nan,np.nan)
    a=np.array(rows)
    ok=((a[:,0]>0)&(a[:,1]>=-tol)) if kind=="bottom" else ((a[:,0]<0)&(a[:,2]<=tol))
    return (len(a), int(ok.sum()), a[:,0].mean(), a[:,1].mean() if kind=="bottom" else a[:,2].mean())

print("="*116)
print("参数标定：K = 确认项强度，g = 增益（决定溢出区宽度）")
print("="*116)
print(f"{'K':>4} {'g':>5} | 底部事件  命中       均收益   均回撤 | 顶部事件  命中       均收益   均踏空")
best=[]
for K in (0, 10, 15, 20, 25, 30):
    for g in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
        tb=ob=tbret=tbmdd=0; tt=ot=ttret=ttrun=0; rb=[]; rm=[]; rt=[]; ru=[]
        for b in C.BOARD_ORDER:
            s=build_score(b,K,g)
            bl, tl = events(b, s)
            n,ok,mr,rk = evaluate(b, bl, "bottom"); tb+=n; ob+=ok
            if n: rb.append(mr); rm.append(rk)
            n2,ok2,mr2,rk2 = evaluate(b, tl, "top"); tt+=n2; ot+=ok2
            if n2: rt.append(mr2); ru.append(rk2)
        pb = ob/tb*100 if tb else 0; pt = ot/tt*100 if tt else 0
        print(f"{K:>4} {g:>5} |  {tb:>3}      {ob}/{tb}({pb:3.0f}%)  "
              f"{np.mean(rb)*100 if rb else 0:+6.2f}%  {np.mean(rm)*100 if rm else 0:+6.2f}% | "
              f"{tt:>3}      {ot}/{tt}({pt:3.0f}%)  {np.mean(rt)*100 if rt else 0:+6.2f}%  {np.mean(ru)*100 if ru else 0:+6.2f}%")
        if 4 <= tb <= 30 and 4 <= tt <= 30:
            best.append((pb+pt, K, g, tb, ob, tt, ot))
best.sort(reverse=True)
print("\n候选（两侧事件数均 4~30，按两侧命中率之和排序）:")
for sc,K,g,tb,ob,tt,ot in best[:8]:
    print(f"   K={K} g={g}  底 {ob}/{tb}({ob/tb*100:.0f}%)  顶 {ot}/{tt}({ot/tt*100:.0f}%)  合计命中 {sc:.0f}")
