"""诊断 13：不对称确认（只保留底部确认 K_b，顶部不加强 K_t=0），并输出事件明细。"""
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
    r["cf_b"]=((r["amt_q"]-0.60)/0.40).clip(0,1)*((12-r["L"])/12).clip(0,1)
    r["cf_t"]=((r["T"]-88)/12).clip(0,1)*((0.5-r["pos"])/0.5).clip(0,1)

def build_score(b, Kb, Kt, g):
    r=RAW[b]
    return 50 + g*((r["U"] + Kt*r["cf_t"] - Kb*r["cf_b"]) - 50)

def events(b, s, gap=20):
    s=s.dropna(); out=[]
    for thr, how in ((0.0,"min"), (100.0,"max")):
        m=(s<thr) if how=="min" else (s>thr)
        if not m.any(): out.append([]); continue
        grp=(m!=m.shift()).cumsum()
        picks=[seg.idxmin() if how=="min" else seg.idxmax() for _,seg in s[m].groupby(grp[m])]
        picks=sorted(set(picks)); res=[]
        for d in picks:
            if res and (d-res[-1]).days<=gap: continue
            res.append(d)
        out.append(res)
    return out[0], out[1]

def ev_detail(b, ds, kind, h=20):
    r=RAW[b]; close=r["c"].values; idx=r.index
    pos={d:i for i,d in enumerate(idx)}; rows=[]
    for d in ds:
        i=pos.get(d)
        if i is None or i+h>=len(close): continue
        c0=close[i]; seg=close[i+1:i+h+1]
        rows.append((d,seg[-1]/c0-1,seg.min()/c0-1,seg.max()/c0-1))
    return rows

def summarize(b, rows, kind, tol=0.03):
    if not rows: return (0,0,np.nan,np.nan)
    a=np.array([[x[1],x[2],x[3]] for x in rows])
    ok=((a[:,0]>0)&(a[:,1]>=-tol)) if kind=="bottom" else ((a[:,0]<0)&(a[:,2]<=tol))
    return (len(a),int(ok.sum()),a[:,0].mean(),a[:,1].mean() if kind=="bottom" else a[:,2].mean())

print("="*112)
print("不对称确认扫描（K_t=0）")
print("="*112)
print(f"{'Kb':>4} {'g':>5} | 底部 命中      均收益  均回撤 | 顶部 命中      均收益  均踏空")
for Kb in (0,15,20,25,30):
    for g in (0.9,1.0,1.1,1.2):
        tb=ob=tt=ot=0; rb=[];rm=[];rt=[];ru=[]
        for b in C.BOARD_ORDER:
            s=build_score(b,Kb,0,g); bl,tl=events(b,s)
            n,ok,mr,rk=summarize(b,ev_detail(b,bl,"bottom"),"bottom"); tb+=n;ob+=ok
            if n: rb.append(mr); rm.append(rk)
            n2,ok2,mr2,rk2=summarize(b,ev_detail(b,tl,"top"),"top"); tt+=n2;ot+=ok2
            if n2: rt.append(mr2); ru.append(rk2)
        print(f"{Kb:>4} {g:>5} | {tb:>3} {ob}/{tb}({ob/tb*100 if tb else 0:3.0f}%) "
              f"{np.mean(rb)*100 if rb else 0:+6.2f}% {np.mean(rm)*100 if rm else 0:+6.2f}% | "
              f"{tt:>3} {ot}/{tt}({ot/tt*100 if tt else 0:3.0f}%) "
              f"{np.mean(rt)*100 if rt else 0:+6.2f}% {np.mean(ru)*100 if ru else 0:+6.2f}%")

print("\n"+"="*112)
print("选定 Kb=25, Kt=0, g=1.0 —— 事件明细")
print("="*112)
for b in C.BOARD_ORDER:
    s=build_score(b,25,0,1.0); bl,tl=events(b,s)
    rb=ev_detail(b,bl,"bottom"); rt=ev_detail(b,tl,"top")
    print(f"\n■ {C.BOARDS[b]['name']}  (真底 {RAW[b]['c'].idxmin().date()} / 真顶 {RAW[b]['c'].idxmax().date()})")
    print(f"  分值区间 {s.min():.1f} ~ {s.max():.1f}   当前(末日) {s.iloc[-1]:.1f}")
    print("  【底部溢出 <0】")
    for d,ret,mdd,ru in rb:
        f="OK " if (ret>0 and mdd>=-0.03) else "FAIL"
        print(f"    {f} {str(d.date())}  分值{s.loc[d]:6.1f}  20日{ret*100:+7.2f}%  最深{mdd*100:+7.2f}%")
    print("  【顶部溢出 >100】")
    for d,ret,mdd,ru in rt:
        f="OK " if (ret<0 and ru<=0.03) else "FAIL"
        print(f"    {f} {str(d.date())}  分值{s.loc[d]:6.1f}  20日{ret*100:+7.2f}%  最高{ru*100:+7.2f}%")
