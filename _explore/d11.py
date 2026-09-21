"""诊断 11：顶部用更长窗口能否区分「真顶」与「牛市回调」。
假设：真顶之后跌幅持续；牛市回调 20 日内小跌但之后创新高。
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
    r["c"]=r["close"]; r["o"]=r["open"]
    r["yang"]=(r["c"]>r["o"]).astype(float); r["amt_q"]=r["amt_pct"]

def events(b,mask,how,gap,col):
    r=RAW[b]; m=mask.fillna(False)
    if not m.any(): return []
    x=r[col]; grp=(m!=m.shift()).cumsum()
    picks=[seg.idxmin() if how=="min" else seg.idxmax() for _,seg in x[m].groupby(grp[m])]
    picks=sorted(set(picks)); out=[]
    for d in picks:
        if out and (d-out[-1]).days<=gap: continue
        out.append(d)
    return out

def ev_at(b, ds, h):
    r=RAW[b]; close=r["c"].values; idx=r.index
    pos={d:i for i,d in enumerate(idx)}; rows=[]
    for d in ds:
        i=pos.get(d)
        if i is None or i+h>=len(close): continue
        c0=close[i]; seg=close[i+1:i+h+1]
        rows.append((d, seg[-1]/c0-1, seg.min()/c0-1, seg.max()/c0-1))
    return rows

CANDS={
 "T>=90":              lambda r:r["T"]>=90,
 "T>=95":              lambda r:r["T"]>=95,
 "T>=90 & 收阴":        lambda r:(r["T"]>=90)&(r["yang"]<=0),
 "T>=90 & 天量>=.95":   lambda r:(r["T"]>=90)&(r["amt_q"]>=.95),
 "L>=90":              lambda r:r["L"]>=90,
}
print("="*120)
print("【顶部】不同持有期的表现（冷却 45 天，严格口径：收益<0 且 踏空≤+5%）")
print("="*120)
for name, fn in CANDS.items():
    print(f"\n  {name}")
    print(f"    {'窗口':<8}" + "".join(f"{C.BOARDS[b]['name']:>8}" for b in C.BOARD_ORDER) + "    合计     均收益   均踏空")
    for h in (20, 40, 60, 90):
        line=f"    T+{h:<6}"; tn=tok=0; rets=[]; runs=[]
        for b in C.BOARD_ORDER:
            ds=events(b, fn(RAW[b]), "max", 45, "T")
            rows=ev_at(b, ds, h)
            if not rows:
                line+=f"{'-':>8}"; continue
            a=np.array([[x[1],x[2],x[3]] for x in rows])
            ok=(a[:,0]<0)&(a[:,2]<=0.05)
            tn+=len(a); tok+=int(ok.sum()); rets.append(a[:,0].mean()); runs.append(a[:,2].mean())
            line+=f"{f'{int(ok.sum())}/{len(a)}':>8}"
        rr=f"{np.mean(rets)*100:+6.2f}%" if rets else "   -   "
        rk=f"{np.mean(runs)*100:+6.2f}%" if runs else "   -   "
        print(line+f"   {tok}/{tn}"+(f"({tok/tn*100:3.0f}%)" if tn else "     ")+f"   {rr}   {rk}")

print("\n" + "="*120)
print("【对照】随机基线：随便挑一天，各窗口满足「收益<0 且 踏空≤+5%」的比例")
print("="*120)
for b in C.BOARD_ORDER:
    r=RAW[b]; close=r["c"].values
    out=[]
    for h in (20,40,60,90):
        ok=[]
        for i in range(len(close)-h):
            c0=close[i]; seg=close[i+1:i+h+1]
            ok.append((seg[-1]/c0-1<0) and (seg.max()/c0-1<=0.05))
        out.append(f"T+{h}:{np.mean(ok)*100:5.1f}%")
    print(f"  {C.BOARDS[b]['name']:>8}  " + "  ".join(out))
