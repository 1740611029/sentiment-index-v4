"""诊断 10：顶部最后一轮 —— 过热 + 动能破位 / 广度回落 / 涨停退潮。"""
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
    r["yang"]=(r["c"]>r["o"]).astype(float)
    r["ret1"]=r["c"].pct_change()
    r["amt_q"]=r["amt_pct"]
    r["m5b20"]=(r["ma5"]<r["ma20"]).astype(float)
    r["m5b20_1st"]=(r["m5b20"]>0)&(r["m5b20"].shift(1)==0)
    r["dn20_1st"]=(r["c"]<r["ma20"])&(r["c"].shift(1)>=r["ma20"].shift(1))
    r["b20_d10"]=r["b20"]-r["b20"].shift(10)
    r["lim_d5"]=r["lim"]-r["lim"].shift(5)
    r["T_d10"]=r["T"]-r["T"].shift(10)

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

def evaluate(b,ds,kind,h=20,tol=0.03):
    r=RAW[b]; close=r["c"].values; idx=r.index
    pos={d:i for i,d in enumerate(idx)}; rows=[]
    for d in ds:
        i=pos.get(d)
        if i is None or i+h>=len(close): continue
        c0=close[i]; seg=close[i+1:i+h+1]
        rows.append((d,seg[-1]/c0-1,seg.min()/c0-1,seg.max()/c0-1))
    if not rows: return (0,0,np.nan,np.nan,[])
    a=np.array([[x[1],x[2],x[3]] for x in rows])
    ok=((a[:,0]>0)&(a[:,1]>=-tol)) if kind=="bottom" else ((a[:,0]<0)&(a[:,2]<=tol))
    return (len(a),int(ok.sum()),a[:,0].mean(),a[:,2].mean(),rows)

CANDS = {
 "T>=80 & MA5<MA20":        lambda r:(r["T"]>=80)&(r["m5b20"]>0),
 "T>=75 & MA5<MA20":        lambda r:(r["T"]>=75)&(r["m5b20"]>0),
 "T>=70 & MA5<MA20":        lambda r:(r["T"]>=70)&(r["m5b20"]>0),
 "T>=70 & 首次MA5<MA20":     lambda r:(r["T"]>=70)&(r["m5b20_1st"]>0),
 "T>=70 & 首次跌破MA20":     lambda r:(r["T"]>=70)&(r["dn20_1st"]>0),
 "T>=85 & b20_d10<0":       lambda r:(r["T"]>=85)&(r["b20_d10"]<0),
 "T>=85 & b20_d10<-0.1":    lambda r:(r["T"]>=85)&(r["b20_d10"]<-0.10),
 "T>=85 & lim_d5<0":        lambda r:(r["T"]>=85)&(r["lim_d5"]<0),
 "T>=85 & T_d10<0":         lambda r:(r["T"]>=85)&(r["T_d10"]<0),
 "T>=85 & T_d10<0 & 收阴":   lambda r:(r["T"]>=85)&(r["T_d10"]<0)&(r["yang"]<=0),
 "T>=90 & T_d10<0":         lambda r:(r["T"]>=90)&(r["T_d10"]<0),
 "T>=90 & b20_d10<-0.1":    lambda r:(r["T"]>=90)&(r["b20_d10"]<-0.10),
 "L>=75 & MA5<MA20":        lambda r:(r["L"]>=75)&(r["m5b20"]>0),
 "L>=80 & 首次跌破MA20":     lambda r:(r["L"]>=80)&(r["dn20_1st"]>0),
}
print("="*116)
print("【顶部】最终搜索 —— 严格口径：20日收益<0 且 踏空≤+3%")
print("="*116)
for gap in (20, 30):
    print(f"\n--- 冷却期 {gap} 天 ---")
    print(f"{'条件':<28}" + "".join(f"{C.BOARDS[b]['name']:>8}" for b in C.BOARD_ORDER) + "    合计      均收益   均踏空")
    for name, fn in CANDS.items():
        line=f"{name:<28}"; tn=tok=0; rets=[]; risks=[]; rowsall=[]
        for b in C.BOARD_ORDER:
            ds=events(b, fn(RAW[b]), "max", gap, "T")
            n,ok,mr,risk,rows=evaluate(b,ds,"top")
            tn+=n; tok+=ok; rowsall+=[(C.BOARDS[b]['name'],)+x for x in rows]
            if n: rets.append(mr); risks.append(risk)
            line+=f"{f'{ok}/{n}':>8}" if n else f"{'-':>8}"
        rr=f"{np.mean(rets)*100:+6.2f}%" if rets else "   -   "
        rk=f"{np.mean(risks)*100:+6.2f}%" if risks else "   -   "
        print(line+f"   {tok}/{tn}"+(f"({tok/tn*100:3.0f}%)" if tn else "     ")+f"   {rr}   {rk}")
