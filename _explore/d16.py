"""诊断 16：
A) 底部补「地量磨底」分支（2024-09-23 科创板/创业板那种缩量磨出来的底）
B) 顶部试「情绪U自身顶背离」：U 从高位回落，但价格仍在高位
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
def pmap(v,lq=2,hq=98):
    lo=np.nanpercentile(v.values,lq); hi=np.nanpercentile(v.values,hq)
    return (100.0*(v-lo)/(hi-lo)).clip(-25,125)

def rk(s,n=250,mp=60): return s.rolling(n,min_periods=mp).rank(pct=True)*100

for b in C.BOARD_ORDER:
    r=RAW[b]
    s=pd.Series(0.0,index=r.index); tw=0.0
    for k in LEVEL_F: s=s+pmap(r[k].astype(float))*W[k]; tw+=W[k]
    r["L"]=s/tw
    r["T"]=(pmap(r["bias"])*.30+pmap(r["ret20"])*.30+pmap(r["rsi"])*.22+r["amt_pct"]*100*.18)
    r["U"]=0.5*r["L"]+0.5*r["T"]
    r["c"]=r["close"]; r["o"]=r["open"]; r["h"]=r["high"]; r["l"]=r["low"]
    rng=(r["h"]-r["l"]).replace(0,np.nan)
    r["yang"]=(r["c"]>r["o"]).astype(float)
    r["ls"]=(r[["o","c"]].min(axis=1)-r["l"])/rng
    r["ret1"]=r["c"].pct_change()
    r["amt_q"]=r["amt_pct"]
    r["vol_q"]=rk(r["vol"],250)
    r["p_pos"]=rk(r["c"],250)
    r["div"]=r["p_pos"]-rk(r["b20"],250)
    r["U_d20"]=r["U"]-r["U"].shift(20)
    r["U_max60"]=r["U"].rolling(60).max()

def ev(b,mask,how,gap=30,col="U"):
    r=RAW[b]; m=mask.fillna(False)
    if not m.any(): return []
    x=r[col]; grp=(m!=m.shift()).cumsum()
    picks=[seg.idxmin() if how=="min" else seg.idxmax() for _,seg in x[m].groupby(grp[m])]
    picks=sorted(set(picks)); out=[]
    for d in picks:
        if out and (d-out[-1]).days<=gap: continue
        out.append(d)
    return out

def stat(b,ds,kind,h=20):
    r=RAW[b]; close=r["c"].values; idx=r.index
    pos={d:i for i,d in enumerate(idx)}; rows=[]
    for d in ds:
        i=pos.get(d)
        if i is None or i+h>=len(close): continue
        c0=close[i]; seg=close[i+1:i+h+1]
        rows.append((d,seg[-1]/c0-1,seg.min()/c0-1,seg.max()/c0-1))
    return rows

def run(title, cands, kind, col="U", show=False):
    print("\n"+"="*120); print(title); print("="*120)
    print(f"{'条件':<40}"+"".join(f"{C.BOARDS[b]['name']:>8}" for b in C.BOARD_ORDER)+"   合计     均收益   均风险")
    for name,fn in cands.items():
        line=f"{name:<40}"; tn=tok=0; rets=[]; risks=[]; allr=[]
        for b in C.BOARD_ORDER:
            ds=ev(b, fn(RAW[b]), "min" if kind=="bottom" else "max", col=col)
            rows=stat(b,ds,kind)
            if not rows: line+=f"{'-':>8}"; continue
            a=np.array([[x[1],x[2],x[3]] for x in rows])
            ok=((a[:,0]>0)&(a[:,1]>=-0.03)) if kind=="bottom" else ((a[:,0]<0)&(a[:,2]<=0.03))
            tn+=len(a); tok+=int(ok.sum()); rets.append(a[:,0].mean())
            risks.append(a[:,1].mean() if kind=="bottom" else a[:,2].mean())
            allr+=[(C.BOARDS[b]['name'],)+x for x in rows]
            line+=f"{f'{int(ok.sum())}/{len(a)}':>8}"
        rr=f"{np.mean(rets)*100:+6.2f}%" if rets else "   -   "
        rk_=f"{np.mean(risks)*100:+6.2f}%" if risks else "   -   "
        print(line+f"   {tok}/{tn}"+(f"({tok/tn*100:3.0f}%)" if tn else "     ")+f"   {rr}   {rk_}")
        if show:
            for nm,d,ret,mdd,ru in allr:
                good = (ret>0 and mdd>=-0.03) if kind=="bottom" else (ret<0 and ru<=0.03)
                print(f"       {'OK ' if good else 'FAIL'} {nm:>8} {str(d.date())}"
                      f"  20日{ret*100:+7.2f}%  最深{mdd*100:+7.2f}%  最高{ru*100:+7.2f}%")

run("【A】底部：补「地量磨底」分支 —— 目标覆盖 2024-09-23 科创板/创业板", {
  "L<=15 & amt<=.20":              lambda r:(r["L"]<=15)&(r["amt_q"]<=.20),
  "L<=15 & amt<=.20 & vol_q<=.35": lambda r:(r["L"]<=15)&(r["amt_q"]<=.20)&(r["vol_q"]<=.35),
  "L<=10 & amt<=.20 & vol_q<=.35": lambda r:(r["L"]<=10)&(r["amt_q"]<=.20)&(r["vol_q"]<=.35),
  "L<=15 & amt<=.15 & vol_q<=.35": lambda r:(r["L"]<=15)&(r["amt_q"]<=.15)&(r["vol_q"]<=.35),
  "L<=20 & amt<=.20 & vol_q<=.30": lambda r:(r["L"]<=20)&(r["amt_q"]<=.20)&(r["vol_q"]<=.30),
  "L<=15 & amt<=.20 & vol_q<=.35 & 收阳": lambda r:(r["L"]<=15)&(r["amt_q"]<=.20)&(r["vol_q"]<=.35)&(r["yang"]>0),
  "L<=15 & amt<=.20 & vol_q<=.35 & 长下影": lambda r:(r["L"]<=15)&(r["amt_q"]<=.20)&(r["vol_q"]<=.35)&(r["ls"]>=.3),
}, "bottom", show=True)

run("【B】顶部：情绪 U 自身顶背离（U 从高位回落但价格仍在高位）", {
  "U>=70 & U_d20<0 & p_pos>=90":      lambda r:(r["U"]>=70)&(r["U_d20"]<0)&(r["p_pos"]>=90),
  "U>=75 & U_d20<-5 & p_pos>=90":     lambda r:(r["U"]>=75)&(r["U_d20"]<-5)&(r["p_pos"]>=90),
  "U>=70 & U<U_max60*0.9 & p_pos>=90":lambda r:(r["U"]>=70)&(r["U"]<r["U_max60"]*0.9)&(r["p_pos"]>=90),
  "U>=70 & div>=40 & p_pos>=90":      lambda r:(r["U"]>=70)&(r["div"]>=40)&(r["p_pos"]>=90),
  "U>=65 & div>=50 & p_pos>=95":      lambda r:(r["U"]>=65)&(r["div"]>=50)&(r["p_pos"]>=95),
  "U>=70 & div>=50":                  lambda r:(r["U"]>=70)&(r["div"]>=50),
  "p_pos>=95 & div>=60":              lambda r:(r["p_pos"]>=95)&(r["div"]>=60),
}, "top", show=True)
