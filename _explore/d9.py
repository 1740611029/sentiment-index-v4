"""诊断 9：冷却期长度对正确率的影响（同一轮行情的连续触发应合并为一次）。"""
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

LEVEL_F = ["b20","b60","r5","nh","lim","rsi","bias","ret20"]
W = {"b20":.18,"b60":.15,"r5":.11,"nh":.13,"lim":.09,"rsi":.14,"bias":.12,"ret20":.08}
def pmap(v, lo_q=2, hi_q=98):
    lo=np.nanpercentile(v.values,lo_q); hi=np.nanpercentile(v.values,hi_q)
    return (100.0*(v-lo)/(hi-lo)).clip(-25,125)

for b in C.BOARD_ORDER:
    r = RAW[b]
    s=pd.Series(0.0,index=r.index); tw=0.0
    for k in LEVEL_F:
        s = s + pmap(r[k].astype(float))*W[k]; tw += W[k]
    r["L"] = s/tw
    r["T"] = (pmap(r["bias"])*.30 + pmap(r["ret20"])*.30 + pmap(r["rsi"])*.22 + r["amt_pct"]*100*.18)
    r["c"]=r["close"]; r["o"]=r["open"]; r["h"]=r["high"]; r["l"]=r["low"]
    rng=(r["h"]-r["l"]).replace(0,np.nan)
    r["yang"]=(r["c"]>r["o"]).astype(float)
    r["us"]=(r["h"]-r[["o","c"]].max(axis=1))/rng
    r["ret1"]=r["c"].pct_change()
    r["amt_q"]=r["amt_pct"]
    r["m5b20"]=(r["ma5"]<r["ma20"]).astype(float)
    r["vol_q"]=r["vol"].rolling(250,min_periods=60).rank(pct=True)

def events(b, mask, how, gap, col):
    r=RAW[b]; m=mask.fillna(False)
    if not m.any(): return []
    x=r[col]; grp=(m!=m.shift()).cumsum()
    picks=[seg.idxmin() if how=="min" else seg.idxmax() for _,seg in x[m].groupby(grp[m])]
    picks=sorted(set(picks)); out=[]
    for d in picks:
        if out and (d-out[-1]).days <= gap: continue
        out.append(d)
    return out

def evaluate(b, ds, kind, h=20, tol=0.03):
    r=RAW[b]; close=r["c"].values; idx=r.index
    pos={d:i for i,d in enumerate(idx)}; rows=[]
    for d in ds:
        i=pos.get(d)
        if i is None or i+h>=len(close): continue
        c0=close[i]; seg=close[i+1:i+h+1]
        rows.append((d, seg[-1]/c0-1, seg.min()/c0-1, seg.max()/c0-1))
    if not rows: return (0,0,np.nan,np.nan,[])
    a=np.array([[x[1],x[2],x[3]] for x in rows])
    ok=((a[:,0]>0)&(a[:,1]>=-tol)) if kind=="bottom" else ((a[:,0]<0)&(a[:,2]<=tol))
    return (len(a),int(ok.sum()),a[:,0].mean(),a[:,1].mean() if kind=="bottom" else a[:,2].mean(),rows)

def run(title, cands, kind, col, gaps=(10,20,30,45), show_gap=None):
    print("\n"+"="*112); print(title); print("="*112)
    for name, fn in cands.items():
        print(f"\n  {name}")
        print(f"    {'冷却期':<8}{'':<6}" + "".join(f"{C.BOARDS[b]['name']:>8}" for b in C.BOARD_ORDER) + "    合计      均收益   均风险")
        for gap in gaps:
            line=f"    {gap}天{'':<7}"; tn=tok=0; rets=[]; risks=[]; allrows=[]
            for b in C.BOARD_ORDER:
                ds=events(b, fn(RAW[b]), "min" if kind=="bottom" else "max", gap, col)
                n,ok,mr,risk,rows=evaluate(b,ds,kind)
                tn+=n; tok+=ok; allrows+=[(C.BOARDS[b]['name'],)+x for x in rows]
                if n: rets.append(mr); risks.append(risk)
                line += f"{f'{ok}/{n}':>8}" if n else f"{'-':>8}"
            rr=f"{np.mean(rets)*100:+6.2f}%" if rets else "   -   "
            rk=f"{np.mean(risks)*100:+6.2f}%" if risks else "   -   "
            print(line + f"   {tok}/{tn}" + (f"({tok/tn*100:3.0f}%)" if tn else "     ") + f"   {rr}   {rk}")
            if show_gap is not None and gap==show_gap and allrows:
                for nm,d,ret,mdd,ru in allrows:
                    flag="OK " if ((ret>0 and mdd>=-0.03) if kind=="bottom" else (ret<0 and ru<=0.03)) else "FAIL"
                    print(f"          {flag} {nm:>8} {str(d.date())}  20日{ret*100:+7.2f}%  最深{mdd*100:+7.2f}%  最高{ru*100:+7.2f}%")

run("【底部】超跌 + 量能确认（冷却期扫描）", {
  "L<=10 & amt>=.90":            lambda r: (r["L"]<=10)&(r["amt_q"]>=.90),
  "L<=10 & amt>=.85":            lambda r: (r["L"]<=10)&(r["amt_q"]>=.85),
  "L<=15 & amt>=.90":            lambda r: (r["L"]<=15)&(r["amt_q"]>=.90),
  "L<=10":                       lambda r: r["L"]<=10,
  "L<=5":                        lambda r: r["L"]<=5,
}, "bottom", "L", show_gap=30)

run("【顶部】过热 T + 确认（冷却期扫描）", {
  "T>=90":                       lambda r: r["T"]>=90,
  "T>=90 & 收阴":                 lambda r: (r["T"]>=90)&(r["yang"]<=0),
  "T>=90 & 天量>=.95":            lambda r: (r["T"]>=90)&(r["amt_q"]>=.95),
  "T>=90 & 天量 & 收阴":          lambda r: (r["T"]>=90)&(r["amt_q"]>=.95)&(r["yang"]<=0),
  "T>=95":                       lambda r: r["T"]>=95,
  "T>=85 & MA5<MA20":            lambda r: (r["T"]>=85)&(r["m5b20"]>0),
}, "top", "T", show_gap=30)
