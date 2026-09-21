"""诊断 21：最终确认 —— 去掉多余的地量约束，只保留「深度回撤 + 高同步性」。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors

big = data.build_stock_indicators()
RAW = {}
for b in C.BOARD_ORDER:
    r = factors.build_board_raw(b, big)
    r = r[r.index >= pd.Timestamp(C.BACKTEST_START)].copy()
    codes = set(data.load_universe(b)); sub = big[big["code"].isin(codes)]
    g = sub.groupby("date", sort=True)
    add = pd.DataFrame({"disp": g["ret5"].std()})
    add.index = pd.to_datetime(add.index)
    r = r.join(add, how="left")
    r["dd250"] = r["close"]/r["close"].rolling(250, min_periods=60).max() - 1
    RAW[b] = r

LEVEL_F = ["b20","b60","r5","nh","lim","rsi","bias","ret20"]
W = {"b20":.18,"b60":.15,"r5":.11,"nh":.13,"lim":.09,"rsi":.14,"bias":.12,"ret20":.08}
def pmap(v, lq=2, hq=98):
    x = v.astype(float); lo = np.nanpercentile(x.values, lq); hi = np.nanpercentile(x.values, hq)
    return (100.0*(x-lo)/(hi-lo)).clip(-25, 125)

for b in C.BOARD_ORDER:
    r = RAW[b]
    s = pd.Series(0.0, index=r.index); tw = 0.0
    for k in LEVEL_F: s = s + pmap(r[k])*W[k]; tw += W[k]
    r["L"] = s/tw
    r["T"] = pmap(r["bias"])*.30 + pmap(r["ret20"])*.30 + pmap(r["rsi"])*.22 + r["amt_pct"]*100*.18
    r["U"] = 0.5*r["L"] + 0.5*r["T"]
    r["cf_b"] = (((r["amt_pct"]-0.60)/0.40).clip(0,1)*((12-r["L"])/12).clip(0,1)).fillna(0)
    r["f_disp"] = (100 - pmap(r["disp"])).fillna(0)
    r["f_dd"] = (100 - pmap(r["dd250"])).fillna(0)

def ev(b, mask, gap=20, col="score"):
    r = RAW[b]; m = mask.fillna(False)
    if not m.any(): return []
    x = r[col]; grp = (m != m.shift()).cumsum()
    picks = [seg.idxmin() for _, seg in x[m].groupby(grp[m])]
    picks = sorted(set(picks)); out = []
    for d in picks:
        if out and (d-out[-1]).days <= gap: continue
        out.append(d)
    return out

def stat(b, ds, h=20):
    r = RAW[b]; close = r["close"].values; idx = r.index
    pos = {d:i for i,d in enumerate(idx)}; rows = []
    for d in ds:
        i = pos.get(d)
        if i is None or i+h >= len(close): continue
        c0 = close[i]; seg = close[i+1:i+h+1]
        rows.append((d, seg[-1]/c0-1, seg.min()/c0-1, seg.max()/c0-1))
    return rows

def score(b, dd, disp, K, use_amt=False, amt=0.20):
    r = RAW[b]
    gr = ((r["f_dd"]>=dd)&(r["f_disp"]>=disp))
    if use_amt: gr = gr & (r["amt_pct"]<=amt)
    r["score"] = r["U"] - 25*r["cf_b"] - K*gr.astype(float)
    return r["score"]

def summary(name, mk):
    tn=tok=0; rets=[]; risks=[]; allr=[]
    for b in C.BOARD_ORDER:
        r = RAW[b]; r["score"] = mk(b)
        rows = stat(b, ev(b, r["score"] < 0))
        if not rows: continue
        a = np.array([[x[1],x[2],x[3]] for x in rows])
        ok = (a[:,0]>0)&(a[:,1]>=-0.03)
        tn += len(a); tok += int(ok.sum()); rets.append(a[:,0].mean()); risks.append(a[:,1].mean())
        allr += [(C.BOARDS[b]['name'],)+x for x in rows]
    print(f"\n  {name:<40} 命中 {tok}/{tn} ({tok/tn*100:.0f}%)  "
          f"均收益 {np.mean(rets)*100:+.2f}%  均回撤 {np.mean(risks)*100:+.2f}%")
    return allr, tok, tn

print("="*116)
print("无地量约束 vs 有地量约束")
print("="*116)
a1,_,_ = summary("无amt  dd95 disp80 K25", lambda b: score(b,95,80,25))
a2,_,_ = summary("有amt<=.20 dd95 disp80 K25", lambda b: score(b,95,80,25,True))
a3,_,_ = summary("基准（无磨底分支）", lambda b: RAW[b]["U"] - 25*RAW[b]["cf_b"])

print("\n" + "="*116)
print("最终方案事件明细（无amt, dd95, disp80, K25）")
print("="*116)
for nm, d, ret, mdd, ru in a1:
    print(f"  {'OK ' if (ret>0 and mdd>=-0.03) else 'FAIL'} {nm:>8} {str(d.date())}  20日{ret*100:+7.2f}%  最深{mdd*100:+7.2f}%")
