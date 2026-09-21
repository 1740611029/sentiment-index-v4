"""诊断 19：底部两种形态分开建模
  暴跌型（2024-02、2025-04）：深跌占比高 + 放量 + 融资去杠杆       —— 已有 cf_b
  磨底型（2024-09）：        深度回撤 + 高同步性 + 地量（不暴跌）   —— 新增
分别测两个分支的命中率，再合并。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors, extdata

big = data.build_stock_indicators()
mg = extdata.margin_factors(extdata.fetch_margin())
RAW = {}
for b in C.BOARD_ORDER:
    r = factors.build_board_raw(b, big)
    r = r[r.index >= pd.Timestamp(C.BACKTEST_START)].copy()
    codes = set(data.load_universe(b))
    sub = big[big["code"].isin(codes)]
    g = sub.groupby("date", sort=True)
    add = pd.DataFrame({
        "deep5": g["ret5"].apply(lambda s: float((s < -0.10).mean())),
        "disp":  g["ret5"].std(),
    })
    add.index = pd.to_datetime(add.index)
    r = r.join(add, how="left").join(mg, how="left")
    for c in ("mg_net5", "mg_bal20"):
        r[c] = r[c].ffill()
    r["dd250"] = r["close"]/r["close"].rolling(250, min_periods=60).max() - 1
    RAW[b] = r

LEVEL_F = ["b20","b60","r5","nh","lim","rsi","bias","ret20"]
W = {"b20":.18,"b60":.15,"r5":.11,"nh":.13,"lim":.09,"rsi":.14,"bias":.12,"ret20":.08}
def pmap(v, lq=2, hq=98):
    x = v.astype(float)
    lo = np.nanpercentile(x.values, lq); hi = np.nanpercentile(x.values, hq)
    return (100.0*(x-lo)/(hi-lo)).clip(-25, 125)

for b in C.BOARD_ORDER:
    r = RAW[b]
    s = pd.Series(0.0, index=r.index); tw = 0.0
    for k in LEVEL_F: s = s + pmap(r[k])*W[k]; tw += W[k]
    r["L"] = s/tw
    r["T"] = pmap(r["bias"])*.30 + pmap(r["ret20"])*.30 + pmap(r["rsi"])*.22 + r["amt_pct"]*100*.18
    r["U"] = 0.5*r["L"] + 0.5*r["T"]
    r["cf_b"] = (((r["amt_pct"]-0.60)/0.40).clip(0,1)*((12-r["L"])/12).clip(0,1)).fillna(0)
    r["f_deep5"] = pmap(r["deep5"])
    r["f_disp"]  = 100 - pmap(r["disp"])
    r["f_dd"]    = 100 - pmap(r["dd250"])
    r["f_mg5"]   = 100 - pmap(r["mg_net5"])

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

def report(title, fn_map, show=True):
    print("\n" + "="*118); print(title); print("="*118)
    for name, fn in fn_map.items():
        tn=tok=0; rets=[]; risks=[]; allr=[]
        for b in C.BOARD_ORDER:
            r = RAW[b]; r["score"] = fn(r)
            ds = ev(b, r["score"] < 0)
            rows = stat(b, ds)
            if not rows: continue
            a = np.array([[x[1],x[2],x[3]] for x in rows])
            ok = (a[:,0]>0)&(a[:,1]>=-0.03)
            tn += len(a); tok += int(ok.sum()); rets.append(a[:,0].mean()); risks.append(a[:,1].mean())
            allr += [(C.BOARDS[b]['name'],)+x for x in rows]
        if tn == 0:
            print(f"\n  {name:<44} 无事件"); continue
        print(f"\n  {name:<44} 命中 {tok}/{tn} ({tok/tn*100:.0f}%)  "
              f"均收益 {np.mean(rets)*100:+.2f}%  均回撤 {np.mean(risks)*100:+.2f}%")
        if show:
            for nm, d, ret, mdd, ru in allr:
                print(f"       {'OK ' if (ret>0 and mdd>=-0.03) else 'FAIL'} {nm:>8} {str(d.date())}"
                      f"  20日{ret*100:+7.2f}%  最深{mdd*100:+7.2f}%")

# ---- A. 现状基线 ----
report("【A】现状基线", {"U - 25·cf_b": lambda r: r["U"] - 25*r["cf_b"]})

# ---- B. 加新因子（系数正确缩放：k 分点位，最大约 k*1.25 分）----
report("【B】加新因子（正确系数）", {
 "U -25cf_b -10·(f_mg5/100)":      lambda r: r["U"] - 25*r["cf_b"] - 10*r["f_mg5"]/100,
 "U -25cf_b -10·(f_dd/100)":       lambda r: r["U"] - 25*r["cf_b"] - 10*r["f_dd"]/100,
 "U -25cf_b -10·(f_disp/100)":     lambda r: r["U"] - 25*r["cf_b"] - 10*r["f_disp"]/100,
 "U -25cf_b -10·(f_deep5/100)":    lambda r: r["U"] - 25*r["cf_b"] - 10*r["f_deep5"]/100,
 "U -25cf_b -6·f_dd -6·f_disp":    lambda r: r["U"] - 25*r["cf_b"] - 6*r["f_dd"]/100 - 6*r["f_disp"]/100,
})

# ---- C. 磨底型专属分支（不暴跌 + 深度回撤 + 高同步 + 地量）----
def grind(r, dd=90, disp=80, amt=0.20):
    g = ((r["f_dd"] >= dd) & (r["f_disp"] >= disp) & (r["amt_pct"] <= amt)).astype(float)
    return g
report("【C】磨底型分支单独看（连续 g 乘以惩罚系数）", {
 "U -25cf_b -20·grind(dd90,disp80,amt.20)": lambda r: r["U"] - 25*r["cf_b"] - 20*grind(r),
 "U -25cf_b -20·grind(dd85,disp75,amt.25)": lambda r: r["U"] - 25*r["cf_b"] - 20*grind(r,85,75,.25),
 "U -25cf_b -20·grind(dd95,disp85,amt.15)": lambda r: r["U"] - 25*r["cf_b"] - 20*grind(r,95,85,.15),
}, show=True)
