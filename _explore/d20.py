"""诊断 20：网格搜索磨底型分支参数，目标 = 保留 2024-09 全部真底 + 挤掉擦边失败。"""
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
    codes = set(data.load_universe(b)); sub = big[big["code"].isin(codes)]
    g = sub.groupby("date", sort=True)
    add = pd.DataFrame({"deep5": g["ret5"].apply(lambda s: float((s < -0.10).mean())),
                        "disp": g["ret5"].std()})
    add.index = pd.to_datetime(add.index)
    r = r.join(add, how="left").join(mg, how="left")
    for c in ("mg_net5", "mg_bal20"): r[c] = r[c].ffill()
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
    r["f_disp"] = 100 - pmap(r["disp"])
    r["f_dd"] = 100 - pmap(r["dd250"])

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

def run(dd, disp, amt, K):
    tn=tok=0; rets=[]; risks=[]; allr=[]; sep_caught=set()
    for b in C.BOARD_ORDER:
        r = RAW[b]
        gr = ((r["f_dd"]>=dd)&(r["f_disp"]>=disp)&(r["amt_pct"]<=amt)).astype(float)
        r["score"] = r["U"] - 25*r["cf_b"] - K*gr
        ds = ev(b, r["score"] < 0)
        rows = stat(b, ds)
        if not rows: continue
        a = np.array([[x[1],x[2],x[3]] for x in rows])
        ok = (a[:,0]>0)&(a[:,1]>=-0.03)
        tn += len(a); tok += int(ok.sum()); rets.append(a[:,0].mean()); risks.append(a[:,1].mean())
        for (d, ret, mdd, ru) in rows:
            allr.append((C.BOARDS[b]['name'], d, ret, mdd))
            if str(d.date())[:7] in ("2024-08","2024-09","2024-10") and ret > 0 and mdd >= -0.03:
                sep_caught.add(C.BOARDS[b]['name'])
    return tn, tok, (np.mean(rets) if rets else 0), (np.mean(risks) if risks else 0), sep_caught, allr

print("="*112)
print("网格搜索：dd阈值 / disp阈值 / 地量阈值 / 惩罚系数K")
print("="*112)
print(f"{'dd':>4}{'disp':>6}{'amt':>7}{'K':>5} | {'命中':>12}{'均收益':>10}{'均回撤':>10}  2024秋底覆盖")
best=[]
for dd in (88, 90, 95):
    for disp in (75, 80, 85):
        for amt in (0.15, 0.20, 0.25):
            for K in (15, 20, 25):
                tn, tok, mr, mrk, sep, _ = run(dd, disp, amt, K)
                if tn == 0: continue
                rate = tok/tn*100
                line = (f"{dd:>4}{disp:>6}{amt:>7}{K:>5} | {f'{tok}/{tn}({rate:.0f}%)':>12}"
                        f"{mr*100:>9.2f}%{mrk*100:>9.2f}%  {len(sep)}/6 {''.join(sorted(sep))}")
                print(line)
                best.append((rate, len(sep), tn, tok, mr, dd, disp, amt, K))
print("\n按 (覆盖率, 命中率) 排序的前 8：")
best.sort(key=lambda x: (x[1], x[0]), reverse=True)
for rate, ns, tn, tok, mr, dd, disp, amt, K in best[:8]:
    print(f"  dd={dd} disp={disp} amt={amt} K={K}  命中 {tok}/{tn}({rate:.0f}%)  秋底{ns}/6  均收益{mr*100:+.2f}%")
