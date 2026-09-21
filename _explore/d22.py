"""诊断 22：把磨底分支从二值开关改成平滑斜坡（避免折线跳变），重新标定。"""
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
    add = pd.DataFrame({"disp": g["ret5"].std()}); add.index = pd.to_datetime(add.index)
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

def run(A, B, W_, K):
    """斜坡：dd 从 A 到 A+W_ 线性 0→1；disp 从 B 到 B+W_ 线性 0→1"""
    tn=tok=0; rets=[]; risks=[]; allr=[]; sep=set()
    for b in C.BOARD_ORDER:
        r = RAW[b]
        g1 = ((r["f_dd"]-A)/W_).clip(0,1)
        g2 = ((r["f_disp"]-B)/W_).clip(0,1)
        r["score"] = r["U"] - 25*r["cf_b"] - K*g1*g2
        rows = stat(b, ev(b, r["score"] < 0))
        if not rows: continue
        a = np.array([[x[1],x[2],x[3]] for x in rows])
        ok = (a[:,0]>0)&(a[:,1]>=-0.03)
        tn += len(a); tok += int(ok.sum()); rets.append(a[:,0].mean()); risks.append(a[:,1].mean())
        for (d, ret, mdd, ru) in rows:
            allr.append((C.BOARDS[b]['name'], d, ret, mdd))
            if str(d.date())[:7] in ("2024-08","2024-09","2024-10") and ret>0 and mdd>=-0.03:
                sep.add(C.BOARDS[b]['name'])
    return tn, tok, (np.mean(rets) if rets else 0), (np.mean(risks) if risks else 0), sep, allr

print("="*112)
print("平滑斜坡网格（A=回撤起点, B=同步起点, W=斜坡宽度, K=惩罚系数）")
print("="*112)
print(f"{'A':>4}{'B':>5}{'W':>4}{'K':>5} | {'命中':>13}{'均收益':>10}{'均回撤':>10}  秋底")
res=[]
for A in (85, 88, 90):
    for B in (70, 75, 78):
        for W_ in (8, 12):
            for K in (25, 30, 35, 40):
                tn, tok, mr, mrk, sep, _ = run(A, B, W_, K)
                if tn == 0: continue
                rate = tok/tn*100
                print(f"{A:>4}{B:>5}{W_:>4}{K:>5} | {f'{tok}/{tn}({rate:.0f}%)':>13}"
                      f"{mr*100:>9.2f}%{mrk*100:>9.2f}%  {len(sep)}/6")
                res.append((len(sep), rate, tn, tok, mr, A, B, W_, K))
print("\n最优（先覆盖秋底，再命中率）:")
res.sort(key=lambda x: (x[0], x[1]), reverse=True)
for ns, rate, tn, tok, mr, A, B, W_, K in res[:6]:
    print(f"  A={A} B={B} W={W_} K={K}  命中 {tok}/{tn}({rate:.0f}%)  秋底{ns}/6  均收益{mr*100:+.2f}%")
