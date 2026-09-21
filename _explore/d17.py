"""诊断 17：跨 11.7 年（2015~2026）多轮牛熊，验证「过热→下跌」到底成不成立。
只用价格/成交量因子（长历史没有成分股广度）。
目的：判断 v4 顶部只有 75% 是「牛市样本偏差」还是「逻辑本身不成立」。
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C

LD = r"D:\情绪指标4\data\cache\long_index"
NAMES = {"SH": "大盘", "HS300": "沪深300", "CSI1000": "中证1000",
         "CHINEXT": "创业板", "STAR": "科创板"}

def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0); dn = (-d).clip(lower=0)
    ru = up.ewm(alpha=1/n, adjust=False).mean(); rd = dn.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + ru/rd.replace(0, np.nan))

def pmap(v):
    x = v.astype(float)
    lo = np.nanpercentile(x.values, 2); hi = np.nanpercentile(x.values, 98)
    return (100.0*(x-lo)/(hi-lo)).clip(-25, 125)

def build(k):
    d = pd.read_parquet(os.path.join(LD, f"{k}.parquet")).sort_values("date")
    d = d.set_index("date")
    c = d["close"].astype(float)
    out = pd.DataFrame({"close": c})
    out["bias"] = c/c.rolling(60).mean() - 1
    out["ret20"] = c.pct_change(20)
    out["rsi"] = rsi(c)
    amt = d["amount"].astype(float) if "amount" in d else d["volume"]*c
    out["amt_q"] = amt.rolling(250, min_periods=60).rank(pct=True)
    out["vol"] = c.pct_change().rolling(20).std()
    out["T"] = pmap(out["bias"])*.30 + pmap(out["ret20"])*.30 + pmap(out["rsi"])*.22 + out["amt_q"]*100*.18
    # 反向：恐慌（用于底部验证）
    out["F"] = 100 - out["T"]
    return out.dropna(subset=["T"])

def events(p, thr, col="T", how="max", gap=30):
    s = p[col]
    m = s > thr if how == "max" else s < thr
    if not m.any(): return []
    grp = (m != m.shift()).cumsum()
    picks = [seg.idxmax() if how == "max" else seg.idxmin() for _, seg in s[m].groupby(grp[m])]
    picks = sorted(set(picks)); out = []
    for d in picks:
        if out and (d - out[-1]).days <= gap: continue
        out.append(d)
    return out

def evaluate(p, ds, h, side):
    close = p["close"].values; idx = p.index
    pos = {d: i for i, d in enumerate(idx)}; rows = []
    for d in ds:
        i = pos[d]
        if i + h >= len(close): continue
        c0 = close[i]; seg = close[i+1:i+h+1]
        rows.append((d, seg[-1]/c0-1, seg.min()/c0-1, seg.max()/c0-1))
    if not rows: return 0, 0, np.nan, np.nan, []
    a = np.array([[x[1], x[2], x[3]] for x in rows])
    if side == "top": ok = (a[:,0] < 0) & (a[:,2] <= 0.03)
    else: ok = (a[:,0] > 0) & (a[:,1] >= -0.03)
    return len(a), int(ok.sum()), a[:,0].mean(), (a[:,2].mean() if side=="top" else a[:,1].mean()), rows

def baseline(p, h, side):
    close = p["close"].values; ok = []
    for i in range(len(close)-h):
        c0 = close[i]; seg = close[i+1:i+h+1]
        if side == "top": ok.append(seg[-1]/c0-1 < 0 and seg.max()/c0-1 <= 0.03)
        else: ok.append(seg[-1]/c0-1 > 0 and seg.min()/c0-1 >= -0.03)
    return float(np.mean(ok))*100

P = {k: build(k) for k in NAMES}
print("="*118)
print("【顶部】11.7 年长样本：过热 T 高 → 未来是跌吗？（严格口径：收益<0 且 踏空≤+3%）")
print("="*118)
print(f"{'板块':>8}{'T阈值':>8}{'事件数':>8}{'T+20命中':>12}{'基线':>8}{'T+40命中':>12}{'基线':>8}{'T+60命中':>12}{'均收益':>10}")
for k, nm in NAMES.items():
    p = P[k]
    for thr in (90, 95):
        ds = events(p, thr, "T", "max")
        cells = []
        for h in (20, 40, 60):
            n, ok, mr, risk, _ = evaluate(p, ds, h, "top")
            cells.append((n, ok, baseline(p, h, "top"), mr))
        n20, ok20, b20, mr = cells[0]
        print(f"{nm:>8}{thr:>8}{len(ds):>8}"
              + "".join(f"{f'{c[1]}/{c[0]}({c[1]/c[0]*100 if c[0] else 0:.0f}%)':>12}{c[2]:>7.1f}%" for c in cells)
              + f"{mr*100:>9.2f}%")

print("\n" + "="*118)
print("【顶部事件明细】T>=90，T+20")
print("="*118)
for k, nm in NAMES.items():
    p = P[k]; ds = events(p, 90, "T", "max")
    n, ok, mr, risk, rows = evaluate(p, ds, 20, "top")
    print(f"\n{nm}  {ok}/{n} 命中")
    for d, ret, mdd, ru in rows:
        print(f"   {'OK ' if (ret<0 and ru<=0.03) else 'FAIL'} {str(d.date())}  20日{ret*100:+7.2f}%  最高{ru*100:+7.2f}%")

print("\n" + "="*118)
print("【底部对照】同一长样本：恐慌 T 低 → 未来是涨吗？")
print("="*118)
print(f"{'板块':>8}{'T阈值':>8}{'事件数':>8}{'T+20命中':>12}{'基线':>8}{'T+40命中':>12}{'基线':>8}{'均收益':>10}")
for k, nm in NAMES.items():
    p = P[k]
    for thr in (10, 5):
        ds = events(p, thr, "T", "min")
        cells = []
        for h in (20, 40):
            n, ok, mr, risk, _ = evaluate(p, ds, h, "bottom")
            cells.append((n, ok, baseline(p, h, "bottom")))
        n20, ok20, b20 = cells[0]
        _, _, mr, _, _ = evaluate(p, ds, 20, "bottom")
        print(f"{nm:>8}{thr:>8}{len(ds):>8}"
              + "".join(f"{f'{c[1]}/{c[0]}({c[1]/c[0]*100 if c[0] else 0:.0f}%)':>12}{c[2]:>7.1f}%" for c in cells)
              + f"{mr*100:>9.2f}%")
