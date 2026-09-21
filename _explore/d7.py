"""诊断 7：顶部侧做细梯度搜索 —— 破位确认 + 顶背离 + 量价背离；底部侧同步找最优。
目标：找到「事件少 + 正确率高」的组合。
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
    for col in ("open", "high", "low"):
        r[col] = d[col].reindex(r.index)
    RAW[b] = r

LEVEL_F = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20"]
W = {"b20": .18, "b60": .15, "r5": .11, "nh": .13, "lim": .09, "rsi": .14, "bias": .12, "ret20": .08}

for b in C.BOARD_ORDER:
    r = RAW[b]
    s = pd.Series(0.0, index=r.index); tw = 0.0
    for k in LEVEL_F:
        v = r[k].astype(float)
        lo = np.nanpercentile(v.values, 2); hi = np.nanpercentile(v.values, 98)
        s = s + (100.0 * (v - lo) / (hi - lo)).clip(-25, 125) * W[k]; tw += W[k]
    r["L"] = s / tw
    r["c"] = r["close"]; r["o"] = r["open"]; r["h"] = r["high"]; r["l"] = r["low"]
    rng = (r["h"] - r["l"]).replace(0, np.nan)
    r["rng"] = rng
    r["yang"] = (r["c"] > r["o"]).astype(float)
    r["us"] = (r["h"] - r[["o", "c"]].max(axis=1)) / rng
    r["ls"] = (r[["o", "c"]].min(axis=1) - r["l"]) / rng
    r["ret1"] = r["c"].pct_change()
    r["dn_ma20"] = (r["c"] < r["ma20"]).astype(float)
    r["m5b20"] = (r["ma5"] < r["ma20"]).astype(float)
    r["dn_ma20_1st"] = (r["dn_ma20"] > 0) & (r["dn_ma20"].shift(1) == 0)
    r["hi60"] = r["c"] >= r["c"].rolling(60).max()
    r["L_d20"] = r["L"] - r["L"].shift(20)          # 情绪 20 日变化（背离用）
    r["amt_q"] = r["amt_pct"]

def events(b, mask, how="min", gap=10):
    r = RAW[b]; m = mask.fillna(False)
    if not m.any(): return []
    L = r["L"]; grp = (m != m.shift()).cumsum()
    picks = [seg.idxmin() if how == "min" else seg.idxmax() for _, seg in L[m].groupby(grp[m])]
    picks = sorted(set(picks)); out = []
    for d in picks:
        if out and (d - out[-1]).days <= gap: continue
        out.append(d)
    return out

def evaluate(b, ds, kind, h=20, tol=0.03):
    r = RAW[b]; close = r["c"].values; idx = r.index
    pos = {d: i for i, d in enumerate(idx)}
    rows = []
    for d in ds:
        i = pos.get(d)
        if i is None or i + h >= len(close): continue
        c0 = close[i]; seg = close[i+1:i+h+1]
        rows.append((seg[-1]/c0-1, seg.min()/c0-1, seg.max()/c0-1))
    if not rows: return (0, 0, np.nan, np.nan, [])
    a = np.array(rows)
    ok = ((a[:,0] > 0) & (a[:,1] >= -tol)) if kind == "bottom" else ((a[:,0] < 0) & (a[:,2] <= tol))
    return (len(a), int(ok.sum()), a[:,0].mean(), a[:,1].mean() if kind=="bottom" else a[:,2].mean(),
            [str(d.date()) for d in ds])

def run(title, cands, kind):
    print("\n" + "="*118)
    print(title)
    print("="*118)
    print(f"{'条件':<34}" + "".join(f"{C.BOARDS[b]['name']:>8}" for b in C.BOARD_ORDER) + "    合计      均收益   均风险")
    best = None
    for name, fn in cands.items():
        line = f"{name:<34}"; tn = tok = 0; rets = []; risks = []
        for b in C.BOARD_ORDER:
            ds = events(b, fn(RAW[b]), "min" if kind == "bottom" else "max")
            n, ok, mr, risk, _ = evaluate(b, ds, kind)
            tn += n; tok += ok
            if n: rets.append(mr); risks.append(risk)
            line += f"{f'{ok}/{n}':>8}" if n else f"{'-':>8}"
        rr = f"{np.mean(rets)*100:+6.2f}%" if rets else "   -   "
        rk = f"{np.mean(risks)*100:+6.2f}%" if risks else "   -   "
        print(line + f"   {tok}/{tn}" + (f"({tok/tn*100:3.0f}%)" if tn else "     ") + f"   {rr}   {rk}")
    return

# ---------------- 顶部侧 ----------------
run("【顶部】破位确认梯度 —— 严格口径：20日收益<0 且 踏空≤+3%", {
    "L>=80 + 首次跌破MA20":      lambda r: (r["L"]>=80) & (r["dn_ma20_1st"]>0),
    "L>=70 + 首次跌破MA20":      lambda r: (r["L"]>=70) & (r["dn_ma20_1st"]>0),
    "L>=60 + 首次跌破MA20":      lambda r: (r["L"]>=60) & (r["dn_ma20_1st"]>0),
    "L>=70 + 跌破MA20":          lambda r: (r["L"]>=70) & (r["dn_ma20"]>0),
    "L>=60 + MA5<MA20":          lambda r: (r["L"]>=60) & (r["m5b20"]>0),
    "L>=70 + MA5<MA20":          lambda r: (r["L"]>=70) & (r["m5b20"]>0),
    "L>=70 + MA5<MA20 + 收阴":   lambda r: (r["L"]>=70) & (r["m5b20"]>0) & (r["yang"]<=0),
    "顶背离:创新高 & b20<0.45":   lambda r: (r["hi60"]>0) & (r["b20"]<0.45),
    "顶背离:创新高 & nh<0":       lambda r: (r["hi60"]>0) & (r["nh"]<0),
    "顶背离:创新高 & L_d20<0":    lambda r: (r["hi60"]>0) & (r["L_d20"]<0),
    "L>=75 & L_d20<0":           lambda r: (r["L"]>=75) & (r["L_d20"]<0),
    "L>=75 & L_d20<-5":          lambda r: (r["L"]>=75) & (r["L_d20"]<-5),
    "创新高 & 收阴 & 长上影>=.35":lambda r: (r["hi60"]>0) & (r["yang"]<=0) & (r["us"]>=0.35),
    "创新高 & 天量(amt>=.95)":    lambda r: (r["hi60"]>0) & (r["amt_q"]>=0.95),
    "创新高 & 天量 & 收阴":       lambda r: (r["hi60"]>0) & (r["amt_q"]>=0.95) & (r["yang"]<=0),
}, "top")

# ---------------- 底部侧 ----------------
run("【底部】梯度 —— 严格口径：20日收益>0 且 最深回撤≥-3%", {
    "L<=10":                     lambda r: r["L"]<=10,
    "L<=5":                      lambda r: r["L"]<=5,
    "L<=0":                      lambda r: r["L"]<=0,
    "L<=10 + b20近20日最低":      lambda r: (r["L"]<=10) & (r["b20"] <= r["b20"].rolling(20).min()),
    "L<=10 + 广度回升(b20升)":    lambda r: (r["L"]<=10) & (r["b20"] > r["b20"].shift(1)),
    "L<=10 + 前日跌>2% + 收阳":   lambda r: (r["L"]<=10) & (r["ret1"].shift(1)<=-0.02) & (r["yang"]>0),
    "L<=10 + 长下影>=.35":        lambda r: (r["L"]<=10) & (r["ls"]>=0.35),
    "L<=10 + 恐慌放量(amt>=.9)":  lambda r: (r["L"]<=10) & (r["amt_q"]>=0.9),
    "L<=10 + 地量(amt<=.15)":     lambda r: (r["L"]<=10) & (r["amt_q"]<=0.15),
    "L<=15 + 收阳":               lambda r: (r["L"]<=15) & (r["yang"]>0),
    "L<=10 + 收阳 + 广度回升":     lambda r: (r["L"]<=10) & (r["yang"]>0) & (r["b20"]>r["b20"].shift(1)),
}, "bottom")
