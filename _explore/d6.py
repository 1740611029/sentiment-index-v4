"""诊断 6：区分「真底」与「下跌途中的超跌」。
核心假设：底部 = 超跌水平 + 转折确认（止跌/长下影/放量收复）
        顶部 = 过热水平 + 转折确认（滞涨/破位/阴线）
测试不同确认条件对正确率的提升。
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
    # 补充日内形态与确认信号
    o, h, l, c = r["close"].shift(0), None, None, r["close"]
    RAW[b] = r

LEVEL_F = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20"]
W = {"b20": .18, "b60": .15, "r5": .11, "nh": .13, "lim": .09, "rsi": .14, "bias": .12, "ret20": .08}

def level_score(r, lo_q=2, hi_q=98):
    s = pd.Series(0.0, index=r.index)
    tw = 0.0
    for k in LEVEL_F:
        v = r[k].astype(float)
        lo = np.nanpercentile(v.values, lo_q); hi = np.nanpercentile(v.values, hi_q)
        m = (100.0 * (v - lo) / (hi - lo)).clip(-25, 125)
        s = s + m * W[k]; tw += W[k]
    return s / tw

def enrich(r):
    """加日内形态 / 确认信号（全部只用当日及历史）"""
    idx = r.index
    # 指数 OHLC：raw 里只有 close，这里从原始指数文件补齐
    return r

# 需要从指数文件补 OHLC
for b in C.BOARD_ORDER:
    d = data.load_index(b).set_index("date")
    r = RAW[b]
    for col in ("open", "high", "low"):
        r[col] = d[col].reindex(r.index)
    r["o"] = r["open"]; r["h"] = r["high"]; r["l"] = r["low"]; r["c"] = r["close"]
    rng = (r["h"] - r["l"]).replace(0, np.nan)
    r["pos_in_range"] = (r["c"] - r["l"]) / rng            # 收盘在当日振幅中的位置 0~1
    r["lower_shadow"] = (r[["o", "c"]].min(axis=1) - r["l"]) / rng   # 下影线占比
    r["upper_shadow"] = (r["h"] - r[["o", "c"]].max(axis=1)) / rng   # 上影线占比
    r["yang"] = (r["c"] > r["o"]).astype(float)            # 收阳
    r["ret1"] = r["c"].pct_change()
    r["dn_ma20"] = (r["c"] < r["ma20"]).astype(float)
    r["ma5_below_ma20"] = (r["ma5"] < r["ma20"]).astype(float)
    r["L"] = level_score(r)

def events(b, mask, how="min", gap=10):
    """连续段合并 + 冷却期 gap 天"""
    r = RAW[b]
    m = mask.fillna(False)
    if not m.any(): return []
    L = r["L"]
    grp = (m != m.shift()).cumsum()
    picks = [seg.idxmin() if how == "min" else seg.idxmax() for _, seg in L[m].groupby(grp[m])]
    picks = sorted(set(picks))
    out = []
    for d in picks:
        if out and (d - out[-1]).days <= gap: continue
        out.append(d)
    return out

def evaluate(b, ds, kind):
    r = RAW[b]; close = r["c"].values; idx = r.index
    pos = {d: i for i, d in enumerate(idx)}
    rows = []
    for d in ds:
        i = pos.get(d)
        if i is None or i + 20 >= len(close): continue
        c0 = close[i]; seg = close[i+1:i+21]
        rows.append((seg[-1]/c0-1, seg.min()/c0-1, seg.max()/c0-1))
    if not rows: return (0, 0, np.nan, np.nan)
    a = np.array(rows)
    ok = ((a[:, 0] > 0) & (a[:, 1] >= -0.03)) if kind == "bottom" else ((a[:, 0] < 0) & (a[:, 2] <= 0.03))
    return (len(a), int(ok.sum()), a[:, 0].mean(), a[:, 1].mean() if kind == "bottom" else a[:, 2].mean())

print("="*116)
print("【底部侧】超跌(L低) + 各种确认条件   —— 严格口径：20日收益>0 且 最深回撤≥-3%")
print("="*116)
cands_b = {
    "无确认(L<=5)":                 lambda r: r["L"] <= 5,
    "无确认(L<=10)":                lambda r: r["L"] <= 10,
    "L<=10 + 收阳":                 lambda r: (r["L"] <= 10) & (r["yang"] > 0),
    "L<=10 + 长下影(>=0.4)":        lambda r: (r["L"] <= 10) & (r["lower_shadow"] >= 0.4),
    "L<=10 + 收在振幅上半(pos>=.6)": lambda r: (r["L"] <= 10) & (r["pos_in_range"] >= 0.6),
    "L<=10 + 收阳 + 长下影":         lambda r: (r["L"] <= 10) & (r["yang"] > 0) & (r["lower_shadow"] >= 0.3),
    "L<=15 + 收阳 + 长下影":         lambda r: (r["L"] <= 15) & (r["yang"] > 0) & (r["lower_shadow"] >= 0.3),
    "L<=10 + 前一日大跌(<-2%)今收阳": lambda r: (r["L"] <= 10) & (r["ret1"].shift(1) <= -0.02) & (r["yang"] > 0),
    "L<=10 + 当日涨幅>0":            lambda r: (r["L"] <= 10) & (r["ret1"] > 0),
}
print(f"{'条件':<28}" + "".join(f"{C.BOARDS[b]['name']:>9}" for b in C.BOARD_ORDER) + "   合计")
for name, fn in cands_b.items():
    line = f"{name:<28}"; tn = tok = 0
    for b in C.BOARD_ORDER:
        ds = events(b, fn(RAW[b]), "min")
        n, ok, mr, risk = evaluate(b, ds, "bottom")
        tn += n; tok += ok
        line += f"{f'{ok}/{n}':>9}" if n else f"{'-':>9}"
    print(line + f"   {tok}/{tn}" + (f" ({tok/tn*100:.0f}%)" if tn else ""))

print("\n" + "="*116)
print("【顶部侧】过热(L高) + 各种确认条件   —— 严格口径：20日收益<0 且 最大踏空≤+3%")
print("="*116)
cands_t = {
    "无确认(L>=95)":                lambda r: r["L"] >= 95,
    "无确认(L>=90)":                lambda r: r["L"] >= 90,
    "L>=90 + 收阴":                 lambda r: (r["L"] >= 90) & (r["yang"] <= 0),
    "L>=90 + 长上影(>=0.4)":        lambda r: (r["L"] >= 90) & (r["upper_shadow"] >= 0.4),
    "L>=90 + 破MA20":               lambda r: (r["L"] >= 90) & (r["dn_ma20"] > 0),
    "L>=90 + 收阴 + 长上影":         lambda r: (r["L"] >= 90) & (r["yang"] <= 0) & (r["upper_shadow"] >= 0.3),
    "L>=85 + 破MA20":               lambda r: (r["L"] >= 85) & (r["dn_ma20"] > 0),
    "L>=85 + MA5<MA20":             lambda r: (r["L"] >= 85) & (r["ma5_below_ma20"] > 0),
    "L>=80 + 破MA20 + 收阴":         lambda r: (r["L"] >= 80) & (r["dn_ma20"] > 0) & (r["yang"] <= 0),
    "L>=80 + MA5<MA20 + 收阴":       lambda r: (r["L"] >= 80) & (r["ma5_below_ma20"] > 0) & (r["yang"] <= 0),
}
print(f"{'条件':<28}" + "".join(f"{C.BOARDS[b]['name']:>9}" for b in C.BOARD_ORDER) + "   合计")
for name, fn in cands_t.items():
    line = f"{name:<28}"; tn = tok = 0
    for b in C.BOARD_ORDER:
        ds = events(b, fn(RAW[b]), "max")
        n, ok, mr, risk = evaluate(b, ds, "top")
        tn += n; tok += ok
        line += f"{f'{ok}/{n}':>9}" if n else f"{'-':>9}"
    print(line + f"   {tok}/{tn}" + (f" ({tok/tn*100:.0f}%)" if tn else ""))
