"""诊断 18：底部侧加新信息源
  ① 融资去杠杆（mg_net5 / mg_bal20）  —— 真实杠杆维度
  ② 深跌占比（近5日跌超10% / 近20日跌超20%）
  ③ 截面离散度（同步暴跌时离散度低）
  ④ 指数距250日高点回撤
目标：抓到 2024-09 地量底，并减少 2024-01-22（中证1000 被套14%）这类失败
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

    # ---- 新的广度因子（从个股长表聚合）----
    codes = set(data.load_universe(b))
    sub = big[big["code"].isin(codes)]
    g = sub.groupby("date", sort=True)
    add = pd.DataFrame({
        "deep5":  g["ret5"].apply(lambda s: float((s < -0.10).mean())),
        "deep20": g["ret20"].apply(lambda s: float((s < -0.20).mean())),
        "disp":   g["ret5"].std(),
        "nl60":   g["is_nl60"].mean(),
    })
    add.index = pd.to_datetime(add.index)
    r = r.join(add, how="left")

    # ---- 融资因子（全市场口径，按日期对齐）----
    r = r.join(mg, how="left")
    r["mg_net5"] = r["mg_net5"].ffill()
    r["mg_bal20"] = r["mg_bal20"].ffill()
    r["mg_net10"] = r["mg_net10"].ffill()

    # ---- 指数回撤 ----
    r["dd250"] = r["close"] / r["close"].rolling(250, min_periods=60).max() - 1
    RAW[b] = r

LEVEL_F = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20"]
W = {"b20": .18, "b60": .15, "r5": .11, "nh": .13, "lim": .09, "rsi": .14, "bias": .12, "ret20": .08}
def pmap(v, lq=2, hq=98):
    x = v.astype(float)
    lo = np.nanpercentile(x.values, lq); hi = np.nanpercentile(x.values, hq)
    return (100.0*(x-lo)/(hi-lo)).clip(-25, 125)

for b in C.BOARD_ORDER:
    r = RAW[b]
    s = pd.Series(0.0, index=r.index); tw = 0.0
    for k in LEVEL_F: s = s + pmap(r[k]) * W[k]; tw += W[k]
    r["L"] = s / tw
    r["T"] = pmap(r["bias"])*.30 + pmap(r["ret20"])*.30 + pmap(r["rsi"])*.22 + r["amt_pct"]*100*.18
    r["U"] = 0.5*r["L"] + 0.5*r["T"]
    r["cf_b"] = (((r["amt_pct"]-0.60)/0.40).clip(0,1)*((12-r["L"])/12).clip(0,1)).fillna(0)
    # 新因子标准化（越高越恐慌）
    r["f_deep5"]  = pmap(r["deep5"])
    r["f_deep20"] = pmap(r["deep20"])
    r["f_disp"]   = 100 - pmap(r["disp"])      # 离散度低=同步暴跌=恐慌
    r["f_mg5"]    = 100 - pmap(r["mg_net5"])   # 净买入越负越恐慌
    r["f_mgb20"]  = 100 - pmap(r["mg_bal20"])
    r["f_dd"]     = 100 - pmap(r["dd250"])     # 回撤越深越恐慌

print("="*120)
print("【1】新因子在真底 / 普通日的读数对比（分位 0~100，越高越恐慌）")
print("="*120)
print(f"{'板块':>8}{'真底日':<12}" + "".join(f"{k:>9}" for k in
      ["f_deep5","f_deep20","f_disp","f_mg5","f_mgb20","f_dd","L"]) + f"{'全窗口中位':>12}")
for b in C.BOARD_ORDER:
    r = RAW[b]; bd = r["close"].idxmin()
    print(f"{C.BOARDS[b]['name']:>8}{str(bd.date()):<12}" +
          "".join(f"{r[k].loc[bd]:9.0f}" for k in
                  ["f_deep5","f_deep20","f_disp","f_mg5","f_mgb20","f_dd","L"]) +
          f"{r['L'].median():12.0f}")

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
    pos = {d: i for i, d in enumerate(idx)}; rows = []
    for d in ds:
        i = pos.get(d)
        if i is None or i+h >= len(close): continue
        c0 = close[i]; seg = close[i+1:i+h+1]
        rows.append((d, seg[-1]/c0-1, seg.min()/c0-1, seg.max()/c0-1))
    return rows

print("\n" + "="*120)
print("【2】底部口径对比（严格：20日收益>0 且 最深回撤≥-3%）")
print("="*120)
VARIANTS = {
 "现状 U-25cf_b":              lambda r: r["U"] - 25*r["cf_b"],
 "+去杠杆: U-25cf_b-15·f_mg5/100": lambda r: r["U"] - 25*r["cf_b"] - 15*r["f_mg5"]/100*3,
 "+深跌占比":                    lambda r: r["U"] - 25*r["cf_b"] - 10*r["f_deep5"]/100*3,
 "+回撤":                       lambda r: r["U"] - 25*r["cf_b"] - 10*r["f_dd"]/100*3,
 "+离散度":                     lambda r: r["U"] - 25*r["cf_b"] - 10*r["f_disp"]/100*3,
 "+去杠杆+深跌":                lambda r: r["U"] - 25*r["cf_b"] - 12*r["f_mg5"]/100*3 - 8*r["f_deep5"]/100*3,
 "+去杠杆+深跌+回撤":           lambda r: r["U"] - 25*r["cf_b"] - 10*r["f_mg5"]/100*3 - 7*r["f_deep5"]/100*3 - 7*r["f_dd"]/100*3,
}
for name, fn in VARIANTS.items():
    tn = tok = 0; rets = []; risks = []; allr = []
    for b in C.BOARD_ORDER:
        r = RAW[b]; r["score"] = fn(r)
        ds = ev(b, r["score"] < 0)
        rows = stat(b, ds)
        if not rows: continue
        a = np.array([[x[1], x[2], x[3]] for x in rows])
        ok = (a[:,0] > 0) & (a[:,1] >= -0.03)
        tn += len(a); tok += int(ok.sum()); rets.append(a[:,0].mean()); risks.append(a[:,1].mean())
        allr += [(C.BOARDS[b]['name'],)+x for x in rows]
    print(f"\n  {name}   命中 {tok}/{tn} ({tok/tn*100 if tn else 0:.0f}%)  "
          f"均收益 {np.mean(rets)*100 if rets else 0:+.2f}%  均回撤 {np.mean(risks)*100 if risks else 0:+.2f}%")
    fails = [x for x in allr if not (x[2] > 0 and x[3] >= -0.03)]
    for nm, d, ret, mdd, ru in fails:
        print(f"       FAIL {nm:>8} {str(d.date())}  20日{ret*100:+7.2f}%  最深{mdd*100:+7.2f}%")
    # 是否抓到 2024-09 底
    for nm, d, ret, mdd, ru in allr:
        if str(d.date())[:7] in ("2024-08","2024-09","2024-10"):
            print(f"       2024H2 {'OK ' if (ret>0 and mdd>=-0.03) else 'FAIL'} {nm:>8} {str(d.date())}  20日{ret*100:+7.2f}%  最深{mdd*100:+7.2f}%")
