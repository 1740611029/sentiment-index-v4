"""诊断 15：顶部改用「顶背离」口径 —— 价格强、广度弱。
关键：必须能区分两次顶
  2024-10-08  924 全民狂热：价格高 + 广度也高（99 分位）→ 背离小
  2026-05/06  无声顶：      价格高 + 广度已塌（创业板 b20 仅 31 分位）→ 背离大
背离量必须连续，才能画进折线。
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
    RAW[b] = r

def rk(s, n=250, mp=60):
    """滚动分位（只用历史，含当日）"""
    return s.rolling(n, min_periods=mp).rank(pct=True) * 100

for b in C.BOARD_ORDER:
    r = RAW[b]
    c = r["close"]
    # 价格位置：收盘价在自身 250 日区间中的百分位（连续）
    r["p_pos"] = rk(c, 250)
    # 广度位置：站上MA20占比 在自身 250 日区间中的百分位（连续）
    r["b_pos"] = rk(r["b20"], 250)
    # 顶背离 = 价格位置 − 广度位置   (越大越背离)
    r["div"] = r["p_pos"] - r["b_pos"]
    r["div125"] = rk(c, 125) - rk(r["b20"], 125)
    r["div60"] = rk(c, 60) - rk(r["b20"], 60)
    # 另一版本：用 b60（更慢的广度）
    r["div_b60"] = rk(c, 250) - rk(r["b60"], 250)

print("="*122)
print("【1】两次顶的背离对比（div = 价格位置 − 广度位置，250日滚动分位）")
print("="*122)
print(f"{'板块':>8} {'真顶日':<12}{'div':>7}{'p_pos':>7}{'b_pos':>7} | "
      f"{'2024-10-08':<12}{'div':>7} | {'div最高3天':<44}")
for b in C.BOARD_ORDER:
    r = RAW[b]; td = r["close"].idxmax()
    o8 = pd.Timestamp("2024-10-08")
    near = r.index[(r.index >= o8 - pd.Timedelta(days=5)) & (r.index <= o8 + pd.Timedelta(days=5))]
    v8 = r.loc[near, "div"].max() if len(near) else np.nan
    hi = r["div"].nlargest(3)
    print(f"{C.BOARDS[b]['name']:>8} {str(td.date()):<12}{r['div'].loc[td]:7.1f}"
          f"{r['p_pos'].loc[td]:7.1f}{r['b_pos'].loc[td]:7.1f} | "
          f"{'':<12}{v8:7.1f} | " + " ".join(f"{str(d.date())[2:]}({v:.0f})" for d, v in hi.items()))

print("\n" + "="*122)
print("【2】背离极值日的未来表现（T+20 / T+40 / T+60，冷却 30 天）")
print("="*122)
def ev(b, mask, how="max", gap=30, col="div"):
    r = RAW[b]; m = mask.fillna(False)
    if not m.any(): return []
    x = r[col]; grp = (m != m.shift()).cumsum()
    picks = [seg.idxmin() if how == "min" else seg.idxmax() for _, seg in x[m].groupby(grp[m])]
    picks = sorted(set(picks)); out = []
    for d in picks:
        if out and (d - out[-1]).days <= gap: continue
        out.append(d)
    return out

def stat(b, ds, h):
    r = RAW[b]; close = r["close"].values; idx = r.index
    pos = {d: i for i, d in enumerate(idx)}; rows = []
    for d in ds:
        i = pos.get(d)
        if i is None or i + h >= len(close): continue
        c0 = close[i]; seg = close[i+1:i+h+1]
        rows.append((d, seg[-1]/c0-1, seg.min()/c0-1, seg.max()/c0-1))
    return rows

CANDS = {
    "div>=60":          lambda r: r["div"] >= 60,
    "div>=70":          lambda r: r["div"] >= 70,
    "div>=80":          lambda r: r["div"] >= 80,
    "div>=70 & p_pos>=90": lambda r: (r["div"]>=70)&(r["p_pos"]>=90),
    "div>=60 & p_pos>=90": lambda r: (r["div"]>=60)&(r["p_pos"]>=90),
    "div125>=70":       lambda r: r["div125"] >= 70,
    "div60>=70":        lambda r: r["div60"] >= 70,
    "div_b60>=70":      lambda r: r["div_b60"] >= 70,
}
for name, fn in CANDS.items():
    print(f"\n  {name}")
    print(f"    {'窗口':<8}" + "".join(f"{C.BOARDS[b]['name']:>8}" for b in C.BOARD_ORDER)
          + "   合计      均收益   均踏空")
    for h in (20, 40, 60):
        line = f"    T+{h:<6}"; tn = tok = 0; rets = []; runs = []; allr = []
        for b in C.BOARD_ORDER:
            ds = ev(b, fn(RAW[b]))
            rows = stat(b, ds, h)
            if not rows: line += f"{'-':>8}"; continue
            a = np.array([[x[1], x[2], x[3]] for x in rows])
            ok = (a[:, 0] < 0) & (a[:, 2] <= 0.05)
            tn += len(a); tok += int(ok.sum()); rets.append(a[:,0].mean()); runs.append(a[:,2].mean())
            allr += [(C.BOARDS[b]['name'],)+x for x in rows]
            line += f"{f'{int(ok.sum())}/{len(a)}':>8}"
        rr = f"{np.mean(rets)*100:+6.2f}%" if rets else "   -   "
        rk_ = f"{np.mean(runs)*100:+6.2f}%" if runs else "   -   "
        print(line + f"   {tok}/{tn}" + (f"({tok/tn*100:3.0f}%)" if tn else "     ") + f"   {rr}   {rk_}")
        if h == 20:
            for nm, d, ret, mdd, ru in allr:
                print(f"          {'OK ' if (ret<0 and ru<=0.05) else 'FAIL'} {nm:>8} {str(d.date())}"
                      f"  20日{ret*100:+7.2f}%  最高{ru*100:+7.2f}%")
