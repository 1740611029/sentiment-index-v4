"""修正 d25 的两个错误后重测：
   1) 收紧方向：score + s（s>0）等价于把 0 线提到 −s；d25 里 shift 方向弄反了
   2) 死区应为「截断式」：cf < dead 直接归零，cf >= dead 保持原值（不缩放），
      这样只砍掉弱确认，强确认力度不变
   并打印每个事件的 cf_b / cf_g，看 3 个 FAIL 到底是弱确认还是强确认。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors, model

MAIN_H, TOL, GAP = 20, 0.03, 20

print("building base ...", flush=True)
stock_ind = data.build_stock_indicators()
BASE = {}
for b in C.BOARD_ORDER:
    raw = factors.build_board_raw(b, stock_ind)
    raw = raw[raw.index >= pd.Timestamp(C.BACKTEST_START)].copy()
    idx = data.load_index(b).set_index("date")
    for col in ("open", "high", "low"):
        raw[col] = idx[col].reindex(raw.index)
    lo_q, hi_q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
    lc, hc = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
    pm = model._pct_map
    num = pd.Series(0.0, index=raw.index); den = 0.0
    for k, w in model.L_WEIGHTS.items():
        num = num + pm(raw[k], lo_q, hi_q, lc, hc) * w; den += w
    raw["L"] = num / den
    raw["T"] = (pm(raw["bias"], lo_q, hi_q, lc, hc) * model.T_WEIGHTS["bias"]
                + pm(raw["ret20"], lo_q, hi_q, lc, hc) * model.T_WEIGHTS["ret20"]
                + pm(raw["rsi"], lo_q, hi_q, lc, hc) * model.T_WEIGHTS["rsi"]
                + raw["amt_pct"] * 100.0 * model.T_WEIGHTS["amt"])
    raw["U"] = model.W_L * raw["L"] + (1 - model.W_L) * raw["T"]
    raw["dd250"] = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    raw["f_disp"] = (100.0 - pm(raw["disp"], lo_q, hi_q, lc, hc)).fillna(0.0)
    raw["f_dd"] = (100.0 - pm(raw["dd250"], lo_q, hi_q, lc, hc)).fillna(0.0)
    raw["cf_b"] = (((raw["amt_pct"].astype(float) - 0.60) / 0.40).clip(0, 1).fillna(0.0)
                   * ((12.0 - raw["L"]) / 12.0).clip(0, 1).fillna(0.0)).fillna(0.0)
    raw["cf_g"] = ((((raw["f_dd"] - 90.0) / 12.0).clip(0, 1))
                   * (((raw["f_disp"] - 75.0) / 12.0).clip(0, 1))).fillna(0.0)
    BASE[b] = raw[["close", "U", "L", "amt_pct", "cf_b", "cf_g"]].copy()

CLOSES = {b: BASE[b]["close"] for b in C.BOARD_ORDER}


def make_score(b, lift=0.0, dead=0.0):
    raw = BASE[b]
    cfb, cfg = raw["cf_b"], raw["cf_g"]
    if dead > 0:                       # 截断式死区：弱确认归零，强确认不动
        cfb = cfb.where(cfb >= dead, 0.0)
        cfg = cfg.where(cfg >= dead, 0.0)
    return (raw["U"] - 25.0 * cfb - 30.0 * cfg).clip(-30.0, 140.0) + lift


def evts(score, hi=False, thr=0.0):
    s = score.dropna()
    m = (s > thr) if hi else (s < thr)
    if not m.any():
        return []
    grp = (m != m.shift()).cumsum()
    picks = sorted({(seg.idxmax() if hi else seg.idxmin()) for _, seg in s[m].groupby(grp[m])})
    res = []
    for d in picks:
        if res and (d - res[-1]).days <= GAP:
            continue
        res.append(d)
    return res


def fwd(close, d, h=MAIN_H):
    i = close.index.get_loc(d)
    if i + h >= len(close):
        return None
    c0 = close.iloc[i]; seg = close.iloc[i + 1:i + h + 1]
    return seg.iloc[-1] / c0 - 1, seg.min() / c0 - 1, seg.max() / c0 - 1


def collect(lift=0.0, dead=0.0, boards=None):
    bd, tp = [], []
    for b in (boards or C.BOARD_ORDER):
        sc = make_score(b, lift, dead)
        for d in evts(sc):
            f = fwd(CLOSES[b], d)
            if f: bd.append((b, d, f[0], f[1]))
        for d in evts(sc, hi=True, thr=100.0):
            f = fwd(CLOSES[b], d)
            if f: tp.append((b, d, f[0], f[2]))
    return bd, tp


def stat(bd, tp):
    bok = sum(1 for x in bd if x[2] > 0 and x[3] >= -TOL)
    bnt = sum(1 for x in bd if x[3] >= -TOL)
    tok = sum(1 for x in tp if x[2] < 0 and x[3] <= TOL)
    r = lambda a, i: (np.mean([x[i] for x in a]) * 100) if a else float("nan")
    return dict(b_n=len(bd), b_ok=bok, b_ntrap=bnt, b_ret=r(bd, 2),
                t_n=len(tp), t_ok=tok)


def show(tag, s):
    bp = f"{s['b_ok']}/{s['b_n']}" if s["b_n"] else "-"
    nt = f"{s['b_ntrap']}/{s['b_n']}" if s["b_n"] else "-"
    tp = f"{s['t_ok']}/{s['t_n']}" if s["t_n"] else "-"
    print(f"  {tag:<24} 底 {bp:>6} ({s['b_ok']/s['b_n']*100 if s['b_n'] else 0:3.0f}%)  "
          f"不被套 {nt:>6} ({s['b_ntrap']/s['b_n']*100 if s['b_n'] else 0:3.0f}%)  "
          f"均收益 {s['b_ret']:+6.2f}%   |  顶 {tp:>5}")


# ------------------------------------------------- 事件明细（含 cf_b / cf_g）
print("\n" + "=" * 128)
print("事件明细：3 个 FAIL 究竟是弱确认还是强确认？")
print("=" * 128)
bd, tp = collect()
print(f"{'板块':>8} {'日期':<11} {'':<5} {'20日%':>7} {'最深%':>7} {'U':>6} {'L':>6} {'cf_b':>6} {'cf_g':>6}")
for b, d, r, m in sorted(bd, key=lambda x: (x[2] > 0 and x[3] >= -TOL, x[1])):
    raw = BASE[b]
    ok = r > 0 and m >= -TOL
    print(f"{C.BOARDS[b]['name']:>8} {str(d.date()):<11} {'OK  ' if ok else 'FAIL':<5} "
          f"{r*100:+7.2f} {m*100:+7.2f} {raw['U'].loc[d]:6.1f} {raw['L'].loc[d]:+6.1f} "
          f"{raw['cf_b'].loc[d]:6.3f} {raw['cf_g'].loc[d]:6.3f}")

print("\n" + "=" * 128)
print("【A】收紧方向（正确版）：score + lift，等价于把 0 线提到 −lift")
print("=" * 128)
for s in [0, 1, 2, 3, 4, 5, 6, 8, 10]:
    show(f"0 线 = −{s}", stat(*collect(lift=float(s))))

print("\n" + "=" * 128)
print("【B】截断式死区（正确版）：cf < dead 归零，cf ≥ dead 原值不动")
print("=" * 128)
for dd in [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]:
    show(f"dead = {dd:.2f}", stat(*collect(dead=dd)))

print("\n" + "=" * 128)
print("【C】死区 × 收紧 联合（找 plateau）")
print("=" * 128)
ds = [0.0, 0.05, 0.10, 0.15, 0.20]
print("            " + "".join(f"{'d='+format(x,'.2f'):>11}" for x in ds))
for s in [0, 1, 2, 3]:
    row = []
    for x in ds:
        st = stat(*collect(lift=float(s), dead=x))
        row.append(f"{st['b_ok']}/{st['b_n']}" if st["b_n"] else "-")
    print(f"  0线=−{s}      " + "".join(f"{c:>11}" for c in row))

print("\n" + "=" * 128)
print("【D】选定方案再做留一板块 / 分时段（0线=−3, dead=0）")
print("=" * 128)
show("全部", stat(*collect(lift=3.0)))
for b in C.BOARD_ORDER:
    sub = [x for x in C.BOARD_ORDER if x != b]
    show(f"去掉 {C.BOARDS[b]['name']}", stat(*collect(lift=3.0, boards=sub)))
half = pd.Timestamp("2025-02-01")
b2, t2 = collect(lift=3.0)
for tag, sel in (("2023-09~2025-01", lambda x: x[1] < half), ("2025-02~2026-09", lambda x: x[1] >= half)):
    show(tag, stat([x for x in b2 if sel(x)], [x for x in t2 if sel(x)]))
