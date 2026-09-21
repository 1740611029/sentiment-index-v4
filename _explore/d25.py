"""1) 留一法检验（按板块 / 按时间段）看 90% 是否稳
   2) 两个「零新增特征」的标定方案：常数平移 / 确认项死区
      —— 目标：让 0 和 100 这两条线站到置信度最高的位置，且不引入新特征
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
    BASE[b] = raw[["close", "L", "U", "amt_pct", "f_dd", "f_disp"]].copy()

CLOSES = {b: BASE[b]["close"] for b in C.BOARD_ORDER}


def make_score(b, shift=0.0, dead=0.0):
    raw = BASE[b]
    amt_q = raw["amt_pct"].astype(float)
    cf_amt = ((amt_q - 0.60) / 0.40).clip(0, 1).fillna(0.0)
    cf_low = ((12.0 - raw["L"]) / 12.0).clip(0, 1).fillna(0.0)
    cf_b = (cf_amt * cf_low).fillna(0.0)
    g1 = ((raw["f_dd"] - 90.0) / 12.0).clip(0, 1)
    g2 = ((raw["f_disp"] - 75.0) / 12.0).clip(0, 1)
    cf_g = (g1 * g2).fillna(0.0)
    if dead > 0:
        cf_b = ((cf_b - dead) / (1 - dead)).clip(0, 1)
        cf_g = ((cf_g - dead) / (1 - dead)).clip(0, 1)
    return (raw["U"] - 25.0 * cf_b - 30.0 * cf_g).clip(-30.0, 140.0) - shift


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


def collect(boards, shift=0.0, dead=0.0, hi_thr=100.0):
    """返回 (底部明细, 顶部明细)"""
    bd, tp = [], []
    for b in boards:
        sc = make_score(b, shift, dead)
        for d in evts(sc):
            f = fwd(CLOSES[b], d)
            if f: bd.append((b, d, f[0], f[1]))
        for d in evts(sc, hi=True, thr=hi_thr):
            f = fwd(CLOSES[b], d)
            if f: tp.append((b, d, f[0], f[2]))
    return bd, tp


def stat(bd, tp):
    bok = sum(1 for x in bd if x[2] > 0 and x[3] >= -TOL)
    bnt = sum(1 for x in bd if x[3] >= -TOL)
    tok = sum(1 for x in tp if x[2] < 0 and x[3] <= TOL)
    r = lambda a, i: (np.mean([x[i] for x in a]) * 100) if a else float("nan")
    return dict(b_n=len(bd), b_ok=bok, b_ntrap=bnt, b_ret=r(bd, 2), b_mdd=r(bd, 3),
                t_n=len(tp), t_ok=tok, t_ret=r(tp, 2))


def show(tag, s):
    bp = f"{s['b_ok']}/{s['b_n']}" if s["b_n"] else "-"
    nt = f"{s['b_ntrap']}/{s['b_n']}" if s["b_n"] else "-"
    tp = f"{s['t_ok']}/{s['t_n']}" if s["t_n"] else "-"
    print(f"  {tag:<26} 底 {bp:>6} ({s['b_ok']/s['b_n']*100 if s['b_n'] else 0:.0f}%)  "
          f"不被套 {nt:>6}   均收益 {s['b_ret']:+6.2f}%   |  顶 {tp:>5}")


ALL = C.BOARD_ORDER
print("\n" + "=" * 124)
print("【1】留一板块法：每次去掉一个板块，看剩余板块的命中率")
print("=" * 124)
show("全部 6 板块", stat(*collect(ALL)))
for b in ALL:
    sub = [x for x in ALL if x != b]
    show(f"去掉 {C.BOARDS[b]['name']}", stat(*collect(sub)))

print("\n" + "=" * 124)
print("【2】分时间段：近 3 年切两半，看是否只是某一段好看")
print("=" * 124)
bd, tp = collect(ALL)
half = pd.Timestamp("2025-02-01")
for tag, sel in (("2023-09 ~ 2025-01", lambda x: x[1] < half),
                 ("2025-02 ~ 2026-09", lambda x: x[1] >= half)):
    show(tag, stat([x for x in bd if sel(x)], [x for x in tp if sel(x)]))

print("\n" + "=" * 124)
print("【3】常数平移：score − s，然后仍用 0 / 100 作边界（等价于把边界整体上移 s）")
print("=" * 124)
for s in [0, 1, 2, 3, 4, 5, 6]:
    show(f"shift = {s}", stat(*collect(ALL, shift=float(s))))

print("\n" + "=" * 124)
print("【4】确认项死区：确认强度低于 dead 的不计入（控制论里的标准死区，非新特征）")
print("=" * 124)
for dd in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    show(f"dead = {dd:.2f}", stat(*collect(ALL, dead=dd)))

print("\n" + "=" * 124)
print("【5】死区 × 平移 联合（看 plateau）")
print("=" * 124)
print("            " + "".join(f"{'d='+format(d,'.2f'):>10}" for d in [0.0, 0.10, 0.15, 0.20, 0.25]))
for s in [0, 2, 3]:
    row = []
    for d in [0.0, 0.10, 0.15, 0.20, 0.25]:
        st = stat(*collect(ALL, shift=float(s), dead=d))
        row.append(f"{st['b_ok']}/{st['b_n']}" if st["b_n"] else "-")
    print(f"  shift={s}   " + "".join(f"{c:>10}" for c in row))
