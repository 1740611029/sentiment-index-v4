"""参数敏感性检验：底部 26/29 到底是不是调参调出来的巧合？

做法：把与待检参数无关的中间量（L / U / amt_pct / f_dd / f_disp / close）先算好缓存，
然后在参数网格上快速重算 cf_b、cf_g、score 并统计命中率。
这样一次缓存就能扫几百组参数。
"""
import sys, os, itertools, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors, model

MAIN_H = 20
TOL = 0.03
GAP = 20

# ---------------------------------------------------------------- 缓存中间量
print("building base panels ...", flush=True)
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
    print(f"  {C.BOARDS[b]['name']:>8} {len(raw)} 行", flush=True)

CLOSES = {b: BASE[b]["close"] for b in C.BOARD_ORDER}


def fwd(close, d, h=MAIN_H):
    i = close.index.get_loc(d)
    if i + h >= len(close):
        return None
    c0 = close.iloc[i]; seg = close.iloc[i + 1:i + h + 1]
    return seg.iloc[-1] / c0 - 1, seg.min() / c0 - 1, seg.max() / c0 - 1


def pick_events(score, thr, gap=GAP):
    """取 score < thr 的连续区间最低点，gap 天内合并。"""
    s = score.dropna()
    m = s < thr
    if not m.any():
        return []
    grp = (m != m.shift()).cumsum()
    picks = [seg.idxmin() for _, seg in s[m].groupby(grp[m])]
    picks = sorted(set(picks)); res = []
    for d in picks:
        if res and (d - res[-1]).days <= gap:
            continue
        res.append(d)
    return res


def score_of(b, p):
    """p = 参数字典 -> 该板块 score 序列"""
    raw = BASE[b]
    amt_q = raw["amt_pct"].astype(float)
    cf_amt = ((amt_q - p["cf_amt_lo"]) / (1.0 - p["cf_amt_lo"])).clip(0, 1).fillna(0.0)
    cf_low = ((p["cf_l_hi"] - raw["L"]) / p["cf_l_hi"]).clip(0, 1).fillna(0.0)
    cf_b = (cf_amt * cf_low).fillna(0.0)
    g1 = ((raw["f_dd"] - p["dd_a"]) / p["dd_w"]).clip(0, 1)
    g2 = ((raw["f_disp"] - p["dp_a"]) / p["dp_w"]).clip(0, 1)
    cf_g = (g1 * g2).fillna(0.0)
    return (raw["U"] - p["k_b"] * cf_b - p["k_g"] * cf_g).clip(-30.0, 140.0), cf_b, cf_g


def evaluate(p, thr=0.0, verbose=False):
    """返回 (b_ok, b_n, 平均20日收益, 平均最深回撤, 被套>-3%的个数, 明细)"""
    ok = n = 0; rets = []; mdds = []; detail = []
    for b in C.BOARD_ORDER:
        sc, cfb, cfg = score_of(b, p)
        for d in pick_events(sc, thr):
            f = fwd(CLOSES[b], d)
            if f is None:
                continue
            n += 1
            good = f[0] > 0 and f[1] >= -TOL
            ok += int(good)
            rets.append(f[0]); mdds.append(f[1])
            detail.append((b, d, good, f[0], f[1], sc.loc[d], cfb.loc[d], cfg.loc[d]))
    return ok, n, float(np.mean(rets)), float(np.mean(mdds)), detail


P0 = dict(k_b=25.0, cf_amt_lo=0.60, cf_l_hi=12.0, k_g=30.0,
          dd_a=90.0, dd_w=12.0, dp_a=75.0, dp_w=12.0)

print("\n" + "=" * 100)
print("基准参数（当前生产值）")
print("=" * 100)
o, n, r, m, det = evaluate(P0)
print(f"  底部 {o}/{n} = {o/n*100:.0f}%   均收益 {r*100:+.2f}%   均最深回撤 {m*100:+.2f}%")

# ---------------------------------------------------- 单参数扰动（其余冻结）
print("\n" + "=" * 100)
print("单参数扰动敏感性（只动一个，其余冻结）—— 看 26/29 有多脆")
print("=" * 100)
GRID = {
    "k_b":       [0, 10, 15, 20, 25, 30, 35, 40, 50],
    "k_g":       [0, 10, 20, 25, 30, 35, 40, 50],
    "cf_amt_lo": [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80],
    "cf_l_hi":   [5, 8, 10, 12, 15, 20, 25],
    "dd_a":      [80, 85, 90, 95, 100],
    "dd_w":      [6, 9, 12, 18, 24],
    "dp_a":      [65, 70, 75, 80, 85],
    "dp_w":      [6, 9, 12, 18, 24],
}
for key, vals in GRID.items():
    line = []
    for v in vals:
        p = dict(P0); p[key] = v
        o, n, r, m, _ = evaluate(p)
        line.append(f"{v}:{o}/{n}")
    print(f"  {key:>10}  " + "  ".join(line))

# ---------------------------------------------------- 事件阈值
print("\n" + "=" * 100)
print("事件阈值敏感性：score < thr 才算触发（当前 thr=0）")
print("=" * 100)
for thr in [0, -1, -2, -3, -4, -5, -6, -8, -10]:
    o, n, r, m, det = evaluate(P0, thr=thr)
    if n == 0:
        print(f"  thr<{thr:>4}   无事件")
        continue
    deep = sum(1 for x in det if x[4] >= -TOL)
    print(f"  thr<{thr:>4}   {o}/{n} = {o/n*100:5.0f}%   均收益 {r*100:+6.2f}%   "
          f"最深回撤≥-3% 的 {deep}/{n} = {deep/n*100:.0f}%")

# ---------------------------------------------------- 联合网格（粗）
print("\n" + "=" * 100)
print("联合网格 k_b × k_g（其余冻结）—— 看 plateau 有多宽")
print("=" * 100)
print("        " + "".join(f"{g:>8}" for g in [0, 15, 20, 25, 30, 35, 40]))
for kb in [0, 15, 20, 25, 30, 35, 40]:
    cells = []
    for kg in [0, 15, 20, 25, 30, 35, 40]:
        p = dict(P0); p["k_b"] = kb; p["k_g"] = kg
        o, n, _, _, _ = evaluate(p)
        cells.append(f"{o}/{n}" if n else "-")
    print(f"  k_b={kb:>3} " + "".join(f"{c:>8}" for c in cells))
