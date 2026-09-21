"""验证一个「有原理」的修改，而不是调阈值：

   把 _pct_map 的下限 clip 从 −25 收到 0，即情绪读数 U 严格落在 0~100 内。
   这样 score < 0 只能由确认项（cf_b / cf_g）产生，
   不会再出现「读数自己溢出到负数」的伪信号。

   顶部侧只受 hi_clip 影响，保持 125 不动 → 顶部事件不变。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors, model

MAIN_H, TOL, GAP = 20, 0.03, 20

print("building stock_ind ...", flush=True)
stock_ind = data.build_stock_indicators()

RAW = {}
for b in C.BOARD_ORDER:
    raw = factors.build_board_raw(b, stock_ind)
    raw = raw[raw.index >= pd.Timestamp(C.BACKTEST_START)].copy()
    idx = data.load_index(b).set_index("date")
    for col in ("open", "high", "low"):
        raw[col] = idx[col].reindex(raw.index)
    RAW[b] = raw
print("done", flush=True)


def build(b, lc, hc=125.0):
    raw = RAW[b].copy()
    lo_q, hi_q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
    pm = lambda v: model._pct_map(v, lo_q, hi_q, lc, hc)
    num = pd.Series(0.0, index=raw.index); den = 0.0
    for k, w in model.L_WEIGHTS.items():
        num = num + pm(raw[k]) * w; den += w
    raw["L"] = num / den
    raw["T"] = (pm(raw["bias"]) * model.T_WEIGHTS["bias"]
                + pm(raw["ret20"]) * model.T_WEIGHTS["ret20"]
                + pm(raw["rsi"]) * model.T_WEIGHTS["rsi"]
                + raw["amt_pct"] * 100.0 * model.T_WEIGHTS["amt"])
    raw["U"] = model.W_L * raw["L"] + (1 - model.W_L) * raw["T"]
    raw["dd250"] = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    f_disp = (100.0 - pm(raw["disp"])).fillna(0.0)
    f_dd = (100.0 - pm(raw["dd250"])).fillna(0.0)
    cfb = (((raw["amt_pct"].astype(float) - 0.60) / 0.40).clip(0, 1).fillna(0.0)
           * ((12.0 - raw["L"]) / 12.0).clip(0, 1).fillna(0.0)).fillna(0.0)
    cfg = (((f_dd - 90.0) / 12.0).clip(0, 1) * ((f_disp - 75.0) / 12.0).clip(0, 1)).fillna(0.0)
    raw["cf_b"], raw["cf_g"] = cfb, cfg
    raw["score"] = (raw["U"] - 25.0 * cfb - 30.0 * cfg).clip(-30.0, 140.0)
    return raw


def evts(score, hi=False):
    s = score.dropna(); thr = 100.0 if hi else 0.0
    m = (s > thr) if hi else (s < thr)
    if not m.any():
        return []
    grp = (m != m.shift()).cumsum()
    picks = sorted({(seg.idxmax() if hi else seg.idxmin()) for _, seg in s[m].groupby(grp[m])})
    res = []
    for d in picks:
        if res and (d - res[-1]).days <= GAP: continue
        res.append(d)
    return res


def fwd(close, d, h=MAIN_H):
    i = close.index.get_loc(d)
    if i + h >= len(close): return None
    c0 = close.iloc[i]; seg = close.iloc[i + 1:i + h + 1]
    return seg.iloc[-1] / c0 - 1, seg.min() / c0 - 1, seg.max() / c0 - 1


def run(lc, boards=None, verbose=False):
    bd, tp = [], []
    P = {}
    for b in (boards or C.BOARD_ORDER):
        p = build(b, lc); P[b] = p
        cl = p["close"]
        for d in evts(p["score"]):
            f = fwd(cl, d)
            if f: bd.append((b, d, f[0], f[1]))
        for d in evts(p["score"], hi=True):
            f = fwd(cl, d)
            if f: tp.append((b, d, f[0], f[2]))
    bok = sum(1 for x in bd if x[2] > 0 and x[3] >= -TOL)
    bnt = sum(1 for x in bd if x[3] >= -TOL)
    tok = sum(1 for x in tp if x[2] < 0 and x[3] <= TOL)
    mr = lambda a, i: (np.mean([x[i] for x in a]) * 100) if a else float("nan")
    if verbose:
        print(f"{'板块':>8} {'日期':<11} {'':<5} {'20日%':>7} {'最深%':>7} {'U':>7} {'cf_b':>6} {'cf_g':>6} {'score':>7}")
        for b, d, r, m in sorted(bd, key=lambda x: (x[2] > 0 and x[3] >= -TOL, x[1])):
            p = P[b]; ok = r > 0 and m >= -TOL
            print(f"{C.BOARDS[b]['name']:>8} {str(d.date()):<11} {'OK  ' if ok else 'FAIL':<5} "
                  f"{r*100:+7.2f} {m*100:+7.2f} {p['U'].loc[d]:7.1f} {p['cf_b'].loc[d]:6.3f} "
                  f"{p['cf_g'].loc[d]:6.3f} {p['score'].loc[d]:7.1f}")
    return dict(b_n=len(bd), b_ok=bok, b_ntrap=bnt, b_ret=mr(bd, 2), b_mdd=mr(bd, 3),
                t_n=len(tp), t_ok=tok, t_ret=mr(tp, 2))


def show(tag, s):
    bp = f"{s['b_ok']}/{s['b_n']}" if s["b_n"] else "-"
    nt = f"{s['b_ntrap']}/{s['b_n']}" if s["b_n"] else "-"
    tp = f"{s['t_ok']}/{s['t_n']}" if s["t_n"] else "-"
    print(f"  {tag:<22} 底 {bp:>6} ({s['b_ok']/s['b_n']*100 if s['b_n'] else 0:3.0f}%)  "
          f"不被套 {nt:>6} ({s['b_ntrap']/s['b_n']*100 if s['b_n'] else 0:3.0f}%)  "
          f"均收益 {s['b_ret']:+6.2f}%  顶 {tp:>5}")


print("\n" + "=" * 122)
print("映射下限 lc 敏感性（hi_clip 固定 125，顶部不受影响）")
print("=" * 122)
for lc in [-25.0, -20.0, -15.0, -10.0, -5.0, -2.0, 0.0, 3.0]:
    show(f"lc = {lc:+6.1f}", run(lc))

print("\n" + "=" * 122)
print("选定 lc = 0 的完整底部明细")
print("=" * 122)
run(0.0, verbose=True)

print("\n" + "=" * 122)
print("lc = 0 的留一板块 / 分时段检验")
print("=" * 122)
show("全部 6 板块", run(0.0))
for b in C.BOARD_ORDER:
    show(f"去掉 {C.BOARDS[b]['name']}", run(0.0, boards=[x for x in C.BOARD_ORDER if x != b]))
half = pd.Timestamp("2025-02-01")
s1 = run(0.0, boards=[b for b in C.BOARD_ORDER])
# 分时段需要重跑取明细
for tag, lo, hi in (("2023-09~2025-01", None, half), ("2025-02~2026-09", half, None)):
    bd, tp = [], []
    for b in C.BOARD_ORDER:
        p = build(b, 0.0); cl = p["close"]
        for d in evts(p["score"]):
            if lo is not None and d < lo: continue
            if hi is not None and d >= hi: continue
            f = fwd(cl, d)
            if f: bd.append((b, d, f[0], f[1]))
        for d in evts(p["score"], hi=True):
            if lo is not None and d < lo: continue
            if hi is not None and d >= hi: continue
            f = fwd(cl, d)
            if f: tp.append((b, d, f[0], f[2]))
    bok = sum(1 for x in bd if x[2] > 0 and x[3] >= -TOL)
    bnt = sum(1 for x in bd if x[3] >= -TOL)
    tok = sum(1 for x in tp if x[2] < 0 and x[3] <= TOL)
    mr = lambda a, i: (np.mean([x[i] for x in a]) * 100) if a else float("nan")
    show(tag, dict(b_n=len(bd), b_ok=bok, b_ntrap=bnt, b_ret=mr(bd, 2),
                   t_n=len(tp), t_ok=tok, t_ret=mr(tp, 2)))
