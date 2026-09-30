# -*- coding: utf-8 -*-
"""t3: 小波段「第 4 模型」自动搜索台。

核心思路（兑现 AGENTS §16 的提示）：
  现役三模型都用「因果锚分位 + (部分)回升确认」的用法。当年 RSI/CCI/威廉/布林%B
  是被「factor<=thr 就买」的**阈值法**证伪的——那是错误用法。这里把它们改造成
  与 SWING-2/SWING-3 同构的**分位 + 拐点**语义，重测。

每个候选都算：
  ① 自身信号级/事件级命中（两窗口：展示窗 & 5.7年）
  ② 与现有三模型并集的 Jaccard 重叠（低重叠才可能有增量）
  ③ 并入并集后的**边际增益**：事件数、事件命中率各变化多少（真正在乎的量）
判定口径与生产一致：T+7 期末>0 且期间最深回撤>=-3%，冷却 4 交易日，逐日模拟因果。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"d:\情绪指标4")
import numpy as np
import pandas as pd
from senti import config as C, store, swing, swing2, swing3, data
from senti.model import _pct_map

LO, HI = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
H, TOL, COOL = 7, 0.03, 4
DISP = pd.Timestamp(C.BACKTEST_START)     # 2023-09-20
L57 = pd.Timestamp("2021-01-01")

def pm(v): return _pct_map(v, LO, HI, LC, HC)

# ---------- 候选因子工厂：返回「越低越超卖」的原始序列 x ----------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)
def wpr(c, n=14):
    hh = c.rolling(n).max(); ll = c.rolling(n).min()
    return (hh - c) / (hh - ll).replace(0, np.nan) * 100     # 越高越超卖？反转
def cci(c, n=20):
    tp = c; ma = tp.rolling(n).mean()
    md = (tp - ma).abs().rolling(n).mean().replace(0, np.nan)
    return (tp - ma) / (0.015 * md)
def pctb(c, n=20):
    ma = c.rolling(n).mean(); sd = c.rolling(n).std()
    return (c - (ma - 2 * sd)) / (4 * sd).replace(0, np.nan)
def stochd(c, n=14):
    hh = c.rolling(n).max(); ll = c.rolling(n).min()
    k = (c - ll) / (hh - ll).replace(0, np.nan) * 100
    return k.rolling(3).mean()
def macd_slope(c):
    hist = ema(c, 12) - ema(c, 26); hist = hist - ema(hist, 9)
    return hist - hist.shift(5)
def accel(c):
    r = c.pct_change(); return (r - r.shift(5))

CAND = {
    "rsi14":        lambda c: rsi(c),
    "wpr14_low":    lambda c: -wpr(c),          # wpr 高=超卖,取负使低=超卖
    "cci20":        lambda c: cci(c),
    "bb_pctb20":    lambda c: pctb(c),
    "stochd":       lambda c: stochd(c),
    "ret5":         lambda c: c.pct_change(5),
    "ret10":        lambda c: c.pct_change(10),
    "bias20":       lambda c: c / c.rolling(20).mean() - 1,
    "bias60":       lambda c: c / c.rolling(60).mean() - 1,
    "dist_high120": lambda c: c / c.rolling(120).max() - 1,
    "macd_hist_slop": lambda c: macd_slope(c),
    "accel5":       lambda c: accel(c),
    "voladj_m20":   lambda c: (c.pct_change(20) /
                               c.pct_change().rolling(20).std().replace(0, np.nan)),
}

# ---------- 加载全历史 close（含预热） ----------
CLOSE = {}
for b in C.BOARD_ORDER:
    idx = data.load_index(b).set_index("date").sort_index()
    CLOSE[b] = idx["close"].astype(float)

# ---------- 现有三模型信号（用于重叠/边际） ----------
sp, s2p, s3p = swing.load(), swing2.load(), swing3.load()
sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
UEX = {b: store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r)
       for b in C.BOARD_ORDER}
ex_dates = {b: {e["date"] for e in UEX[b]} for b in C.BOARD_ORDER}

def sim_board(c, trig):
    """给定 close(全史索引) 与 bool 触发序列，逐日模拟，返回 [(date, ok)] 已判定的。"""
    idx = c.index; n = len(idx)
    cv = c.to_numpy(float); tv = trig.reindex(idx).fillna(False).to_numpy(bool)
    out, last = [], -10**9
    for i in range(n):
        if not tv[i] or i - last < COOL or i + H > n - 1:
            continue
        last = i
        seg = cv[i+1:i+H+1]
        out.append((str(idx[i].date()), bool(seg[-1]/cv[i]-1 > 0 and seg.min()/cv[i]-1 >= -TOL)))
    return out

def evaluate(xfn, recovery=True, thr=10.0):
    per_board = {}
    for b in C.BOARD_ORDER:
        c = CLOSE[b]
        s = pm(xfn(c))
        trig = (s <= thr) if not recovery else ((s <= thr) & (s > s.shift(1)))
        per_board[b] = sim_board(c, trig)
    return per_board

def cluster_dates(rows):
    """同板块内相隔<=4自然日只算第一枪（事件级）。rows=[(datestr,ok)]。"""
    evs = sorted(rows, key=lambda x: x[0])
    last = None; out = []
    for d, ok in evs:
        dt = pd.Timestamp(d)
        if last is None or (dt - last).days > 4:
            out.append((d, ok)); last = dt
    return out

def summarize(per_board, window):
    sig = [(b, d, ok) for b, rows in per_board.items() for (d, ok) in rows
           if pd.Timestamp(d) >= window]
    ev = []
    for b in {x[0] for x in sig}:
        ev += cluster_dates([(d, ok) for (bb, d, ok) in sig if bb == b])
    nsig = len(sig); ksig = sum(1 for _b, _d, ok in sig if ok)
    nev = len(ev); kev = sum(1 for _, ok in ev if ok)
    return (nsig, round(ksig/nsig*100, 1) if nsig else None,
            nev, round(kev/nev*100, 1) if nev else None, sig, ev)

# 现有并集基线
def union_stats(per_board, window):
    """把候选信号并入现有并集，重算事件级 n/命中。"""
    combined = {b: [(d, ok) for (d, ok) in per_board[b]] for b in C.BOARD_ORDER}
    for b in C.BOARD_ORDER:
        combined[b] = [(e["date"], e["ok"]) for e in UEX[b]
                       if pd.Timestamp(e["date"]) >= window] + \
                      [(d, ok) for d, ok in combined[b] if pd.Timestamp(d) >= window]
    ev = []
    for b in C.BOARD_ORDER:
        ev += cluster_dates(combined[b])
    n = len(ev); k = sum(1 for _, ok in ev if ok)
    return n, round(k/n*100, 1) if n else None

# 现有并集事件基线
base_n, base_k = union_stats({b: [] for b in C.BOARD_ORDER}, DISP)
print(f"现有并集(展示窗)事件级基线: {base_n} 件 / {base_k}%\n")

print("候选                                    窗  信号n/率   事件n/率   重叠Jacc  并入后事件n/率(Δ件 Δ率)")
print("-"*104)
for name, fn in CAND.items():
    pb = evaluate(fn, recovery=True, thr=10.0)
    ns, rs, ne, re_, sig, ev = summarize(pb, DISP)
    cd = {b: {d for (d, _o) in pb[b]} for b in C.BOARD_ORDER}
    inter = sum(len(cd[b] & ex_dates.get(b, set())) for b in C.BOARD_ORDER)
    unionset = sum(len(cd[b] | ex_dates.get(b, set())) for b in C.BOARD_ORDER)
    jac = round(inter / unionset, 2) if unionset else 0
    nn, nrate = union_stats(pb, DISP)
    dcnt = nn - base_n; drate = None if (nrate is None or base_k is None) else round(nrate - base_k, 1)
    print(f"{name:<38} 展示 {ns:>4}/{rs}%  {ne:>4}/{re_}%   J={jac:<5}   {nn:>4}/{nrate}% (Δ{dcnt:+d}, Δ{drate})")
