"""d74：回答两个用户问题
  Q1 这个模型是不是大波段？一年到底出几次信号？（频率 / 间隔 / 一年被占用多少交易日）
  Q4 入场点必须在 0 附近吗？有没有胜率更高的分值？（阈值扫描 + 长历史验证 + 随机基线）

铁律：任何结论必须过 2015-2016（长历史 8 宽基），并配随机基线。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C
from senti import model as M
from senti import factors as F
import longhist

H, H2, COOL, TOL = 20, 60, 20, 0.03
W2 = pd.Timestamp("2023-09-20")
LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
LO_Q, HI_Q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
EXCLUDE = {"CSI2000"}
KEYS = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20", "disp", "dd250"]
WIN, MINP = C.MODEL["anchor_window"], C.MODEL["anchor_min"]
CLIP_LO, CLIP_HI = -30.0, 140.0
rng = np.random.default_rng(20260921)


def make_anchors(raw, mode):
    src = raw.copy()
    src["dd250"] = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    if mode == "trail":
        return {k: (src[k].rolling(WIN, min_periods=MINP).quantile(LO_Q / 100.0).shift(1),
                    src[k].rolling(WIN, min_periods=MINP).quantile(HI_Q / 100.0).shift(1)) for k in KEYS}
    out = {}
    for k in KEYS:
        x = src[k].to_numpy(float); n = len(x)
        lo = np.full(n, np.nan); hi = np.full(n, np.nan); j = 500
        while j < n:
            seg = x[:j]; seg = seg[~np.isnan(seg)]
            if len(seg) >= 500:
                e = min(j + 20, n)
                lo[j:e] = np.nanpercentile(seg, LO_Q); hi[j:e] = np.nanpercentile(seg, HI_Q)
            j += 20
        out[k] = (pd.Series(lo, index=src.index).ffill(), pd.Series(hi, index=src.index).ffill())
    return out


def pmm(v, a, k):
    lo, hi = a[k]; d = (hi - lo).replace(0.0, np.nan)
    return (100.0 * (v - lo) / d).clip(LC, HC)


def score(raw, a):
    num = pd.Series(0.0, index=raw.index); den = 0.0
    for k, w in M.L_WEIGHTS.items():
        num = num + pmm(raw[k], a, k) * w; den += w
    L = num / den
    T = (pmm(raw["bias"], a, "bias") * M.T_WEIGHTS["bias"]
         + pmm(raw["ret20"], a, "ret20") * M.T_WEIGHTS["ret20"]
         + pmm(raw["rsi"], a, "rsi") * M.T_WEIGHTS["rsi"]
         + raw["amt_pct"] * 100.0 * M.T_WEIGHTS["amt"])
    U = M.W_L * L + (1 - M.W_L) * T
    cf_low = ((M.CF_L_HI - L) / M.CF_L_HI).clip(0, 1).fillna(0.0)
    cf_amt = ((raw["amt_pct"].astype(float) - M.CF_AMT_LO) / (1.0 - M.CF_AMT_LO)).clip(0, 1).fillna(0.0)
    cfb = (cf_amt * cf_low).fillna(0.0)
    return (U - M.K_B * cfb).clip(CLIP_LO, CLIP_HI)


def signals(panels, thr):
    S = pd.DataFrame({nm: p["score"] for nm, p in panels.items()}).dropna(how="all")
    CL = pd.DataFrame({nm: p["close"] for nm, p in panels.items()}).reindex(S.index)
    rows = []
    for nm in panels:
        v = S[nm].to_numpy(float); cl = CL[nm].to_numpy(float)
        n = len(v); last = -10 ** 9
        for i in range(1, n - H2):
            if np.isnan(v[i]) or np.isnan(v[i - 1]): continue
            if i - last < COOL: continue
            if v[i - 1] > thr or v[i] <= v[i - 1]: continue
            last = i
            c0 = cl[i]
            seg = cl[i + 1:i + H + 1]; seg2 = cl[i + 1:i + H2 + 1]
            rows.append({"board": nm, "date": S.index[i], "prev": v[i - 1], "score": v[i],
                         "ret20": seg[-1] / c0 - 1, "mdd20": seg.min() / c0 - 1,
                         "ret60": seg2[-1] / c0 - 1,
                         "ok20": bool(seg[-1] / c0 - 1 > 0 and seg.min() / c0 - 1 >= -TOL)})
    return pd.DataFrame(rows)


def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def baseline(panels, hp, nsim=4000):
    """随机挑一天买入持有 hp 天，统计 P(收益>0) 与 P(收益>0 且最深回撤>=-3%)"""
    outs = []
    for nm, p in panels.items():
        cl = p["close"].dropna().to_numpy(float); n = len(cl)
        if n < hp + 5: continue
        idx = rng.integers(0, n - hp - 1, nsim)
        a = cl[idx]; segs = np.stack([cl[i + 1:i + hp + 1] for i in idx[:400]])
        outs.append(((segs[:, -1] / a[:400] - 1 > 0).mean(),
                     ((segs[:, -1] / a[:400] - 1 > 0) & (segs.min(axis=1) / a[:400] - 1 >= -TOL)).mean()))
    return np.mean([o[0] for o in outs]) * 100, np.mean([o[1] for o in outs]) * 100


# ---------------- 载入 ----------------
RAW = {b: F.build_board_raw(b) for b in C.BOARD_ORDER}
PP = {b: pd.DataFrame({"score": score(RAW[b], make_anchors(RAW[b], "trail")),
                       "close": RAW[b]["close"]}) for b in C.BOARD_ORDER}
PP = {b: p[p.index >= W2] for b, p in PP.items()}

print("=" * 118)
print("Q1 · 信号频率：是不是大波段？一年出几次？（交付口径：6 板块 × 2023-09-20 起）")
print("=" * 118)
D0 = signals(PP, 0.0)
ndays = len(PP[C.BOARD_ORDER[0]])
years = ndays / 244.0
print(f"窗口 {PP[C.BOARD_ORDER[0]].index[0].date()} ~ {PP[C.BOARD_ORDER[0]].index[-1].date()}"
      f"  {ndays} 交易日 ≈ {years:.2f} 年\n")
print(f"  {'板块':<9}{'信号数':>6}{'次/年':>8}{'平均间隔(交易日)':>16}{'中位间隔':>10}{'持仓占用天数':>12}{'占比':>8}")
occ = 0
for b in C.BOARD_ORDER:
    d = D0[D0["board"] == b].sort_values("date")
    if len(d) == 0:
        print(f"  {b:<9}{0:>6}"); continue
    gaps = d["date"].diff().dt.days.dropna()
    tgap = np.diff([PP[b].index.get_loc(x) for x in d["date"]])
    occ += len(d) * 60
    print(f"  {b:<9}{len(d):>6}{len(d)/years:>8.2f}{tgap.mean() if len(tgap) else 0:>16.0f}"
          f"{np.median(tgap) if len(tgap) else 0:>10.0f}{len(d)*60:>12}{len(d)*60/ndays*100:>8.1f}%")
print(f"\n  6 板块合计 {len(D0)} 次信号 / {years:.2f} 年 = 每板块每年 {len(D0)/6/years:.2f} 次")
print(f"  若每次持有 60 交易日，单个板块一年被占用约 {len(D0)/6/years*60:.0f} / 244 天 "
      f"（{len(D0)/6/years*60/244*100:.0f}%）")
print("\n  信号日分布：")
for y, g in D0.groupby(D0["date"].dt.year):
    print(f"    {y}: {len(g)} 次  " + ", ".join(f"{r.board}({r.date:%m-%d})" for r in g.itertuples()))

b20, b60 = baseline(PP, 20), baseline(PP, 60)
print(f"\n  随机基线：持有20天盈利 {b20[0]:.1f}%（含回撤约束 {b20[1]:.1f}%）　持有60天盈利 {b60[0]:.1f}%")

print("\n" + "=" * 118)
print("Q4 · 阈值扫描：入场必须 ≤0 吗？（生产 6 板块 + 长历史 8 宽基双重验证）")
print("=" * 118)
PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
LP = {nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "trail")), "close": raw["close"]})
      for nm, raw in PAN.items()}
EP = {nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "expand")), "close": raw["close"]})
      for nm, raw in PAN.items()}

print(f"\n【生产 6 板块 · {years:.2f} 年】基线：T+20 {b20[1]:.1f}%　T+60 {b60[0]:.1f}%")
print(f"  {'阈值':>6}{'信号数':>7}{'T+20命中':>10}{'下界':>7}{'T+60盈利':>10}{'均T+20':>9}{'最差浮亏':>10}")
for t in [-15, -10, -5, 0, 5, 10, 15, 20, 25]:
    d = signals(PP, float(t))
    if len(d) == 0:
        print(f"  {t:>6}{0:>7}"); continue
    ok = int(d["ok20"].sum()); n = len(d)
    print(f"  {t:>6}{n:>7}{ok/n*100:>9.1f}%{wil(ok,n)*100:>7.1f}"
          f"{int((d['ret60']>0).sum())/n*100:>9.1f}%{d['ret20'].mean()*100:>9.2f}{d['mdd20'].min()*100:>10.2f}")

print("\n【长历史 8 宽基 2013~2026 · trail 锚】—— 铁律：过不了 2015-2016 就不采")
print(f"  {'阈值':>6}{'信号数':>7}{'T+20命中':>10}{'2015':>8}{'2016':>8}{'2020':>8}{'T+60盈利':>10}")
for t in [-15, -10, -5, 0, 5, 10, 15, 20, 25]:
    d = signals(LP, float(t))
    if len(d) == 0:
        print(f"  {t:>6}{0:>7}"); continue
    n = len(d)
    def yr(y):
        g = d[d["date"].dt.year == y]
        return f"{int(g['ok20'].sum())}/{len(g)}" if len(g) else "-"
    print(f"  {t:>6}{n:>7}{int(d['ok20'].sum())/n*100:>9.1f}%{yr(2015):>8}{yr(2016):>8}{yr(2020):>8}"
          f"{int((d['ret60']>0).sum())/n*100:>9.1f}%")

print("\n【长历史 8 宽基 · 扩展锚】")
print(f"  {'阈值':>6}{'信号数':>7}{'T+20命中':>10}{'2015':>8}{'2016':>8}{'2020':>8}{'T+60盈利':>10}")
for t in [-15, -10, -5, 0, 5, 10, 15, 20, 25]:
    d = signals(EP, float(t))
    if len(d) == 0:
        print(f"  {t:>6}{0:>7}"); continue
    n = len(d)
    def yr(y):
        g = d[d["date"].dt.year == y]
        return f"{int(g['ok20'].sum())}/{len(g)}" if len(g) else "-"
    print(f"  {t:>6}{n:>7}{int(d['ok20'].sum())/n*100:>9.1f}%{yr(2015):>8}{yr(2016):>8}{yr(2020):>8}"
          f"{int((d['ret60']>0).sum())/n*100:>9.1f}%")

print("\n【信号当日的分值分布】（生产，阈值 0）")
d0 = signals(PP, 0.0)
print(f"  当日分值：min {d0['score'].min():.1f}　中位 {d0['score'].median():.1f}　max {d0['score'].max():.1f}")
print(f"  昨日分值：min {d0['prev'].min():.1f}　中位 {d0['prev'].median():.1f}　max {d0['prev'].max():.1f}")
print(f"  → 昨日 ≤-5 的占 {int((d0['prev']<=-5).sum())}/{len(d0)}，≤-10 的占 {int((d0['prev']<=-10).sum())}/{len(d0)}")
