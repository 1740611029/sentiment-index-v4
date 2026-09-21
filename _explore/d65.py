"""横截面共振：同一天里，是不是「所有板块一起跌到 0 以下」才算真底部？

动机（来自唯一失败的两次事件）：
  2024-01-23 大盘板块(SH)的信号赚 +11.46%，同一天中证2000 亏 −1.29%、中证1000 最深浮亏 −15.48%。
  同一天、同一个模型、同一个阈值，结果相反 —— 说明判别信息可能不在单板块的**时间序列**里，
  而在**横截面**上：那天只有中小盘在恐慌，大盘没跟上，那是「局部流动性危机」，不是「全市场投降」。

之前被否决的 4 条过滤全部是「单板块内部时序特征」（放量 / 不创新低 / cf_b>0 / 相对放量）。
这次是完全不同的维度，且不需要任何新数据源。

待检特征（在信号日当天，用全体板块的 score 快照算）：
  reso  当日 score ≤ 0 的板块数（含自己）—— 共振广度
  frac  同上 / 参与计算的板块总数
  gap   自己的 score − 当日全体板块 score 的中位数（负 = 比别人更恐慌）
  rank  自己的 score 在当日全体中的升序排名（0 = 全场最惨）
  lowN  过去 20 个交易日内，有多少个**不同**板块曾触及 ≤ 0

纪律：生产好看不算，必须过 2015-2016。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C
from senti import model as M
from senti import factors as F
import longhist

H, COOL, THR = 20, 20, 0.0
H2 = 60
W2 = pd.Timestamp("2023-09-20")
LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
LO_Q, HI_Q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
EXCLUDE = {"CSI2000"}
KEYS = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20", "disp", "dd250"]
WIN, MINP = C.MODEL["anchor_window"], C.MODEL["anchor_min"]
CLIP_LO, CLIP_HI = -30.0, 140.0
TOL = 0.03


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


def signals(panels):
    """panels: {name: DataFrame(score, close)} → 逐板块信号 + 横截面特征"""
    S = pd.DataFrame({nm: p["score"] for nm, p in panels.items()})
    S = S.dropna(how="all")
    CL = pd.DataFrame({nm: p["close"] for nm, p in panels.items()}).reindex(S.index)
    rows = []
    for nm in panels:
        v = S[nm].to_numpy(float); cl = CL[nm].to_numpy(float)
        n = len(v); last = -10 ** 9
        for i in range(1, n - H2):
            if np.isnan(v[i]) or np.isnan(v[i - 1]): continue
            if i - last < COOL: continue
            if v[i - 1] > THR or v[i] <= v[i - 1]: continue
            last = i
            c0 = cl[i]
            seg = cl[i + 1:i + H + 1]; seg2 = cl[i + 1:i + H2 + 1]
            snap = S.iloc[i]
            others = snap.drop(nm).dropna()
            reso = int((snap <= THR).sum())
            rows.append({
                "board": nm, "date": S.index[i],
                "ret20": seg[-1] / c0 - 1, "mdd20": seg.min() / c0 - 1,
                "ret60": seg2[-1] / c0 - 1, "mdd60": seg2.min() / c0 - 1,
                "ok20": bool(seg[-1] / c0 - 1 > 0 and seg.min() / c0 - 1 >= -TOL),
                "reso": reso, "frac": reso / max(1, int(snap.notna().sum())),
                "gap": float(v[i] - others.median()) if len(others) else np.nan,
                "rank": int((snap < v[i]).sum()),
                "ntot": int(snap.notna().sum()),
                "lowN": int((S.iloc[max(0, i - 19):i + 1] <= THR).any().sum()),
            })
    return pd.DataFrame(rows)


def line(df, label):
    if len(df) == 0:
        print(f"  {label:<28} 无样本"); return
    ok = int(df["ok20"].sum()); n = len(df)
    p = ok / n; z = 1.96
    wl = (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    print(f"  {label:<28} n={n:>3}  命中 {ok}/{n:<3} {p*100:5.1f}%  下界 {wl*100:5.1f}%   "
          f"均T+20 {df['ret20'].mean()*100:+6.2f}%   最差浮亏 {df['mdd20'].min()*100:+7.2f}%   "
          f"T+60盈利 {(df['ret60']>0).mean()*100:5.1f}%")


# ==================== ① 生产 6 板块 ====================
print("=" * 124)
print("① 生产 6 板块（因果锚，阈值 0，H=20 主口径）")
print("=" * 124)
RAW = {b: F.build_board_raw(b) for b in C.BOARD_ORDER}
prod = {}
for b in C.BOARD_ORDER:
    raw = RAW[b]
    sc = score(raw, make_anchors(raw, "trail"))
    prod[b] = pd.DataFrame({"score": sc, "close": raw["close"]})
DP = signals(prod)
line(DP, "全部信号")
print("\n  【共振广度 reso 分组】")
for r in sorted(DP["reso"].unique()):
    line(DP[DP["reso"] == r], f"reso = {r}")
print("\n  【共振门槛：reso >= X】")
for x in range(1, 7):
    line(DP[DP["reso"] >= x], f"reso >= {x}")
print("\n  【gap 分组（自己 − 当日其他板块中位数）】")
for lo, hi in [(-99, -12), (-12, -6), (-6, -2), (-2, 0), (0, 99)]:
    line(DP[(DP["gap"] >= lo) & (DP["gap"] < hi)], f"gap ∈ [{lo}, {hi})")
print("\n  【rank 分组（0 = 全场最惨）】")
for r in sorted(DP["rank"].unique()):
    line(DP[DP["rank"] == r], f"rank = {r}")

print("\n  【两次失败事件在哪一组】")
print(DP[~DP["ok20"]][["board", "date", "ret20", "mdd20", "reso", "frac", "gap", "rank", "lowN"]].to_string(index=False))

# ==================== ② 长历史 ====================
PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
for mode, tag in (("trail", "trailing-750"), ("expand", "扩展因果锚（含 2015-2016）")):
    L = {nm: pd.DataFrame({"score": score(raw, make_anchors(raw, mode)), "close": raw["close"]})
         for nm, raw in PAN.items()}
    DL = signals(L)
    print("\n" + "=" * 124)
    print(f"② 长历史 {len(PAN)} 宽基 · {tag}")
    print("=" * 124)
    line(DL, "全部信号")
    line(DL[DL["date"] >= W2], "　└ 仅交付窗口")
    print("\n  【共振门槛 reso >= X】")
    for x in range(1, len(PAN) + 1):
        line(DL[DL["reso"] >= x], f"reso >= {x}")
    print("\n  【gap 分组】")
    for lo, hi in [(-99, -12), (-12, -6), (-6, -2), (-2, 0), (0, 99)]:
        line(DL[(DL["gap"] >= lo) & (DL["gap"] < hi)], f"gap ∈ [{lo}, {hi})")
    print("\n  【分年度 reso>=1 / 全部】")
    yrs = sorted(DL["date"].dt.year.unique())
    print(f"    {'':<12}" + "".join(f"{y:>10}" for y in yrs))
    for lab, sub in (("全部", DL), ("reso>=2", DL[DL["reso"] >= 2]), ("reso>=3", DL[DL["reso"] >= 3])):
        cells = ""
        for y in yrs:
            s = sub[sub["date"].dt.year == y]
            cells += (f"{int(s['ok20'].sum())}/{len(s)}" if len(s) else "-").rjust(10)
        print(f"    {lab:<12}" + cells)
