"""顶部定稿：杠杆挤满确认层 —— 配随机基线 + 邻域扰动 + 换尺度。

d72 的核心发现（与底部严格对称）：
  底部：近 3 年破净分位全部 ≥60（都是真投降） → 底部信号有效 89.5%
  顶部：近 3 年融资余额分位全部 < 70（都不是真狂热） → 顶部信号无效 22%
  → **近 3 年根本没出现过真正的「杠杆顶」。**

  加了「融资余额分位 ≥70」之后：
    长历史 trail  T+60 下跌 8/8
    长历史 expand T+60 下跌 7/7
  但 T+20 命中率仍只有 37%~43% —— 说明顶部信号的**自然尺度是季度，不是月度**。

d73 要回答：
  ① T+60 下跌 8/8 相对随机基线是不是显著？（A股持有60天下跌基线约 47%）
  ② 门槛 60/70/80/90 是不是平台？
  ③ 顶部该不该把主口径从 T+20 换成 T+60？
  ④ 生产窗口为什么无样本（要能自圆其说）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from math import comb
from senti import config as C
from senti import model as M
from senti import factors as F
import longhist

H, H2, COOL, TOL = 20, 60, 20, 0.03
TOP = 100.0
W2 = pd.Timestamp("2023-09-20")
LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
LO_Q, HI_Q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
EXCLUDE = {"CSI2000"}
KEYS = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20", "disp", "dd250"]
WIN, MINP = C.MODEL["anchor_window"], C.MODEL["anchor_min"]
CLIP_LO, CLIP_HI = -30.0, 140.0
MKT_DIR = os.path.join(C.DATA_DIR, "cache", "market")
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


def top_signals(panels):
    S = pd.DataFrame({nm: p["score"] for nm, p in panels.items()}).dropna(how="all")
    CL = pd.DataFrame({nm: p["close"] for nm, p in panels.items()}).reindex(S.index)
    ZH = (S > TOP)
    rows = []
    for nm in panels:
        v = S[nm].to_numpy(float); cl = CL[nm].to_numpy(float)
        n = len(v); last = -10 ** 9
        for i in range(1, n - H2):
            if np.isnan(v[i]) or np.isnan(v[i - 1]): continue
            if i - last < COOL: continue
            if v[i - 1] > TOP or v[i] <= TOP: continue
            last = i
            c0 = cl[i]
            seg = cl[i + 1:i + H + 1]; seg2 = cl[i + 1:i + H2 + 1]
            rows.append({"board": nm, "date": S.index[i], "score": v[i],
                         "ret20": seg[-1] / c0 - 1, "run20": seg.max() / c0 - 1,
                         "ret60": seg2[-1] / c0 - 1, "run60": seg2.max() / c0 - 1,
                         "ok20": bool(seg[-1] / c0 - 1 < 0 and seg.max() / c0 - 1 <= TOL),
                         "ok60": bool(seg2[-1] / c0 - 1 < 0 and seg2.max() / c0 - 1 <= TOL),
                         "down60": bool(seg2[-1] / c0 - 1 < 0),
                         "top_reso": int(ZH.iloc[i].sum())})
    return pd.DataFrame(rows)


mg = pd.read_parquet(os.path.join(MKT_DIR, "margin.parquet")).set_index("date")["margin"].astype(float)
mg = mg[~mg.index.duplicated(keep="last")].sort_index().shift(1)
MG_PCT = (mg.rolling(250, min_periods=120).rank(pct=True) * 100)


def attach(df):
    out = df.copy()
    out["mg_pct"] = MG_PCT.reindex(pd.DatetimeIndex(out["date"]), method="ffill").to_numpy()
    return out


RAW = {b: F.build_board_raw(b) for b in C.BOARD_ORDER}
PP = {b: pd.DataFrame({"score": score(RAW[b], make_anchors(RAW[b], "trail")),
                       "close": RAW[b]["close"]}) for b in C.BOARD_ORDER}
DP = attach(top_signals(PP))
PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
LP = {nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "trail")), "close": raw["close"]})
      for nm, raw in PAN.items()}
EP = {nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "expand")), "close": raw["close"]})
      for nm, raw in PAN.items()}
LT = attach(top_signals(LP)); LE = attach(top_signals(EP))

print("=" * 132)
print("① 邻域扰动：顶部「杠杆挤满」门槛 50 → 95")
print("=" * 132)
print(f"  {'门槛':<10}{'生产 n':>8}{'trail T+60跌':>14}{'expand T+60跌':>16}{'expand T+20命中':>16}{'最大踏空':>12}")
for x in (50, 60, 65, 70, 75, 80, 90, 95):
    d, t, e = DP[DP["mg_pct"] >= x], LT[LT["mg_pct"] >= x], LE[LE["mg_pct"] >= x]
    f = lambda s, c: (f"{int(s[c].sum())}/{len(s)}" if len(s) else "-")
    mx = max([s["run20"].max() for s in (d, t, e) if len(s)] or [0])
    print(f"  >= {x:<7}{len(d):>8}{f(t,'down60'):>14}{f(e,'down60'):>16}{f(e,'ok20'):>16}{mx*100:>+11.2f}%")

print("\n" + "=" * 132)
print("② 随机基线对照（T+60 下跌）—— 不放对照就不能信")
print("=" * 132)


def baseline_down(panels, sub, name):
    """同板块、同信号数分布，随机取日的 T+60 下跌比例。"""
    cnt = sub["board"].value_counts()
    pool = {}
    for nm, p in panels.items():
        cl = p["close"].to_numpy(float); n = len(cl); hi = n - H2 - 1
        c0 = cl[:hi]
        pool[nm] = cl[H2:hi + H2] / c0 - 1
    obs = float((sub["ret60"] < 0).mean())
    sims = np.empty(4000)
    for s in range(4000):
        tot = 0.0; c = 0
        for nm, k in cnt.items():
            arr = pool[nm]
            tot += (arr[rng.integers(0, len(arr), k)] < 0).sum(); c += k
        sims[s] = tot / c
    base = sims.mean(); n = len(sub); kk = int(round(obs * n))
    pv = sum(comb(n, j) * base ** j * (1 - base) ** (n - j) for j in range(kk, n + 1))
    print(f"  【{name}】 n={n}   T+60 下跌 {kk}/{n} = {obs*100:5.1f}%   "
          f"随机基线 {base*100:5.1f}% ± {sims.std()*100:.1f}%   P(≥观测|基线) = {pv:.5f}   "
          f"提升 {(obs-base)*100:+.1f}pp")


for thr in (70, 80):
    print(f"\n  ── 融资余额分位 >= {thr} ──")
    baseline_down(LP, LT[LT["mg_pct"] >= thr], f"长历史 trail · mg_pct>={thr}")
    baseline_down(EP, LE[LE["mg_pct"] >= thr], f"长历史 expand · mg_pct>={thr}")

print("\n  对照：不加任何过滤的顶部信号")
baseline_down(LP, LT, "长历史 trail · 无过滤")
baseline_down(EP, LE, "长历史 expand · 无过滤")

print("\n" + "=" * 132)
print("③ 顶部该不该换尺度？T+20 vs T+60（长历史 expand，杠杆分位>=70）")
print("=" * 132)
sub = LE[LE["mg_pct"] >= 70]
for hh, lab in ((20, "T+20"), (60, "T+60")):
    dn = int((sub[f"ret{hh}"] < 0).sum())
    ok = int(sub[f"ok{hh}"].sum())
    up = sub[f"run{hh}"].max() * 100
    print(f"  {lab}   下跌 {dn}/{len(sub)} = {dn/len(sub)*100:5.1f}%   "
          f"严格命中(下跌且不踏空>3%) {ok}/{len(sub)}   最大踏空 {up:+6.2f}%   "
          f"均收益 {sub[f'ret{hh}'].mean()*100:+6.2f}%")
print("\n  对照：无过滤")
for hh, lab in ((20, "T+20"), (60, "T+60")):
    dn = int((LE[f"ret{hh}"] < 0).sum()); ok = int(LE[f"ok{hh}"].sum())
    print(f"  {lab}   下跌 {dn}/{len(LE)} = {dn/len(LE)*100:5.1f}%   "
          f"严格命中 {ok}/{len(LE)}   最大踏空 {LE[f'run{hh}'].max()*100:+6.2f}%   "
          f"均收益 {LE[f'ret{hh}'].mean()*100:+6.2f}%")

print("\n" + "=" * 132)
print("④ 生产窗口为什么无样本 —— 近 3 年每次破 100 时的融资余额分位")
print("=" * 132)
t = DP.copy(); t["date"] = t["date"].dt.strftime("%Y-%m-%d")
print(t[["board", "date", "score", "ret20", "ret60", "mg_pct", "top_reso"]].sort_values("date").to_string(
    index=False, formatters={"score": lambda x: f"{x:6.1f}", "ret20": lambda x: f"{x*100:+6.2f}%",
                             "ret60": lambda x: f"{x*100:+7.2f}%", "mg_pct": lambda x: f"{x:5.1f}"}))
print(f"\n  生产窗口顶部信号的 mg_pct：最小 {DP['mg_pct'].min():.1f}  最大 {DP['mg_pct'].max():.1f}   "
      f"中位 {DP['mg_pct'].median():.1f}")
print(f"  长历史 trail  顶部信号的 mg_pct：最小 {LT['mg_pct'].min():.1f}  最大 {LT['mg_pct'].max():.1f}   "
      f"中位 {LT['mg_pct'].median():.1f}")
print("\n  全市场融资余额分位的分布（近 3 年 vs 全历史）：")
rec = MG_PCT.dropna()
print(f"    近 3 年（>= {W2.date()}）：中位 {rec[rec.index >= W2].median():.1f}   "
      f"最高 {rec[rec.index >= W2].max():.1f}   处于 >=70 的天数占比 {(rec[rec.index >= W2] >= 70).mean()*100:.1f}%")
print(f"    全历史           ：中位 {rec.median():.1f}   最高 {rec.max():.1f}   "
      f"处于 >=70 的天数占比 {(rec >= 70).mean()*100:.1f}%")
