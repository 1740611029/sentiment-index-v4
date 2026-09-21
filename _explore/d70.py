"""新信息源的定稿检验：用可解释的绝对阈值，做网格组合 + 分年度 + 邻域扰动 + 与共振叠加。

d69 已经证明三件事：
  ① 2015 年 12 个信号按「融资余额 20 日降幅」二分：剧烈的一半 T+60 6/6，缓慢的一半 0/6。
     语义：**杠杆还没出清，底就不是底。**
  ② 破净率（bna）与其一年分位，在长历史扩展锚上的分离度分别是 1.63 和 3.07（最强）。
  ③ 但这些是**市场级**数据 —— 同一天的 SH（赚）和 CSI2000（亏）读数完全一样，
     所以它做的是「全局择时」，不能区分板块。与 reso（横截面）互补，应当叠加。

d70 要回答：
  ① 阈值能不能用「人能说清楚理由的整数」，而不是从分位数里挑出来的？
  ② 组合之后是不是还成立（生产 + 长历史 + 2015-2016）？
  ③ 邻域扰动有没有悬崖（防止挑中尖峰）？
  ④ 与 reso 叠加后是多少？
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C
from senti import model as M
from senti import factors as F
import longhist

H, H2, COOL, THR, TOL = 20, 60, 20, 0.0, 0.03
W2 = pd.Timestamp("2023-09-20")
LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
LO_Q, HI_Q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
EXCLUDE = {"CSI2000"}
KEYS = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20", "disp", "dd250"]
WIN, MINP = C.MODEL["anchor_window"], C.MODEL["anchor_min"]
CLIP_LO, CLIP_HI = -30.0, 140.0
MKT_DIR = os.path.join(C.DATA_DIR, "cache", "market")


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
    S = pd.DataFrame({nm: p["score"] for nm, p in panels.items()}).dropna(how="all")
    CL = pd.DataFrame({nm: p["close"] for nm, p in panels.items()}).reindex(S.index)
    Z = (S <= THR)
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
            rows.append({"board": nm, "date": S.index[i],
                         "ret20": seg[-1] / c0 - 1, "mdd20": seg.min() / c0 - 1,
                         "ret60": seg2[-1] / c0 - 1,
                         "ok20": bool(seg[-1] / c0 - 1 > 0 and seg.min() / c0 - 1 >= -TOL),
                         "reso": int(Z.iloc[i].sum())})
    return pd.DataFrame(rows)


mg = pd.read_parquet(os.path.join(MKT_DIR, "margin.parquet")).set_index("date")["margin"].astype(float)
bn = pd.read_parquet(os.path.join(MKT_DIR, "bna.parquet")).set_index("date")["bna"].astype(float)
mg = mg[~mg.index.duplicated(keep="last")].sort_index()
bn = bn[~bn.index.duplicated(keep="last")].sort_index()
D = pd.DataFrame({"mg": mg, "bna": bn}).sort_index()
D["mg"] = D["mg"].shift(1); D["bna"] = D["bna"].shift(1)
r250 = lambda x: x.rolling(250, min_periods=120)
D["mg_pct"] = r250(D["mg"]).rank(pct=True) * 100
D["mg_chg20"] = D["mg"] / D["mg"].shift(20) - 1
D["bna_pct"] = r250(D["bna"]).rank(pct=True) * 100


def attach(df):
    out = df.copy()
    for f in ("mg_pct", "mg_chg20", "bna", "bna_pct"):
        out[f] = D[f].reindex(pd.DatetimeIndex(out["date"]), method="ffill").to_numpy()
    return out


def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def line(df, label):
    if len(df) == 0:
        print(f"  {label:<34} 无样本"); return
    ok = int(df["ok20"].sum()); n = len(df)
    w60 = int((df["ret60"] > 0).sum())
    print(f"  {label:<34} n={n:>3}  命中 {ok}/{n:<3} {ok/n*100:5.1f}%  下界 {wil(ok,n)*100:5.1f}%   "
          f"均T+20 {df['ret20'].mean()*100:+6.2f}%  最差浮亏 {df['mdd20'].min()*100:+7.2f}%   T+60 {w60}/{n}")


RAW = {b: F.build_board_raw(b) for b in C.BOARD_ORDER}
DP = attach(signals({b: pd.DataFrame({"score": score(RAW[b], make_anchors(RAW[b], "trail")),
                                      "close": RAW[b]["close"]}) for b in C.BOARD_ORDER}))
PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
LT = attach(signals({nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "trail")),
                                       "close": raw["close"]}) for nm, raw in PAN.items()}))
LE = attach(signals({nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "expand")),
                                       "close": raw["close"]}) for nm, raw in PAN.items()}))

# ---------- 可解释的整数阈值 ----------
# mg_chg20 <= -3%    ：20 个交易日内融资余额掉了 3% 以上 —— 一轮像样的去杠杆已经发生
# mg_pct   <= 30     ：融资余额处于近一年的后 30% —— 杠杆水位已经退下来
# bna      >= 8%     ：每 12 只股票就有 1 只破净 —— 估值侧的投降
# bna_pct  >= 60     ：破净率处于近一年偏高的位置（跨时代可比）
COND = {
    "去杠杆已发生 mg_chg20<=-3%": lambda d: d["mg_chg20"] <= -0.03,
    "去杠杆已发生 mg_chg20<=-5%": lambda d: d["mg_chg20"] <= -0.05,
    "杠杆水位低 mg_pct<=30": lambda d: d["mg_pct"] <= 30.0,
    "杠杆水位低 mg_pct<=40": lambda d: d["mg_pct"] <= 40.0,
    "破净率 bna>=8%": lambda d: d["bna"] >= 0.08,
    "破净率 bna>=10%": lambda d: d["bna"] >= 0.10,
    "破净分位 bna_pct>=60": lambda d: d["bna_pct"] >= 60.0,
    "破净分位 bna_pct>=75": lambda d: d["bna_pct"] >= 75.0,
    "共振 reso>=2": lambda d: d["reso"] >= 2,
}

print("=" * 138)
print("① 单条件：生产 / 长历史trail / 长历史expand（含 2015-2016）")
print("=" * 138)
for lab, fn in COND.items():
    print(f"\n──── {lab} ────")
    for nm, d in (("生产 33", DP), ("长历史trail 77", LT), ("长历史expand 35", LE)):
        line(d[fn(d)], f"　{nm}")
print("\n　（对照）无过滤")
for nm, d in (("生产 33", DP), ("长历史trail 77", LT), ("长历史expand 35", LE)):
    line(d, f"　{nm}")

print("\n" + "=" * 138)
print("② 组合条件")
print("=" * 138)
COMB = {
    "去杠杆(-3%) + 破净(>=8%)": lambda d: COND["去杠杆已发生 mg_chg20<=-3%"](d) & COND["破净率 bna>=8%"](d),
    "去杠杆(-3%) + 杠杆水位(<=30)": lambda d: COND["去杠杆已发生 mg_chg20<=-3%"](d) & COND["杠杆水位低 mg_pct<=30"](d),
    "去杠杆(-3%) 或 破净(>=8%)": lambda d: COND["去杠杆已发生 mg_chg20<=-3%"](d) | COND["破净率 bna>=8%"](d),
    "去杠杆(-3%) 或 杠杆水位(<=30)": lambda d: COND["去杠杆已发生 mg_chg20<=-3%"](d) | COND["杠杆水位低 mg_pct<=30"](d),
    "破净分位(>=60) + 共振(>=2)": lambda d: COND["破净分位 bna_pct>=60"](d) & COND["共振 reso>=2"](d),
    "去杠杆(-3%) + 共振(>=2)": lambda d: COND["去杠杆已发生 mg_chg20<=-3%"](d) & COND["共振 reso>=2"](d),
    "去杠杆(-3%) 或 共振(>=2)": lambda d: COND["去杠杆已发生 mg_chg20<=-3%"](d) | COND["共振 reso>=2"](d),
    "三项全满足": lambda d: COND["去杠杆已发生 mg_chg20<=-3%"](d) & COND["破净率 bna>=8%"](d) & COND["共振 reso>=2"](d),
    "去杠杆(-3%) + 破净(>=8%) + 共振(>=2) 中至少2项": None,
}
for lab, fn in COMB.items():
    if fn is None:
        n3 = (COND["去杠杆已发生 mg_chg20<=-3%"], COND["破净率 bna>=8%"], COND["共振 reso>=2"])
        fn = lambda d, n3=n3: (n3[0](d).astype(int) + n3[1](d).astype(int) + n3[2](d).astype(int)) >= 2
    print(f"\n──── {lab} ────")
    for nm, d in (("生产 33", DP), ("长历史trail 77", LT), ("长历史expand 35", LE)):
        line(d[fn(d)], f"　{nm}")

print("\n" + "=" * 138)
print("③ 分年度（长历史 expand 锚，含 2015-2016）—— 准入门槛")
print("=" * 138)
CAND = ["去杠杆已发生 mg_chg20<=-3%", "杠杆水位低 mg_pct<=30", "破净率 bna>=8%",
        "破净分位 bna_pct>=60", "共振 reso>=2"]
years = sorted(LE["date"].dt.year.unique())
print(f"  {'条件':<32}" + "".join(f"{y:>9}" for y in years) + f"{'合计':>12}")
for lab in CAND + ["无过滤"]:
    fn = COND.get(lab)
    sub = LE[fn(LE)] if fn else LE
    cells = ""
    for y in years:
        s = sub[sub["date"].dt.year == y]
        cells += (f"{int(s['ok20'].sum())}/{len(s)}" if len(s) else "-").rjust(9)
    ok = int(sub["ok20"].sum())
    cells += f"{ok}/{len(sub)}".rjust(12)
    print(f"  {lab:<32}" + cells)
print("  （T+60 盈利版）")
for lab in CAND + ["无过滤"]:
    fn = COND.get(lab)
    sub = LE[fn(LE)] if fn else LE
    cells = ""
    for y in years:
        s = sub[sub["date"].dt.year == y]
        cells += (f"{int((s['ret60']>0).sum())}/{len(s)}" if len(s) else "-").rjust(9)
    w = int((sub["ret60"] > 0).sum())
    cells += f"{w}/{len(sub)}".rjust(12)
    print(f"  {lab:<32}" + cells)

print("\n" + "=" * 138)
print("④ 邻域扰动（防尖峰）—— 主候选：mg_chg20<=-3% 与 bna>=8%")
print("=" * 138)
for x in (-0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.08):
    d = DP[DP["mg_chg20"] <= x]; e = LE[LE["mg_chg20"] <= x]
    print(f"  mg_chg20 <= {x*100:+.0f}%   生产 {int(d['ok20'].sum())}/{len(d)}"
          f"   长历史expand {int(e['ok20'].sum())}/{len(e)}   T+60 {int((e['ret60']>0).sum())}/{len(e)}"
          f"   最差浮亏 {min(d['mdd20'].min() if len(d) else 0, e['mdd20'].min() if len(e) else 0)*100:+.2f}%")
print()
for x in (0.04, 0.06, 0.08, 0.10, 0.12, 0.14):
    d = DP[DP["bna"] >= x]; e = LE[LE["bna"] >= x]
    print(f"  bna >= {x*100:.0f}%       生产 {int(d['ok20'].sum())}/{len(d)}"
          f"   长历史expand {int(e['ok20'].sum())}/{len(e)}   T+60 {int((e['ret60']>0).sum())}/{len(e)}"
          f"   最差浮亏 {min(d['mdd20'].min() if len(d) else 0, e['mdd20'].min() if len(e) else 0)*100:+.2f}%")
