"""bna_pct>=60 的定稿验证：邻域扰动 + 与共振叠加 + 随机基线对照。

d70 的关键结论：在长历史扩展锚（含 2015-2016）上，
  bna_pct >= 60 是**唯一**把 2015 年从 6/12 救到 6/6 的条件 ——
  而且它保留了 6 个样本（不是靠「那年干脆不出信号」混过去）。
  其余条件（mg_chg20<=-3%、mg_pct<=30、bna>=8%）在 2015 年要么无样本、要么回避。

bna_pct = 破净股占比在过去 250 个交易日中的分位（0~100，越大 = 破净越严重）。
为什么必须用**分位**而不是绝对值：破净率的绝对水平有强烈的时代偏移
（2015 年最高只有 1.76%，2024 年能到 16.64%）。用分位才能跨时代可比。

d71 回答：
  ① 60 是不是尖峰？（邻域 40~85 全扫）
  ② 与 reso>=2 叠加会怎样？
  ③ 相对随机基线是不是真的显著？（不能只看 100%）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from math import comb
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
rng = np.random.default_rng(20260920)


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


bn = pd.read_parquet(os.path.join(MKT_DIR, "bna.parquet")).set_index("date")["bna"].astype(float)
bn = bn[~bn.index.duplicated(keep="last")].sort_index().shift(1)   # 次日才可得
BNA_PCT = (bn.rolling(250, min_periods=120).rank(pct=True) * 100)


def attach(df):
    out = df.copy()
    out["bna_pct"] = BNA_PCT.reindex(pd.DatetimeIndex(out["date"]), method="ffill").to_numpy()
    return out


def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def line(df, label):
    if len(df) == 0:
        print(f"  {label:<32} 无样本"); return
    ok = int(df["ok20"].sum()); n = len(df)
    print(f"  {label:<32} n={n:>3}  命中 {ok}/{n:<3} {ok/n*100:5.1f}%  下界 {wil(ok,n)*100:5.1f}%   "
          f"均T+20 {df['ret20'].mean()*100:+6.2f}%  最差浮亏 {df['mdd20'].min()*100:+7.2f}%   "
          f"T+60 {int((df['ret60']>0).sum())}/{n}")


RAW = {b: F.build_board_raw(b) for b in C.BOARD_ORDER}
PP = {b: pd.DataFrame({"score": score(RAW[b], make_anchors(RAW[b], "trail")),
                       "close": RAW[b]["close"]}) for b in C.BOARD_ORDER}
DP = attach(signals(PP))
PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
LP = {nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "trail")), "close": raw["close"]})
      for nm, raw in PAN.items()}
EP = {nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "expand")), "close": raw["close"]})
      for nm, raw in PAN.items()}
LT = attach(signals(LP)); LE = attach(signals(EP))

print("=" * 136)
print("① 邻域扰动：bna_pct 门槛 40 → 85")
print("=" * 136)
print(f"  {'门槛':<10}{'生产 命中':>12}{'生产 T+60':>12}{'长历史trail 命中':>18}{'长历史trail T+60':>18}"
      f"{'长历史expand 命中':>18}{'expand T+60':>14}{'expand 2015':>14}")
for x in (40, 45, 50, 55, 60, 65, 70, 75, 80, 85):
    d, t, e = DP[DP["bna_pct"] >= x], LT[LT["bna_pct"] >= x], LE[LE["bna_pct"] >= x]
    e15 = e[e["date"].dt.year == 2015]
    f = lambda s, c: (f"{int(s[c].sum())}/{len(s)}" if len(s) else "-")
    print(f"  >= {x:<7}{f(d,'ok20'):>12}{f(d,'ret60'):>12}{f(t,'ok20'):>18}{f(t,'ret60'):>18}"
          f"{f(e,'ok20'):>18}{f(e,'ret60'):>14}{f(e15,'ret60'):>14}")

print("\n" + "=" * 136)
print("② bna_pct>=60 与 reso>=2 的组合")
print("=" * 136)
for lab, fn in (("无过滤", lambda d: d["bna_pct"] >= -1),
                ("bna_pct>=60", lambda d: d["bna_pct"] >= 60),
                ("reso>=2", lambda d: d["reso"] >= 2),
                ("bna_pct>=60 且 reso>=2", lambda d: (d["bna_pct"] >= 60) & (d["reso"] >= 2)),
                ("bna_pct>=60 或 reso>=2", lambda d: (d["bna_pct"] >= 60) | (d["reso"] >= 2))):
    print(f"\n──── {lab} ────")
    for nm, d in (("生产 33", DP), ("长历史trail 77", LT), ("长历史expand 35", LE)):
        line(d[fn(d)], f"　{nm}")

print("\n" + "=" * 136)
print("③ 分年度（T+60 盈利 / H20 命中）—— 长历史 expand 锚，2015-2016 是门槛")
print("=" * 136)
years = sorted(LE["date"].dt.year.unique())
for lab, fn in (("无过滤", lambda d: d["bna_pct"] >= -1),
                ("bna_pct>=60", lambda d: d["bna_pct"] >= 60),
                ("bna_pct>=60 或 reso>=2", lambda d: (d["bna_pct"] >= 60) | (d["reso"] >= 2))):
    sub = LE[fn(LE)]
    for metric, col in (("T+60 盈利", "ret60"), ("H20 命中", "ok20")):
        cells = ""
        for y in years:
            s = sub[sub["date"].dt.year == y]
            k = int((s[col] > 0).sum()) if col == "ret60" else int(s[col].sum())
            cells += (f"{k}/{len(s)}" if len(s) else "-").rjust(9)
        tot = int((sub[col] > 0).sum()) if col == "ret60" else int(sub[col].sum())
        cells += f"{tot}/{len(sub)}".rjust(12)
        print(f"  {lab:<24}{metric:<12}" + cells)

print("\n" + "=" * 136)
print("④ 随机基线对照（不放对照就不能信）—— bna_pct>=60")
print("=" * 136)


def baselines(panels, sub, name):
    cnt = sub["board"].value_counts()
    pool = {}
    for nm, p in panels.items():
        cl = p["close"].to_numpy(float); n = len(cl); hi = n - H2 - 1
        c0 = cl[:hi]
        pool[nm] = {"r60": cl[H2:hi + H2] / c0 - 1, "r20": cl[H:hi + H] / c0 - 1,
                    "m20": np.array([cl[j + 1:j + H + 1].min() / cl[j] - 1 for j in range(hi)])}
    print(f"\n  【{name}】 bna_pct>=60  n={len(sub)}")
    for metric, key, scol, mode in (("T+60 盈利", "r60", "ret60", "pos"),
                                    ("T+20 盈利", "r20", "ret20", "pos"),
                                    ("H20 未被套>3%", "m20", "mdd20", "tol")):
        hit = (lambda a: a > 0) if mode == "pos" else (lambda a: a >= -TOL)
        obs = float(hit(sub[scol].to_numpy()).mean())
        sims = np.empty(4000)
        for s in range(4000):
            tot = 0.0; c = 0
            for nm, k in cnt.items():
                arr = pool[nm][key]
                tot += hit(arr[rng.integers(0, len(arr), k)]).sum(); c += k
            sims[s] = tot / c
        base = sims.mean(); n = len(sub); kk = int(round(obs * n))
        pv = sum(comb(n, j) * base ** j * (1 - base) ** (n - j) for j in range(kk, n + 1))
        print(f"    {metric:<16} 信号 {kk}/{n} = {obs*100:5.1f}%   基线 {base*100:5.1f}%   "
              f"P(≥观测|基线)={pv:.5f}   提升 {(obs-base)*100:+5.1f}pp")


baselines(PP, DP[DP["bna_pct"] >= 60], "生产 6 板块")
baselines(LP, LT[LT["bna_pct"] >= 60], "长历史 8 宽基 · trail")
baselines(EP, LE[LE["bna_pct"] >= 60], "长历史 8 宽基 · expand（含 2015-2016）")

print("\n" + "=" * 136)
print("⑤ 合并两个独立样本集：bna_pct>=60 的 T+60 盈利")
print("=" * 136)
a = DP[DP["bna_pct"] >= 60]; b = LE[LE["bna_pct"] >= 60]
na, ka = len(a), int((a["ret60"] > 0).sum())
nb, kb = len(b), int((b["ret60"] > 0).sum())
print(f"  生产 {ka}/{na}   +   长历史expand {kb}/{nb}   =   {ka+kb}/{na+nb}")
for base in (0.53, 0.60, 0.65, 0.70):
    print(f"    若基线 {base*100:.0f}%  →  P(至少 {ka+kb}/{na+nb}) = "
          f"{sum(comb(na+nb, j)*base**j*(1-base)**(na+nb-j) for j in range(ka+kb, na+nb+1)):.6f}")
