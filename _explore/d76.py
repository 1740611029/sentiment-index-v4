"""d76：既然信号是「市场级事件」，就该有一条全市场分值，一次事件只出一次信号。

背景（d74/d75 的结论）：
  - 近 3 年 17 次底部信号只落在 5 个日期 → 本质是市场级恐慌事件
  - 因此按板块分别出信号 = 同一次事件重复计 6 遍 → 样本量虚高、p 值假精确
  - 而且共振层定义「信号日收盘还有几个板块 ≤0」几乎永不成立
    （因为大家是同日一起回升的）→ A 级只有 3 个，最准的那一层形同虚设

d76 要回答：
  ① 用全市场分值（6 板块分值的中位数）出信号，是不是一次事件只响一次？
  ② 它的命中率 vs 现在的分板块口径（近 3 年 / 13 年长历史）
  ③ 「事件广度」能不能替代共振层：±5 交易日内有多少个板块跌到过 ≤0
  ④ 事件来了买什么？等权买全部 / 买当日分值最低的那个，哪个更好
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


def build_panels(raws, mode):
    return {nm: pd.DataFrame({"score": score(r, make_anchors(r, mode)), "close": r["close"]})
            for nm, r in raws.items()}


def mkt_signals(panels, thr=0.0, cool=COOL):
    """全市场分值 = 各板块分值的中位数；一次事件只出一次信号"""
    S = pd.DataFrame({nm: p["score"] for nm, p in panels.items()}).dropna(how="all")
    CL = pd.DataFrame({nm: p["close"] for nm, p in panels.items()}).reindex(S.index)
    ms = S.median(axis=1)
    mv = ms.to_numpy(float)
    names = list(panels.keys())
    cl = CL.to_numpy(float)
    n = len(mv); last = -10 ** 9
    rows = []
    for i in range(1, n - H2):
        if np.isnan(mv[i]) or np.isnan(mv[i - 1]): continue
        if i - last < cool: continue
        if mv[i - 1] > thr or mv[i] <= mv[i - 1]: continue
        last = i
        # 等权买入所有板块
        r20_ew = np.mean([cl[i + H, j] / cl[i, j] - 1 for j in range(len(names))])
        r60_ew = np.mean([cl[i + H2, j] / cl[i, j] - 1 for j in range(len(names))])
        mdd_ew = np.mean([cl[i + 1:i + H + 1, j].min() / cl[i, j] - 1 for j in range(len(names))])
        # 买当日分值最低的板块（最恐慌的那个）
        jmin = int(np.nanargmin(S.iloc[i].to_numpy(float)))
        r20_lo = cl[i + H, jmin] / cl[i, jmin] - 1
        mdd_lo = cl[i + 1:i + H + 1, jmin].min() / cl[i, jmin] - 1
        # 广度：信号日当天还有几个板块 ≤0（现口径）
        b0 = int((S.iloc[i] <= thr).sum())
        # 事件广度：±5 交易日内有几个板块跌到过 ≤0（新口径）
        lo2 = max(0, i - 5); hi2 = min(n - 1, i + 5)
        b5 = int((S.iloc[lo2:hi2 + 1] <= thr).any(axis=0).sum())
        rows.append({"date": S.index[i], "mscore": mv[i], "mprev": mv[i - 1],
                     "r20_ew": r20_ew, "r60_ew": r60_ew, "mdd_ew": mdd_ew,
                     "r20_lo": r20_lo, "mdd_lo": mdd_lo, "lowest": names[jmin],
                     "b0": b0, "b5": b5, "nboard": len(names),
                     "ok_ew": bool(r20_ew > 0 and mdd_ew >= -TOL)})
    return pd.DataFrame(rows), ms


def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def tail(k, n, p0):
    return sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1))


def ew_baseline(panels, hp=H, nsim=8000):
    CL = pd.DataFrame({nm: panels[nm]["close"] for nm in panels}).dropna(how="all")
    cl = CL.to_numpy(float); n = len(cl)
    idx = rng.integers(0, n - hp - 1, nsim)
    r = np.array([np.mean(cl[i + hp, :] / cl[i, :] - 1) for i in idx])
    return (r > 0).mean() * 100


def rep(tag, panels):
    df, ms = mkt_signals(panels)
    if len(df) == 0:
        print(f"\n【{tag}】无信号"); return None
    n = len(df); w = int(df["ok_ew"].sum()); w60 = int((df["r60_ew"] > 0).sum())
    base = ew_baseline(panels)
    print(f"\n【{tag}】全市场信号 {n} 次（等权买全部板块）")
    print(f"  命中 {w}/{n} = {w/n*100:.1f}%　下界 {wil(w,n)*100:.1f}%　基线 {base:.1f}%"
          f"　p={tail(w,n,base/100):.4f}")
    print(f"  T+60 盈利 {w60}/{n} = {w60/n*100:.1f}%　均 T+20 {df['r20_ew'].mean()*100:+.2f}%"
          f"　最差 {df['r20_ew'].min()*100:+.2f}%　最深浮亏 {df['mdd_ew'].min()*100:+.2f}%")
    print(f"  {'信号日':<12}{'昨分值':>8}{'当日':>8}{'广度±5':>7}{'当日≤0':>7}"
          f"{'T+20等权':>10}{'T+20最低':>10}{'最低板块':>10}{'判定':>5}")
    for r in df.itertuples():
        print(f"  {str(r.date)[:10]:<12}{r.mprev:>8.1f}{r.mscore:>8.1f}{r.b5:>7}{r.b0:>7}"
              f"{r.r20_ew*100:>9.2f}%{r.r20_lo*100:>9.2f}%{r.lowest:>10}"
              f"{'✔' if r.r20_ew>0 else '✘':>5}")
    return df


print("=" * 118)
print("d76 · 全市场分值出信号：一次市场级恐慌 = 一次信号")
print("=" * 118)

RAW = {b: F.build_board_raw(b) for b in C.BOARD_ORDER}
PP = build_panels(RAW, "trail")
PP = {b: p[p.index >= W2] for b, p in PP.items()}
DP = rep("生产 6 板块 2023-09-20 起", PP)

PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
LP = build_panels(PAN, "trail")
EP = build_panels(PAN, "expand")
LT = rep("长历史 8 宽基 · trail 锚", LP)
EE = rep("长历史 8 宽基 · 扩展锚", EP)

print("\n" + "=" * 118)
print("③ 事件广度（±5 交易日内跌到过 ≤0 的板块数）能不能当确认层？")
print("=" * 118)
for tag, df in [("生产 6 板块", DP), ("长历史 trail", LT), ("长历史 扩展锚", EE)]:
    if df is None or len(df) == 0: continue
    print(f"\n  【{tag}】")
    base = ew_baseline(PP if tag.startswith("生产") else (LP if "trail" in tag else EP))
    print(f"    {'条件':<24}{'样本':>6}{'命中':>8}{'下界':>8}{'T+60':>8}{'均T+20':>9}  p值")
    for lab, m in [("广度±5 ≥ 6", df["b5"] >= 6), ("广度±5 ≥ 5", df["b5"] >= 5),
                   ("广度±5 ≥ 4", df["b5"] >= 4), ("广度±5 ≥ 3", df["b5"] >= 3),
                   ("（对照）全部信号", df["b5"] >= 0)]:
        g = df[m]
        if len(g) == 0:
            print(f"    {lab:<24}{0:>6}"); continue
        k = int(g["ok_ew"].sum()); n = len(g)
        print(f"    {lab:<24}{n:>6}{k/n*100:>7.1f}%{wil(k,n)*100:>7.1f}"
              f"{int((g['r60_ew']>0).sum())/n*100:>7.1f}%{g['r20_ew'].mean()*100:>9.2f}"
              f"  {tail(k,n,base/100):.4f}")
    # 现口径共振（信号日当天 ≤0 的板块数）对比
    for lab, m in [("现口径：当日≤0 ≥2", df["b0"] >= 2), ("现口径：当日≤0 ≥1", df["b0"] >= 1)]:
        g = df[m]
        if len(g) == 0:
            print(f"    {lab:<24}{0:>6}"); continue
        k = int(g["ok_ew"].sum()); n = len(g)
        print(f"    {lab:<24}{n:>6}{k/n*100:>7.1f}%{wil(k,n)*100:>7.1f}"
              f"{int((g['r60_ew']>0).sum())/n*100:>7.1f}%{g['r20_ew'].mean()*100:>9.2f}"
              f"  {tail(k,n,base/100):.4f}")

print("\n" + "=" * 118)
print("④ 事件来了买什么：等权买全部 vs 买当日分值最低的板块")
print("=" * 118)
for tag, df in [("生产 6 板块", DP), ("长历史 trail", LT), ("长历史 扩展锚", EE)]:
    if df is None or len(df) == 0: continue
    n = len(df)
    print(f"  {tag:<16} 等权全部：命中 {int(df['ok_ew'].sum())}/{n}"
          f"　均T+20 {df['r20_ew'].mean()*100:+.2f}%　最深 {df['mdd_ew'].min()*100:+.2f}%")
    klo = int(((df["r20_lo"] > 0) & (df["mdd_lo"] >= -TOL)).sum())
    print(f"  {'':<16} 买最低分：命中 {klo}/{n}"
          f"　均T+20 {df['r20_lo'].mean()*100:+.2f}%　最深 {df['mdd_lo'].min()*100:+.2f}%")
    print(f"  {'':<16} 买最低分赢过等权的比例：{int((df['r20_lo']>df['r20_ew']).sum())}/{n}")
