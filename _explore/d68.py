"""给「reso_d0>=2 → T+60 盈利 26/26」找一个对照组。

不放对照就不能下结论：A 股宽基持有 60 个交易日（约 3 个月）本来上涨概率就不低。
如果随机买入的 T+60 盈利基线是 85%，那 18/18 的惊喜程度就大打折扣。

对照口径（同一批板块、同一段历史、同样的持有期）：
  B1 全样本基线        该板块所有交易日买入，T+H 盈利的比例
  B2 分板块基线        同上，按板块分别算（信号在不同板块分布不均，必须加权对齐）
  B3 事件数量加权基线   按 reso_d0>=2 信号在各板块的实际分布，对 B2 加权
  B4 同月基线          只在信号所在月份内随机取日（控制年代/估值水平）
然后算：在 B3 基线下，观察到 26/26 的概率（二项尾概率）。
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


def signals(panels, t0=None):
    S = pd.DataFrame({nm: p["score"] for nm, p in panels.items()}).dropna(how="all")
    CL = pd.DataFrame({nm: p["close"] for nm, p in panels.items()}).reindex(S.index)
    Z = (S <= THR)
    rows = []
    for nm in panels:
        v = S[nm].to_numpy(float); cl = CL[nm].to_numpy(float)
        n = len(v); last = -10 ** 9
        for i in range(1, n - H2):
            if t0 is not None and S.index[i] < t0: continue
            if np.isnan(v[i]) or np.isnan(v[i - 1]): continue
            if i - last < COOL: continue
            if v[i - 1] > THR or v[i] <= v[i - 1]: continue
            last = i
            c0 = cl[i]
            seg = cl[i + 1:i + H + 1]; seg2 = cl[i + 1:i + H2 + 1]
            rows.append({
                "board": nm, "date": S.index[i], "idx": i,
                "ret20": seg[-1] / c0 - 1, "mdd20": seg.min() / c0 - 1,
                "ret60": seg2[-1] / c0 - 1,
                "ok20": bool(seg[-1] / c0 - 1 > 0 and seg.min() / c0 - 1 >= -TOL),
                "reso_d0": int(Z.iloc[i].sum()),
            })
    return pd.DataFrame(rows)


def report(name, panels, sig, sub):
    print("\n" + "=" * 128)
    print(f"{name}")
    print("=" * 128)
    # 板块分布对齐的随机基线
    board_w = sig["board"].value_counts(normalize=True)
    cnt = sub["board"].value_counts()
    pool = {}
    for nm, p in panels.items():
        cl = p["close"].to_numpy(float); n = len(cl); hi = n - H2 - 1
        c0 = cl[:hi]
        pool[nm] = {"r60": cl[H2:hi + H2] / c0 - 1,
                    "r20": cl[H:hi + H] / c0 - 1,
                    "m20": np.array([cl[j + 1:j + H + 1].min() / cl[j] - 1 for j in range(hi)])}
    # (口径名, 池里的列名, 信号表里的列名, 判定方式)
    for metric, key, scol, lab in (("T+60 盈利", "r60", "ret60", "pos"),
                                   ("T+20 盈利", "r20", "ret20", "pos"),
                                   ("H20 未被套>3%", "m20", "mdd20", "tol")):
        def hit(arr):
            return (arr > 0) if lab == "pos" else (arr >= -TOL)
        obs = float(hit(sub[scol].to_numpy()).mean())
        sims = np.empty(4000)
        for s in range(4000):
            tot = 0.0; c = 0
            for nm, k in cnt.items():
                arr = pool[nm][key]
                tot += hit(arr[rng.integers(0, len(arr), k)]).sum()
                c += k
            sims[s] = tot / c
        base = sims.mean()
        # 二项尾概率：在基线 base 下，观察到 obs*n 次里全部/这么多次成功的概率
        n = len(sub); k = int(round(obs * n))
        from math import comb
        pval = sum(comb(n, j) * base ** j * (1 - base) ** (n - j) for j in range(k, n + 1))
        print(f"  {metric:<16} 信号 {k}/{n} = {obs*100:5.1f}%   随机基线 {base*100:5.1f}% ± {sims.std()*100:.1f}%   "
              f"P(≥观测值 | 基线) = {pval:.4f}   提升 {(obs-base)*100:+.1f}个百分点")
    # 全样本基线（不加权，简单对照）
    allr = np.concatenate([pool[nm]["r60"] for nm in panels])
    print(f"  {'（参考）全样本 T+60 盈利基线':<16} {(allr>0).mean()*100:5.1f}%")


RAW = {b: F.build_board_raw(b) for b in C.BOARD_ORDER}
PROD_PANELS = {b: pd.DataFrame({"score": score(RAW[b], make_anchors(RAW[b], "trail")),
                                "close": RAW[b]["close"]}) for b in C.BOARD_ORDER}
DP = signals(PROD_PANELS)
PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
LH_PANELS = {nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "trail")), "close": raw["close"]})
             for nm, raw in PAN.items()}
LT = signals(LH_PANELS)

report("① 生产 6 板块 · reso_d0 >= 2", PROD_PANELS, DP, DP[DP["reso_d0"] >= 2])
report("①b 生产 6 板块 · 全部信号（对照）", PROD_PANELS, DP, DP)
report("② 长历史 8 宽基 · reso_d0 >= 2", LH_PANELS, LT, LT[LT["reso_d0"] >= 2])
report("②b 长历史 8 宽基 · 全部信号（对照）", LH_PANELS, LT, LT)

print("\n" + "=" * 128)
print("③ 合并：reso_d0>=2 的 T+60 盈利合计")
print("=" * 128)
a = DP[DP["reso_d0"] >= 2]; b = LT[LT["reso_d0"] >= 2]
n = len(a) + len(b); k = int((a["ret60"] > 0).sum() + (b["ret60"] > 0).sum())
print(f"  生产 {int((a['ret60']>0).sum())}/{len(a)}   +   长历史 {int((b['ret60']>0).sum())}/{len(b)}"
      f"   =   {k}/{n}")
from math import comb
for base in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
    print(f"    若真实基线 = {base*100:.0f}%  →  P({n}/{n} 全中) = {base**n:.5f}")
