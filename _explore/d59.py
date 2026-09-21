"""验证「只在恐慌放量时买」（cf_b > 0）能否泛化。

生产上：19 次 → 9 次，命中 17/19 → **9/9**，最差回撤 −23.61% → **−2.48%**。
但这个条件是从 19 个样本的诊断里发现的，必须在长历史上验：
  - trailing-750 锚（与生产同款）
  - 扩展因果锚（能覆盖 2015-2016 最难区段）
两套都验过才算数。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C
from senti import model as M
import longhist

H, TOL, COOL, THR = 20, 0.03, 20, 0.0
W2 = pd.Timestamp("2023-09-20")
LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
LO_Q, HI_Q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
EXCLUDE = {"CSI2000"}
KEYS = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20", "disp", "dd250"]
WIN, MINP = C.MODEL["anchor_window"], C.MODEL["anchor_min"]


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


def score2(raw, a, k_g=M.K_G):
    num = pd.Series(0.0, index=raw.index); den = 0.0
    for k, w in M.L_WEIGHTS.items():
        num = num + pmm(raw[k], a, k) * w; den += w
    L = num / den
    T = (pmm(raw["bias"], a, "bias") * M.T_WEIGHTS["bias"]
         + pmm(raw["ret20"], a, "ret20") * M.T_WEIGHTS["ret20"]
         + pmm(raw["rsi"], a, "rsi") * M.T_WEIGHTS["rsi"]
         + raw["amt_pct"] * 100.0 * M.T_WEIGHTS["amt"])
    U = M.W_L * L + (1 - M.W_L) * T
    cfb = (((raw["amt_pct"].astype(float) - M.CF_AMT_LO) / (1.0 - M.CF_AMT_LO)).clip(0, 1).fillna(0.0)
           * ((M.CF_L_HI - L) / M.CF_L_HI).clip(0, 1).fillna(0.0)).fillna(0.0)
    return (U - M.K_B * cfb - k_g * 0.0).clip(M.CLIP_LO, M.CLIP_HI), cfb


def entries(s, close, cfb, thr=THR, cf_lo=None):
    dates = list(s.index)
    s = s.to_numpy(float); cl = close.to_numpy(float); cf = cfb.to_numpy(float)
    out = []; last = -10 ** 9; n = len(s)
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if not (s[i - 1] <= thr and s[i] > s[i - 1]): continue
        if cf_lo is not None and (np.isnan(cf[i]) or cf[i] <= cf_lo): continue
        last = i
        c0 = cl[i]; seg = cl[i + 1:i + H + 1]
        out.append((dates[i], seg[-1] / c0 - 1, seg.min() / c0 - 1, s[i]))
    return out


def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def line(rows, label):
    if not rows:
        print(f"  {label:<26} 无样本"); return
    o = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
    nt = sum(1 for x in rows if x[3] >= -TOL)
    print(f"  {label:<26} n={len(rows):>3}  命中 {o}/{len(rows):<3} {o/len(rows)*100:5.1f}%   "
          f"未被套 {nt/len(rows)*100:5.1f}%   均T+20 {np.mean([x[2] for x in rows])*100:+6.2f}%   "
          f"最差 {min(x[3] for x in rows)*100:+7.2f}%   下界 {wilson(o,len(rows))*100:5.1f}%")


PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
print("板块:", " ".join(sorted(PAN.keys())), flush=True)

for mode, tag in (("trail", "trailing-750（生产同款）"), ("expand", "扩展因果锚（含2015-2016）")):
    A = {nm: make_anchors(raw, mode) for nm, raw in PAN.items()}
    SC = {nm: score2(PAN[nm], A[nm]) for nm in PAN}
    print("\n" + "=" * 128)
    print(f"长历史 · {tag} · 阈值 0 · K_G=0")
    print("=" * 128)
    RES = {}
    for lab, lo in (("不限 cf_b（现状）", None), ("cf_b > 0", 0.0),
                    ("cf_b > 0.05", 0.05), ("cf_b > 0.10", 0.10)):
        rows = []
        for nm, raw in PAN.items():
            sc, cfb = SC[nm]
            rows += [(nm,) + e for e in entries(sc, raw["close"], cfb, THR, lo)]
        RES[lab] = rows
        line(rows, lab)
        line([x for x in rows if x[1] >= W2], lab + " · 交付窗口")
    print("\n  【分年度】")
    yrs = sorted({x[1].year for r in RES.values() for x in r})
    print(f"    {'条件':<20}" + "".join(f"{y:>8}" for y in yrs))
    for lab, rows in RES.items():
        cells = ""
        for y in yrs:
            sub = [x for x in rows if x[1].year == y]
            cells += (f"{sum(1 for x in sub if x[2] > 0 and x[3] >= -TOL)}/{len(sub)}" if sub else "-").rjust(8)
        print(f"    {lab:<20}" + cells)
