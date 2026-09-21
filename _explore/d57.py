"""交叉验证：用**生产同款** trailing-750 因果锚，在长历史上再验一次 K_G=0。

d54/d55 用的是扩展因果锚；生产实际用 trailing-750。两套都验过才算数。
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


def make_anchors(raw):
    src = raw.copy()
    src["dd250"] = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    return {k: (src[k].rolling(WIN, min_periods=MINP).quantile(LO_Q / 100.0).shift(1),
                src[k].rolling(WIN, min_periods=MINP).quantile(HI_Q / 100.0).shift(1)) for k in KEYS}


def pmm(v, a, k):
    lo, hi = a[k]; d = (hi - lo).replace(0.0, np.nan)
    return (100.0 * (v - lo) / d).clip(LC, HC)


def score(raw, a, k_g=M.K_G):
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
    dd = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    f_disp = (100.0 - pmm(raw["disp"], a, "disp")).fillna(0.0)
    f_dd = (100.0 - pmm(dd, a, "dd250")).fillna(0.0)
    cfg = (((f_dd - M.GR_DD_A) / M.GR_DD_W).clip(0, 1)
           * ((f_disp - M.GR_DP_A) / M.GR_DP_W).clip(0, 1)).fillna(0.0)
    return (U - M.K_B * cfb - k_g * cfg).clip(M.CLIP_LO, M.CLIP_HI)


def entries(s, close, thr=THR):
    dates = list(s.index)
    s = s.to_numpy(float); cl = close.to_numpy(float)
    out = []; last = -10 ** 9; n = len(s)
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if s[i - 1] <= thr and s[i] > s[i - 1]:
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
A = {nm: make_anchors(raw) for nm, raw in PAN.items()}

print("\n" + "=" * 126)
print("长历史 · trailing-750 因果锚（与生产同款）· 阈值 0")
print("=" * 126)
RES = {}
for tag, kg in (("K_G = 30（旧）", 30.0), ("K_G = 0（新）", 0.0)):
    rows = []
    for nm, raw in PAN.items():
        rows += [(nm,) + e for e in entries(score(raw, A[nm], kg), raw["close"])]
    RES[tag] = rows
    line(rows, tag)
    line([x for x in rows if x[1] >= W2], tag + " · 交付窗口")

print("\n【分年度】")
yrs = sorted({x[1].year for r in RES.values() for x in r})
print(f"  {'配置':<18}" + "".join(f"{y:>8}" for y in yrs))
for tag, rows in RES.items():
    cells = ""
    for y in yrs:
        sub = [x for x in rows if x[1].year == y]
        cells += (f"{sum(1 for x in sub if x[2] > 0 and x[3] >= -TOL)}/{len(sub)}" if sub else "-").rjust(8)
    print(f"  {tag:<18}" + cells)
