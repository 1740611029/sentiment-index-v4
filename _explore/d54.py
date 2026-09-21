"""验证 d53 发现的两条线索，放到长历史（含 2015-2016 最难区段）里检验。

线索 1：cf_g（磨底项）在因果锚下与失败绑定 —— 试 K_G = 30 / 15 / 0
线索 2：失败时成交额分位低、且仍在创新低 —— 试过滤条件

用**扩展因果锚**，因为只有它能在 2015-2016 产生信号（trailing 750 要到 2020 年才有信号）。
若在 2015-2016 也能改善，才算真的有用，否则只是 22 个样本的拟合。
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


def expand_anchor(v, min_n=500, step=20):
    x = v.to_numpy(float); n = len(x)
    lo = np.full(n, np.nan); hi = np.full(n, np.nan)
    j = min_n
    while j < n:
        seg = x[:j]; seg = seg[~np.isnan(seg)]
        if len(seg) >= min_n:
            k = min(j + step, n)
            lo[j:k] = np.nanpercentile(seg, LO_Q); hi[j:k] = np.nanpercentile(seg, HI_Q)
        j += step
    return pd.Series(lo, index=v.index).ffill(), pd.Series(hi, index=v.index).ffill()


def make_anchors(raw):
    src = raw.copy()
    src["dd250"] = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    return {k: expand_anchor(src[k]) for k in KEYS}


def pmm(v, a, k):
    lo, hi = a[k]; d = hi - lo
    if isinstance(d, pd.Series):
        d = d.replace(0.0, np.nan)
    elif d == 0:
        d = np.nan
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


def entries(s, close, amt_pct, thr=THR, amt_lo=None, no_new_low=False):
    dates = list(s.index)
    s = s.to_numpy(float); cl = close.to_numpy(float)
    am = amt_pct.to_numpy(float)
    out = []; last = -10 ** 9; n = len(s)
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if not (s[i - 1] <= thr and s[i] > s[i - 1]): continue
        if amt_lo is not None and (np.isnan(am[i]) or am[i] < amt_lo): continue
        if no_new_low and i >= 5 and cl[i] <= np.nanmin(cl[i - 4:i + 1]): continue
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
        print(f"  {label:<34} 无样本"); return
    o = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
    nt = sum(1 for x in rows if x[3] >= -TOL)
    print(f"  {label:<34} n={len(rows):>3}  命中 {o}/{len(rows):<3} {o/len(rows)*100:5.1f}%   "
          f"未被套 {nt/len(rows)*100:5.1f}%   均T+20 {np.mean([x[2] for x in rows])*100:+6.2f}%   "
          f"最差 {min(x[3] for x in rows)*100:+7.2f}%   下界 {wilson(o,len(rows))*100:5.1f}%")


PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
print("板块:", " ".join(sorted(PAN.keys())), flush=True)
A = {nm: make_anchors(raw) for nm, raw in PAN.items()}

CONFIGS = [
    ("① 现状 K_G=30",                 30,   None, False),
    ("② K_G=15",                      15,   None, False),
    ("③ K_G=0（去掉磨底项）",           0.0,  None, False),
    ("④ K_G=0 + 成交额分位≥0.35",       0.0,  0.35, False),
    ("⑤ K_G=0 + 近5日不创新低",         0.0,  None, True),
    ("⑥ K_G=0 + 两条都要",              0.0,  0.35, True),
    ("⑦ 现状 + 成交额分位≥0.35",        30,   0.35, False),
    ("⑧ 现状 + 两条都要",               30,   0.35, True),
]

print("\n" + "=" * 132)
print("长历史（2013~2026，8 宽基，扩展因果锚，阈值 0）—— 看能否救回 2015-2016")
print("=" * 132)
RES = {}
for tag, kg, alo, nnl in CONFIGS:
    rows = []
    for nm, raw in PAN.items():
        sc = score(raw, A[nm], kg)
        rows += [(nm,) + e for e in entries(sc, raw["close"], raw["amt_pct"],
                                            THR, alo, nnl)]
    RES[tag] = rows
    line(rows, tag)

print("\n" + "=" * 132)
print("分年度（命中/样本）")
print("=" * 132)
yrs = sorted({x[1].year for r in RES.values() for x in r})
print(f"  {'配置':<28}" + "".join(f"{y:>8}" for y in yrs) + f"{'交付窗口':>10}")
for tag, rows in RES.items():
    cells = ""
    for y in yrs:
        sub = [x for x in rows if x[1].year == y]
        cells += (f"{sum(1 for x in sub if x[2] > 0 and x[3] >= -TOL)}/{len(sub)}" if sub else "-").rjust(8)
    w = [x for x in rows if x[1] >= W2]
    cells += (f"{sum(1 for x in w if x[2] > 0 and x[3] >= -TOL)}/{len(w)}" if w else "-").rjust(10)
    print(f"  {tag:<28}" + cells)

print("\n" + "=" * 132)
print("交付窗口细看（2023-09-20 ~ 今）")
print("=" * 132)
for tag, rows in RES.items():
    line([x for x in rows if x[1] >= W2], tag)
