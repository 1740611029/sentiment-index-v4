"""长历史（剔除 CSI2000 坏数据）· 入场规则变体扫描。

目的：d42 显示 2015/2016 是主要失分点（命中 25%）。这里检验：
  加价格确认 / 加回升幅度 / 加长冷却 / 加均线过滤，能否把 2015-2016 救回来，
  同时不破坏 2018/2020/2025 的表现。

评估不只看总数，必须分年度——只有跨牛熊都稳的变体才算数。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C
from senti import model as M
import longhist

H, TOL = 20, 0.03
W2 = pd.Timestamp("2023-09-20")
LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
LO_Q, HI_Q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
EXCLUDE = {"CSI2000"}          # 该指数 2023-08 才发布，2013-2023 段是替代序列，不可用

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
    cfb = (((raw["amt_pct"].astype(float) - M.CF_AMT_LO) / (1.0 - M.CF_AMT_LO)).clip(0, 1).fillna(0.0)
           * ((M.CF_L_HI - L) / M.CF_L_HI).clip(0, 1).fillna(0.0)).fillna(0.0)
    dd = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    f_disp = (100.0 - pmm(raw["disp"], a, "disp")).fillna(0.0)
    f_dd = (100.0 - pmm(dd, a, "dd250")).fillna(0.0)
    cfg = (((f_dd - M.GR_DD_A) / M.GR_DD_W).clip(0, 1)
           * ((f_disp - M.GR_DP_A) / M.GR_DP_W).clip(0, 1)).fillna(0.0)
    return (U - M.K_B * cfb - M.K_G * cfg).clip(M.CLIP_LO, M.CLIP_HI)


def ma(close, n):
    return close.rolling(n).mean()


def run(s, close, thr, cool, variant):
    dates = list(s.index)
    s = s.to_numpy(float); cl = close.to_numpy(float)
    mv = ma(close, 5).to_numpy(float)
    out = []; last = -10 ** 9; n = len(s)
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < cool: continue
        if not (s[i - 1] <= thr and s[i] > s[i - 1]):
            continue
        if variant == "V2" and not (cl[i] > cl[i - 1]): continue
        if variant == "V3" and not (s[i] - s[i - 1] >= 3.0): continue
        if variant == "V4" and not (i >= 2 and s[i - 1] > s[i - 2]): continue
        if variant == "V5" and not (cl[i] > mv[i]): continue
        if variant == "V6" and not (cl[i] > cl[i - 1] and not np.isnan(mv[i]) and cl[i] > mv[i]): continue
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
        print(f"  {label:<30} 无样本"); return
    o = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
    nt = sum(1 for x in rows if x[3] >= -TOL)
    print(f"  {label:<30} n={len(rows):>3}  命中 {o}/{len(rows):<3} {o/len(rows)*100:5.1f}%   "
          f"未被套 {nt/len(rows)*100:5.1f}%   均T+20 {np.mean([x[2] for x in rows])*100:+6.2f}%   "
          f"最差 {min(x[3] for x in rows)*100:+7.2f}%   下界 {wilson(o,len(rows))*100:5.1f}%")


PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
print("板块:", " ".join(sorted(PAN.keys())), flush=True)
SC = {nm: score(raw, make_anchors(raw)) for nm, raw in PAN.items()}
CL = {nm: raw["close"] for nm, raw in PAN.items()}

VARIANTS = [
    ("V1 昨≤thr 且回升（现状）", "V1", 20),
    ("V2 +当日收涨", "V2", 20),
    ("V3 +回升幅度≥3分", "V3", 20),
    ("V4 +昨日已在回升", "V4", 20),
    ("V5 +收盘站上MA5", "V5", 20),
    ("V6 收涨 且 站上MA5", "V6", 20),
    ("V1c40 现状+冷却40", "V1", 40),
    ("V1c60 现状+冷却60", "V1", 60),
    ("V6c40 收涨站上MA5+冷却40", "V6", 40),
]

print("\n" + "=" * 130)
print("规则变体扫描（扩展因果锚 · 阈值 −6 · 全程 2015~2026）")
print("=" * 130)
RES = {}
for label, var, cool in VARIANTS:
    rows = []
    for nm in PAN:
        rows += [(nm,) + e for e in run(SC[nm], CL[nm], -6.0, cool, var)]
    RES[label] = rows
    line(rows, label)

print("\n" + "=" * 130)
print("分年度命中率（只有跨牛熊都稳的才算数）")
print("=" * 130)
yrs = sorted({x[1].year for r in RES.values() for x in r})
hdr = f"  {'变体':<24}" + "".join(f"{y:>7}" for y in yrs) + f"{'交付窗口':>10}"
print(hdr)
for label, rows in RES.items():
    cells = ""
    for y in yrs:
        sub = [x for x in rows if x[1].year == y]
        if not sub:
            cells += f"{'-':>7}"; continue
        o = sum(1 for x in sub if x[2] > 0 and x[3] >= -TOL)
        cells += f"{o}/{len(sub)}".rjust(7)
    w = [x for x in rows if x[1] >= W2]
    if w:
        o = sum(1 for x in w if x[2] > 0 and x[3] >= -TOL)
        cells += f"{o}/{len(w)}".rjust(10)
    else:
        cells += f"{'-':>10}"
    print(f"  {label:<24}" + cells)

print("\n【最佳候选 · V6 收涨且站上MA5 · 阈值扫描】")
for t in [-2, -4, -6, -8, -10, -12]:
    rows = []
    for nm in PAN:
        rows += [(nm,) + e for e in run(SC[nm], CL[nm], float(t), 20, "V6")]
    line(rows, f"阈值 {t:>3}")
print("\n【V6 分年度明细 · 阈值 −6】")
rows = []
for nm in PAN:
    rows += [(nm,) + e for e in run(SC[nm], CL[nm], -6.0, 20, "V6")]
for y in sorted({x[1].year for x in rows}):
    line([x for x in rows if x[1].year == y], f"{y} 年")
