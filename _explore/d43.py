"""方案 B：把「样本分位锚」换成「语义固定的绝对锚」。

动机：d39/d42 暴露出 p2/p98 锚点会随样本漂移，−6 阈值不可移植。
而这个模型里绝大多数因子本身就有自然单位，根本不需要靠样本分位来归一化：

  广度类  b20/b60/r5 ∈ [0,1]、nh/lim ∈ [−1,1]   —— 天然有刻度
  技术类  rsi ∈ [0,100]、bias/ret20 是百分比     —— 天然有刻度
  成交额  amt_pct 已经是滚动分位                 —— 本身就是 [0,1]

所以可以给出一组「不依赖样本」的固定锚点。关键在于这些值必须**先于回测给出**，
是行业常识值而不是调出来的：RSI 20/80 是教科书定义，bias ±20% 是长周期乖离的常识量级。

disp（截面离散度）是唯一没有自然单位的，用「自身 250 日滚动中位数」归一成无量纲比率。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C
from senti import model as M
import longhist

H, TOL, COOL = 20, 0.03, 20
W2 = pd.Timestamp("2023-09-20")
LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]

# ---- 语义固定锚点（先于回测给出的常识值，不随样本变化）----
ANCH = {
    "b20":   (0.05, 0.90),
    "b60":   (0.05, 0.90),
    "r5":    (0.05, 0.85),
    "nh":    (-0.30, 0.20),
    "lim":   (-0.15, 0.10),
    "rsi":   (20.0, 80.0),
    "bias":  (-0.20, 0.20),
    "ret20": (-0.25, 0.25),
    "dr":    (0.60, 1.60),     # disp 相对自身常态的比率
    "dd250": (-0.45, -0.02),   # 距 250 日高点回撤；越深越恐慌 → 反向
}


def pmm(v, k):
    lo, hi = ANCH[k]
    return (100.0 * (v - lo) / (hi - lo)).clip(LC, HC)


def score(raw):
    num = pd.Series(0.0, index=raw.index); den = 0.0
    for k, w in M.L_WEIGHTS.items():
        num = num + pmm(raw[k], k) * w; den += w
    L = num / den
    T = (pmm(raw["bias"], "bias") * M.T_WEIGHTS["bias"]
         + pmm(raw["ret20"], "ret20") * M.T_WEIGHTS["ret20"]
         + pmm(raw["rsi"], "rsi") * M.T_WEIGHTS["rsi"]
         + raw["amt_pct"] * 100.0 * M.T_WEIGHTS["amt"])
    U = M.W_L * L + (1 - M.W_L) * T

    cfb = (((raw["amt_pct"].astype(float) - M.CF_AMT_LO) / (1.0 - M.CF_AMT_LO)).clip(0, 1).fillna(0.0)
           * ((M.CF_L_HI - L) / M.CF_L_HI).clip(0, 1).fillna(0.0)).fillna(0.0)

    dd = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    med = raw["disp"].rolling(250, min_periods=60).median()
    dr = (raw["disp"] / med.replace(0.0, np.nan)).clip(0.0, 4.0)
    f_disp = (100.0 - pmm(dr, "dr")).fillna(0.0)
    f_dd = (100.0 - pmm(dd, "dd250")).fillna(0.0)
    cfg = (((f_dd - M.GR_DD_A) / M.GR_DD_W).clip(0, 1)
           * ((f_disp - M.GR_DP_A) / M.GR_DP_W).clip(0, 1)).fillna(0.0)
    return (U - M.K_B * cfb - M.K_G * cfg).clip(M.CLIP_LO, M.CLIP_HI)


def entries(s, c, thr):
    dates = list(s.index)
    s = s.to_numpy(float); cl = c.to_numpy(float)
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


def stat(rows, label):
    if not rows:
        print(f"  {label:<26} 无样本"); return
    o = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
    nt = sum(1 for x in rows if x[3] >= -TOL)
    print(f"  {label:<26} 样本 {len(rows):>3}  命中 {o}/{len(rows)} = {o/len(rows)*100:5.1f}%   "
          f"未被套 {nt}/{len(rows)} = {nt/len(rows)*100:5.1f}%   "
          f"均T+20 {np.mean([x[2] for x in rows])*100:+6.2f}%   "
          f"最差 {min(x[3] for x in rows)*100:+6.2f}%   威尔逊下界 {wilson(o,len(rows))*100:5.1f}%")


print("载入长历史面板 ...", flush=True)
PAN = longhist.build()
SC = {}
for nm, raw in PAN.items():
    SC[nm] = score(raw)
    print(f"  {nm:<8} {len(raw)} 行  分值区间 "
          f"{np.nanmin(SC[nm].to_numpy()):.0f}~{np.nanmax(SC[nm].to_numpy()):.0f}", flush=True)

print("\n" + "=" * 128)
print("语义固定锚（不依赖样本分位）· 长历史 2013~2026")
print("=" * 128)

print("\n【阈值扫描 · 全程】")
for t in [0, -2, -4, -6, -8, -10, -12, -15, -20]:
    rows = []
    for nm, raw in PAN.items():
        rows += [(nm,) + e for e in entries(SC[nm], raw["close"], thr=float(t))]
    stat(rows, f"阈值 {t:>3}")

print("\n【阈值扫描 · 交付窗口 2023-09-20~今】")
for t in [0, -2, -4, -6, -8, -10, -12, -15, -20]:
    rows = []
    for nm, raw in PAN.items():
        rows += [(nm,) + e for e in entries(SC[nm], raw["close"], thr=float(t)) if e[0] >= W2]
    stat(rows, f"阈值 {t:>3}")

print("\n【分年度 · 阈值 −6】")
rows = []
for nm, raw in PAN.items():
    rows += [(nm,) + e for e in entries(SC[nm], raw["close"], thr=-6.0)]
for y in sorted({x[1].year for x in rows}):
    stat([x for x in rows if x[1].year == y], f"{y} 年")

print("\n【分板块 · 阈值 −6 · 全程】")
for nm in sorted({x[0] for x in rows}):
    stat([x for x in rows if x[0] == nm], nm)
