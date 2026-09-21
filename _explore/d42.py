"""长历史锚点稳定性检验：三种锚定口径 × 两个评估窗口。

背景：d39 证明「−6 这个阈值刻在一把会伸缩的尺子上」——
_pct_map 用样本自身 p2/p98 作锚，标定样本从 726 天变 803 天，
命中率就从 95.1% 掉到 67.4%。

本脚本检验：把历史拉长到 13 年后，p2/p98 是否足够稳定，−6 是否可移植。

三种锚定口径：
  A 全样本固定锚     —— 用 2013~2026 全样本的 p2/p98（非因果，能力上界）
  B 扩展因果锚       —— 只用 t 之前的全部历史算 p2/p98（因果，可实盘）
  C 生产现状（3年）  —— 对照：仍用近 3 年

评估窗口：
  全程      2015-01 ~ 2026-09
  交付窗口  2023-09-20 ~ 2026-09-20（页面实际展示的那段）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C
from senti import model as M
import longhist

H, TOL, COOL, THR = 20, 0.03, 20, -6.0
STEP, MIN_N = 20, 500            # 因果锚：每 20 天更新一次，至少 500 个交易日历史
W1 = pd.Timestamp("2015-01-01")  # 因果锚可用起点
W2 = pd.Timestamp("2023-09-20")  # 交付窗口起点

LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
LO_Q, HI_Q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]

KEYS = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20", "disp", "dd250"]


def expand_anchor(v: pd.Series, min_n=MIN_N, step=STEP):
    """因果锚：第 i 天用的锚点只由 [0, i) 的历史算得。"""
    x = v.to_numpy(float)
    n = len(x)
    lo = np.full(n, np.nan); hi = np.full(n, np.nan)
    j = min_n
    while j < n:
        seg = x[:j]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= min_n:
            k = min(j + step, n)
            lo[j:k] = np.nanpercentile(seg, LO_Q)
            hi[j:k] = np.nanpercentile(seg, HI_Q)
        j += step
    return (pd.Series(lo, index=v.index).ffill(), pd.Series(hi, index=v.index).ffill())


def make_anchors(raw, mode, sl=None):
    """返回 {key: (lo, hi)}，lo/hi 可以是标量或 Series。"""
    src = raw if sl is None else raw.loc[sl]
    dd = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    src = src.copy(); src["dd250"] = dd
    if mode == "expand":
        return {k: expand_anchor(src[k]) for k in KEYS}
    return {k: (float(np.nanpercentile(src[k].to_numpy(float), LO_Q)),
                float(np.nanpercentile(src[k].to_numpy(float), HI_Q))) for k in KEYS}


def pmm(v, a, k):
    lo, hi = a[k]
    d = hi - lo
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


def entries(s, c, thr=THR):
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
        print(f"  {label:<28} 无样本"); return
    o = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
    nt = sum(1 for x in rows if x[3] >= -TOL)
    print(f"  {label:<28} 样本 {len(rows):>3}  命中 {o}/{len(rows)} = {o/len(rows)*100:5.1f}%   "
          f"未被套 {nt}/{len(rows)} = {nt/len(rows)*100:5.1f}%   "
          f"均T+20 {np.mean([x[2] for x in rows])*100:+6.2f}%   "
          f"最差 {min(x[3] for x in rows)*100:+6.2f}%   威尔逊下界 {wilson(o,len(rows))*100:5.1f}%")


print("载入长历史面板 ...", flush=True)
PAN = longhist.build()
for nm, raw in PAN.items():
    print(f"  {nm:<8} {len(raw)} 行  {raw.index.min().date()} ~ {raw.index.max().date()}", flush=True)

MODES = [("A 全样本固定锚(非因果)", "full", None),
         ("B 扩展因果锚(可实盘)", "expand", None)]

print("\n" + "=" * 132)
print(f"长历史锚点检验   规则：昨 ≤ {THR:.0f} 且今日回升 → 今日收盘买（T+{H}，容差 {TOL:.0%}，冷却 {COOL} 交易日）")
print("=" * 132)

for label, mode, _ in MODES:
    print(f"\n【{label}】")
    allrows, winrows = [], []
    for nm, raw in PAN.items():
        r = raw.copy()
        a = make_anchors(r, mode)
        sc = score(r, a)
        ev = entries(sc, r["close"])
        allrows += [(nm,) + e for e in ev]
        winrows += [(nm,) + e for e in ev if e[0] >= W2]
        print(f"    {nm:<8} 信号 {len(ev):>3} 个   分值区间 "
              f"{np.nanmin(sc.to_numpy()):.0f}~{np.nanmax(sc.to_numpy()):.0f}", flush=True)
    stat(allrows, f"全程 {W1.date()}~今")
    stat(winrows, f"交付窗口 {W2.date()}~今")

# ---- 阈值敏感性：在因果锚下扫一遍 ----
print("\n【阈值敏感性 · 扩展因果锚 · 全程】")
a_all = {nm: make_anchors(raw, "expand") for nm, raw in PAN.items()}
SC = {nm: score(raw, a_all[nm]) for nm, raw in PAN.items()}
for t in [-2, -4, -6, -8, -10, -12, -15]:
    rows = []
    for nm, raw in PAN.items():
        rows += [(nm,) + e for e in entries(SC[nm], raw["close"], thr=float(t))]
    stat(rows, f"阈值 {t:>3}")

print("\n【阈值敏感性 · 扩展因果锚 · 交付窗口 2023-09-20~今】")
for t in [-2, -4, -6, -8, -10, -12, -15]:
    rows = []
    for nm, raw in PAN.items():
        rows += [(nm,) + e for e in entries(SC[nm], raw["close"], thr=float(t)) if e[0] >= W2]
    stat(rows, f"阈值 {t:>3}")

# ---- 分年度（因果锚）----
print("\n【分年度 · 扩展因果锚 · 阈值 −6】")
rows = []
for nm, raw in PAN.items():
    rows += [(nm,) + e for e in entries(SC[nm], raw["close"])]
for y in sorted({x[1].year for x in rows}):
    sub = [x for x in rows if x[1].year == y]
    stat(sub, f"{y} 年")
