"""锚点窗口长度扫描：从「3年样本内锚」到「13年全样本锚」之间找稳定区。

现状两端都不行：
  3 年样本内锚  —— 命中率高但锚点随样本漂移（726天→803天，95%→67%）
  13 年全样本锚 —— 锚点稳定但刻度被 2015 拉到极端，近 3 年几乎不出信号

中间方案：trailing W 日滚动分位（因果，只看过去）。
  W 够长 → p2/p98 估计稳定；W 有限 → 能跟随市场结构变化。
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
LO_Q, HI_Q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
EXCLUDE = {"CSI2000"}
KEYS = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20", "disp", "dd250"]


def make_anchors(raw, win):
    """trailing win 日滚动分位；shift(1) 保证严格因果。"""
    src = raw.copy()
    src["dd250"] = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    if win is None:      # 全样本固定锚（非因果，上界参考）
        return {k: (float(np.nanpercentile(src[k].to_numpy(float), LO_Q)),
                    float(np.nanpercentile(src[k].to_numpy(float), HI_Q))) for k in KEYS}
    return {k: (src[k].rolling(win, min_periods=win).quantile(LO_Q / 100.0).shift(1),
                src[k].rolling(win, min_periods=win).quantile(HI_Q / 100.0).shift(1)) for k in KEYS}


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

print("\n" + "=" * 128)
print("锚点窗口扫描（阈值 −6 · 全程 2015-01 ~ 今）")
print("=" * 128)
RES = {}
for win in [500, 750, 1000, 1250, 1500, 2000, 2500, None]:
    tag = "全样本" if win is None else f"{win}日"
    SC = {nm: score(raw, make_anchors(raw, win)) for nm, raw in PAN.items()}
    rows = []
    for nm, raw in PAN.items():
        rows += [(nm,) + e for e in entries(SC[nm], raw["close"], -6.0)]
    RES[tag] = rows
    line(rows, f"trailing {tag}")

print("\n" + "=" * 128)
print("分年度（命中/样本）—— 跨牛熊稳定性")
print("=" * 128)
yrs = sorted({x[1].year for r in RES.values() for x in r})
print(f"  {'锚点窗口':<14}" + "".join(f"{y:>8}" for y in yrs) + f"{'交付窗口':>10}")
for tag, rows in RES.items():
    cells = ""
    for y in yrs:
        sub = [x for x in rows if x[1].year == y]
        cells += (f"{sum(1 for x in sub if x[2] > 0 and x[3] >= -TOL)}/{len(sub)}" if sub else "-").rjust(8)
    w = [x for x in rows if x[1] >= W2]
    cells += (f"{sum(1 for x in w if x[2] > 0 and x[3] >= -TOL)}/{len(w)}" if w else "-").rjust(10)
    print(f"  {tag:<14}" + cells)

print("\n" + "=" * 128)
print("交付窗口细看（2023-09-20 ~ 今）")
print("=" * 128)
for tag, rows in RES.items():
    w = [x for x in rows if x[1] >= W2]
    line(w, f"trailing {tag}")

print("\n【锚点漂移检验：trailing 1250 日下，锚点随时间是否稳定】")
nm = "HS300"
raw = PAN[nm]
a = make_anchors(raw, 1250)
lo, hi = a["b20"]
s = pd.DataFrame({"b20_p2": lo, "b20_p98": hi}).dropna()
for y in range(2016, 2027):
    sub = s[s.index.year == y]
    if len(sub):
        print(f"  {y}  b20 锚点  p2 {sub['b20_p2'].mean():.4f} (±{sub['b20_p2'].std():.4f})   "
              f"p98 {sub['b20_p98'].mean():.4f} (±{sub['b20_p98'].std():.4f})")
