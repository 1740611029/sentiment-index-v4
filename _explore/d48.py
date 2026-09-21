"""量化前视偏差：生产面板（近 3 年）上，全样本锚 vs 因果锚 的差距。

生产的 _pct_map 用「整个面板」的 p2/p98 作锚点，也就是 2023-09 的分值
依赖 2026-09 的数据 —— 这是前视函数（look-ahead）。后果有两个：
  1) 历史回测的 95% 是**事后**算出来的，实盘做不到
  2) 每天新增数据，整条历史曲线会被重新缩放，昨天的 −10 明天可能变 −3

本脚本在同一份 3 年面板上对比三种锚点口径：
  A 全样本锚（生产现状，前视）
  B 扩展因果锚（只看 t 之前）
  C trailing 500 日因果锚（只看 t 之前 500 日）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors
from senti import model as M

H, TOL, COOL, THR = 20, 0.03, 20, -6.0
LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
LO_Q, HI_Q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
KEYS = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20", "disp", "dd250"]

BOARDS = ["SH", "STAR", "CHINEXT", "CSI2000", "CSI1000", "HS300"]


def mkeys(raw):
    src = raw.copy()
    src["dd250"] = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    return src


def anchors_full(src):
    return {k: (float(np.nanpercentile(src[k].to_numpy(float), LO_Q)),
                float(np.nanpercentile(src[k].to_numpy(float), HI_Q))) for k in KEYS}


def anchors_expand(src, min_n=120, step=10):
    out = {}
    for k in KEYS:
        x = src[k].to_numpy(float); n = len(x)
        lo = np.full(n, np.nan); hi = np.full(n, np.nan)
        j = min_n
        while j < n:
            seg = x[:j]; seg = seg[~np.isnan(seg)]
            if len(seg) >= min_n:
                e = min(j + step, n)
                lo[j:e] = np.nanpercentile(seg, LO_Q); hi[j:e] = np.nanpercentile(seg, HI_Q)
            j += step
        out[k] = (pd.Series(lo, index=src.index).ffill(), pd.Series(hi, index=src.index).ffill())
    return out


def anchors_trail(src, win=500):
    return {k: (src[k].rolling(win, min_periods=win).quantile(LO_Q / 100.0).shift(1),
                src[k].rolling(win, min_periods=win).quantile(HI_Q / 100.0).shift(1)) for k in KEYS}


def pmm(v, a, k):
    lo, hi = a[k]; d = hi - lo
    if isinstance(d, pd.Series):
        d = d.replace(0.0, np.nan)
    elif d == 0:
        d = np.nan
    return (100.0 * (v - lo) / d).clip(LC, HC)


def score(src, a):
    num = pd.Series(0.0, index=src.index); den = 0.0
    for k, w in M.L_WEIGHTS.items():
        num = num + pmm(src[k], a, k) * w; den += w
    L = num / den
    T = (pmm(src["bias"], a, "bias") * M.T_WEIGHTS["bias"]
         + pmm(src["ret20"], a, "ret20") * M.T_WEIGHTS["ret20"]
         + pmm(src["rsi"], a, "rsi") * M.T_WEIGHTS["rsi"]
         + src["amt_pct"] * 100.0 * M.T_WEIGHTS["amt"])
    U = M.W_L * L + (1 - M.W_L) * T
    cfb = (((src["amt_pct"].astype(float) - M.CF_AMT_LO) / (1.0 - M.CF_AMT_LO)).clip(0, 1).fillna(0.0)
           * ((M.CF_L_HI - L) / M.CF_L_HI).clip(0, 1).fillna(0.0)).fillna(0.0)
    f_disp = (100.0 - pmm(src["disp"], a, "disp")).fillna(0.0)
    f_dd = (100.0 - pmm(src["dd250"], a, "dd250")).fillna(0.0)
    cfg = (((f_dd - M.GR_DD_A) / M.GR_DD_W).clip(0, 1)
           * ((f_disp - M.GR_DP_A) / M.GR_DP_W).clip(0, 1)).fillna(0.0)
    return (U - M.K_B * cfb - M.K_G * cfg).clip(M.CLIP_LO, M.CLIP_HI)


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


print("构建生产面板 ...", flush=True)
SRC = {}
for k in BOARDS:
    try:
        SRC[k] = mkeys(factors.build_board_raw(k))
        print(f"  {k:<9} {len(SRC[k])} 行  {SRC[k].index.min().date()} ~ {SRC[k].index.max().date()}", flush=True)
    except Exception as e:
        print(f"  {k} 失败: {e}")

print("\n" + "=" * 126)
print("前视偏差量化：同一份 3 年面板，不同锚点口径（阈值 −6）")
print("=" * 126)
for tag, fn in (("A 全样本锚（生产现状·前视）", anchors_full),
                ("B 扩展因果锚", anchors_expand),
                ("C trailing 500 日因果锚", anchors_trail)):
    rows = []
    for k, src in SRC.items():
        a = fn(src)
        sc = score(src, a)
        rows += [(k,) + e for e in entries(sc, src["close"])]
    line(rows, tag)

print("\n【B 扩展因果锚 · 阈值扫描】")
A = {k: anchors_expand(s) for k, s in SRC.items()}
SCB = {k: score(s, A[k]) for k, s in SRC.items()}
for t in [0, -3, -6, -9, -12, -15]:
    rows = []
    for k in SRC:
        rows += [(k,) + e for e in entries(SCB[k], SRC[k]["close"], thr=float(t))]
    line(rows, f"阈值 {t:>3}")

print("\n【分值区间对比（同一板块 HS300）】")
for tag, fn in (("A 全样本", anchors_full), ("B 扩展因果", anchors_expand)):
    sc = score(SRC["HS300"], fn(SRC["HS300"]))
    print(f"  {tag:<12} 分值 {np.nanmin(sc.to_numpy()):.1f} ~ {np.nanmax(sc.to_numpy()):.1f}   "
          f"2024-02-05 = {sc.get(pd.Timestamp('2024-02-05'), float('nan')):.1f}   "
          f"2024-09-30 = {sc.get(pd.Timestamp('2024-09-30'), float('nan')):.1f}")
