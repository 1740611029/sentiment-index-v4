"""把「放量」从「250 日绝对分位」改成「相对本轮下跌的相对放量」。

动机：250 日分位是跨年绝对比较。2015 年整个市场成交量都在天上，
amt_pct > 0.6 稀松平常 → 恐慌放量信号满天飞 → 2015 年只有 4/12。
而「恐慌高潮」的正确定义应当是**相对于最近这段下跌的放量** —— 自适应当前环境，
不受年代影响。

  amt_rel = 成交额 / 过去 20 日平均成交额
  cf_amt  = 斜坡(amt_rel; REL_LO → REL_HI)
  cf_b    = cf_amt × 超跌程度

检验顺序（纪律：生产好看不算，必须过 2015-2016）：
  ① 生产 6 板块（trailing-750 因果锚）
  ② 长历史 8 宽基 · trailing-750
  ③ 长历史 8 宽基 · 扩展因果锚（含 2015-2016）
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


def score(raw, a, amt_rel=None, rel_lo=None, rel_hi=None):
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
    if amt_rel is None:      # 现状：250 日分位
        cf_amt = ((raw["amt_pct"].astype(float) - M.CF_AMT_LO) / (1.0 - M.CF_AMT_LO)).clip(0, 1).fillna(0.0)
    else:                    # 新：相对放量
        cf_amt = ((amt_rel - rel_lo) / (rel_hi - rel_lo)).clip(0, 1).fillna(0.0)
    cfb = (cf_amt * cf_low).fillna(0.0)
    return (U - M.K_B * cfb).clip(M.CLIP_LO, M.CLIP_HI)


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


CANDS = [("现状 250日分位", None, None)] + [
    (f"相对放量 {lo}→{hi}", lo, hi) for lo, hi in
    [(1.0, 2.0), (1.2, 2.0), (1.5, 2.5), (1.5, 3.0), (2.0, 3.0)]]

# ---------------- ① 生产 6 板块 ----------------
from senti import data as D
print("=" * 126)
print("① 生产 6 板块（trailing-750 因果锚，阈值 0）")
print("=" * 126)
PROD = {}
for b in C.BOARD_ORDER:
    idx = D.load_index(b).set_index("date")
    amt = idx["amount"].astype(float)
    PROD[b] = (None, amt)   # raw 由 model.build 生成，见下
from senti import factors as F
RAW = {b: F.build_board_raw(b) for b in C.BOARD_ORDER}
for tag, lo, hi in CANDS:
    rows = []
    for b in C.BOARD_ORDER:
        raw = RAW[b]
        a = make_anchors(raw, "trail")
        idx = D.load_index(b).set_index("date")
        amt = idx["amount"].astype(float).reindex(raw.index)
        if lo is None:
            sc = score(raw, a)
        else:
            rel = amt / amt.rolling(20, min_periods=10).mean()
            sc = score(raw, a, rel.reindex(raw.index), lo, hi)
        sc = sc[sc.index >= W2]
        cl = raw["close"][raw.index >= W2]
        rows += [(b,) + e for e in entries(sc, cl)]
    line(rows, tag)

# ---------------- ②③ 长历史 ----------------
PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
NAME2FILE = {k: v[1] for k, v in longhist.BOARDS.items()}   # board_key -> 指数文件名
NAME2FILE.update({v[0]: v[1] for v in longhist.BOARDS.values()})   # 兼容「板块名」键
IH = os.path.join(C.DATA_DIR, "cache", "index_hist")
AMT = {}
for nm in PAN:
    x = pd.read_parquet(os.path.join(IH, f"{NAME2FILE[nm]}.parquet"))
    x["date"] = pd.to_datetime(x["date"])
    AMT[nm] = x.set_index("date")["amount"].astype(float)

for mode, tag in (("trail", "trailing-750"), ("expand", "扩展因果锚（含2015-2016）")):
    print("\n" + "=" * 126)
    print(f"{'②' if mode == 'trail' else '③'} 长历史 8 宽基 · {tag} · 阈值 0")
    print("=" * 126)
    A = {nm: make_anchors(raw, mode) for nm, raw in PAN.items()}
    RES = {}
    for lab, lo, hi in CANDS:
        rows = []
        for nm, raw in PAN.items():
            if lo is None:
                sc = score(raw, A[nm])
            else:
                amt = AMT[nm].reindex(raw.index)
                rel = amt / amt.rolling(20, min_periods=10).mean()
                sc = score(raw, A[nm], rel, lo, hi)
            rows += [(nm,) + e for e in entries(sc, raw["close"])]
        RES[lab] = rows
        line(rows, lab)
        line([x for x in rows if x[1] >= W2], lab + " · 交付窗口")
    print("\n  【分年度】")
    yrs = sorted({x[1].year for r in RES.values() for x in r})
    print(f"    {'放量口径':<20}" + "".join(f"{y:>8}" for y in yrs))
    for lab, rows in RES.items():
        cells = ""
        for y in yrs:
            sub = [x for x in rows if x[1].year == y]
            cells += (f"{sum(1 for x in sub if x[2] > 0 and x[3] >= -TOL)}/{len(sub)}" if sub else "-").rjust(8)
        print(f"    {lab:<20}" + cells)
