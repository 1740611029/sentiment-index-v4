"""生产面板体检（改成因果锚之后）：分值区间、阈值重扫、历史稳定性复验。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, store, model as M, data, factors

H, TOL = store.H, store.TOL
COOL = store.COOL_TRADING
START = pd.Timestamp(C.BACKTEST_START)


def entries(s, c, thr, cool=COOL):
    dates = list(s.index)
    s = s.to_numpy(float); cl = c.to_numpy(float)
    out = []; last = -10 ** 9; n = len(s)
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < cool: continue
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
        print(f"  {label:<22} 无样本"); return
    o = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
    nt = sum(1 for x in rows if x[3] >= -TOL)
    print(f"  {label:<22} n={len(rows):>3}  命中 {o}/{len(rows):<3} {o/len(rows)*100:5.1f}%   "
          f"未被套 {nt/len(rows)*100:5.1f}%   均T+20 {np.mean([x[2] for x in rows])*100:+6.2f}%   "
          f"最差 {min(x[3] for x in rows)*100:+7.2f}%   下界 {wilson(o,len(rows))*100:5.1f}%")


P = store.load()
print("=" * 118)
print("生产面板（因果锚）· 分值区间")
print("=" * 118)
for b in C.BOARD_ORDER:
    p = P[b]
    sc = p["score"].dropna()
    nan = p["score"].isna().sum()
    print(f"  {C.BOARDS[b]['name']:<8} {len(p):>4} 行  {str(p.index.min().date())} ~ {str(p.index.max().date())}  "
          f"分值 {sc.min():6.1f} ~ {sc.max():6.1f}   空值 {nan}")

print("\n" + "=" * 118)
print("阈值重扫（因果锚新刻度上，−6 是旧刻度调出来的，必须重扫）")
print("=" * 118)
for t in [5, 0, -3, -6, -9, -12, -15, -20]:
    rows = []
    for b in C.BOARD_ORDER:
        rows += [(b,) + e for e in entries(P[b]["score"], P[b]["close"], float(t))]
    line(rows, f"阈值 {t:>3}")

print("\n【阈值 −6 分板块明细】")
for b in C.BOARD_ORDER:
    ev = entries(P[b]["score"], P[b]["close"], -6.0)
    line([(b,) + e for e in ev], C.BOARDS[b]["name"])
    for d, r, m, s0 in ev:
        print(f"        {d}  分值 {s0:6.1f}  T+20 {r*100:+6.2f}%  最深 {m*100:+6.2f}%")

print("\n" + "=" * 118)
print("历史稳定性复验：新增数据是否还会改写已发布的分值")
print("=" * 118)
T1 = pd.Timestamp("2026-06-30")
for b in ["SH", "STAR", "CHINEXT", "CSI2000", "CSI1000", "HS300"]:
    full = M.build(b)
    idx = data.load_index(b).set_index("date")
    idx1 = idx[idx.index <= T1]
    raw1 = factors.build_board_raw(b).copy()
    raw1 = raw1[raw1.index <= T1]
    # 用截断数据重算：直接复用 build 的逻辑但输入已截断
    lo_q, hi_q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
    lc, hc = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
    num = pd.Series(0.0, index=raw1.index); den = 0.0
    for k, w in M.L_WEIGHTS.items():
        num = num + M._pct_map(raw1[k], lo_q, hi_q, lc, hc) * w; den += w
    raw1["L"] = num / den
    raw1["T"] = (M._pct_map(raw1["bias"], lo_q, hi_q, lc, hc) * M.T_WEIGHTS["bias"]
                 + M._pct_map(raw1["ret20"], lo_q, hi_q, lc, hc) * M.T_WEIGHTS["ret20"]
                 + M._pct_map(raw1["rsi"], lo_q, hi_q, lc, hc) * M.T_WEIGHTS["rsi"]
                 + raw1["amt_pct"] * 100.0 * M.T_WEIGHTS["amt"])
    raw1["U"] = M.W_L * raw1["L"] + (1 - M.W_L) * raw1["T"]
    amt_q = raw1["amt_pct"].astype(float)
    raw1["cf_b"] = ((((amt_q - M.CF_AMT_LO) / (1.0 - M.CF_AMT_LO)).clip(0, 1).fillna(0.0))
                    * ((M.CF_L_HI - raw1["L"]) / M.CF_L_HI).clip(0, 1).fillna(0.0)).fillna(0.0)
    raw1["dd250"] = raw1["close"] / raw1["close"].rolling(250, min_periods=60).max() - 1.0
    f_disp = (100.0 - M._pct_map(raw1["disp"], lo_q, hi_q, lc, hc)).fillna(0.0)
    f_dd = (100.0 - M._pct_map(raw1["dd250"], lo_q, hi_q, lc, hc)).fillna(0.0)
    raw1["cf_g"] = ((((f_dd - M.GR_DD_A) / M.GR_DD_W).clip(0, 1))
                    * ((f_disp - M.GR_DP_A) / M.GR_DP_W).clip(0, 1)).fillna(0.0)
    s1 = (raw1["U"] - M.K_B * raw1["cf_b"] - M.K_G * raw1["cf_g"]).clip(M.CLIP_LO, M.CLIP_HI)
    s2 = full["score"]
    j = pd.concat([s1.rename("a"), s2.rename("b")], axis=1).dropna()
    j = j[j.index <= T1]
    d = (j["a"] - j["b"]).abs()
    print(f"  {C.BOARDS[b]['name']:<8} 重叠 {len(j):>4} 天   完全相同 {(d < 1e-9).mean()*100:6.1f}%   "
          f"均绝对差 {d.mean():.4f}   最大差 {d.max():.4f}")
