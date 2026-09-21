"""共振的正确定义：看「恐慌那天」，不是「信号那天」。

d66 的 reso 定义在信号日（score 已经反弹的那天）取快照，这有个明显缺陷：
  2025-04-08 六个板块全部触发、全部大赚（+5.35% ~ +16.87%），
  但因为当天所有板块都已反弹到 0 以上，reso = 0，被过滤条件排除了 —— 把最好的机会扔掉了。

恐慌发生在**前一天**（信号规则是「昨日 ≤ 0 且今日 > 昨日」），所以共振度应该在昨日取。

d67 同时扫多个定义：
  reso_d0  信号日当天 ≤0 的板块数（d66 用的，作对照）
  reso_d1  昨日 ≤0 的板块数           ← 主要候选
  reso_mx  max(d0, d1)
  lowN5    过去 5 个交易日内曾 ≤0 的不同板块数
  lowN10   过去 10 个交易日
  sigN5    过去 5 个交易日内曾触发底部信号的不同板块数
  gap_d1   自己昨日 score − 昨日其他板块中位数
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C
from senti import model as M
from senti import factors as F
import longhist

H, H2, COOL, THR, TOL = 20, 60, 20, 0.0, 0.03
W2 = pd.Timestamp("2023-09-20")
LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
LO_Q, HI_Q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
EXCLUDE = {"CSI2000"}
KEYS = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20", "disp", "dd250"]
WIN, MINP = C.MODEL["anchor_window"], C.MODEL["anchor_min"]
CLIP_LO, CLIP_HI = -30.0, 140.0


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
    cf_low = ((M.CF_L_HI - L) / M.CF_L_HI).clip(0, 1).fillna(0.0)
    cf_amt = ((raw["amt_pct"].astype(float) - M.CF_AMT_LO) / (1.0 - M.CF_AMT_LO)).clip(0, 1).fillna(0.0)
    cfb = (cf_amt * cf_low).fillna(0.0)
    return (U - M.K_B * cfb).clip(CLIP_LO, CLIP_HI)


def signals(panels):
    S = pd.DataFrame({nm: p["score"] for nm, p in panels.items()}).dropna(how="all")
    CL = pd.DataFrame({nm: p["close"] for nm, p in panels.items()}).reindex(S.index)
    Z = (S <= THR)
    rows = []
    for nm in panels:
        v = S[nm].to_numpy(float); cl = CL[nm].to_numpy(float)
        n = len(v); last = -10 ** 9
        for i in range(1, n - H2):
            if np.isnan(v[i]) or np.isnan(v[i - 1]): continue
            if i - last < COOL: continue
            if v[i - 1] > THR or v[i] <= v[i - 1]: continue
            last = i
            c0 = cl[i]
            seg = cl[i + 1:i + H + 1]; seg2 = cl[i + 1:i + H2 + 1]
            d0 = Z.iloc[i]; d1 = Z.iloc[i - 1]
            o1 = S.iloc[i - 1].drop(nm).dropna()
            rows.append({
                "board": nm, "date": S.index[i],
                "ret20": seg[-1] / c0 - 1, "mdd20": seg.min() / c0 - 1,
                "ret60": seg2[-1] / c0 - 1,
                "ok20": bool(seg[-1] / c0 - 1 > 0 and seg.min() / c0 - 1 >= -TOL),
                "reso_d0": int(d0.sum()), "reso_d1": int(d1.sum()),
                "reso_mx": int(max(d0.sum(), d1.sum())),
                "lowN5": int(Z.iloc[max(0, i - 4):i + 1].any().sum()),
                "lowN10": int(Z.iloc[max(0, i - 9):i + 1].any().sum()),
                "gap_d1": float(v[i - 1] - o1.median()) if len(o1) else np.nan,
                "ntot": int(d0.notna().sum()),
            })
    return pd.DataFrame(rows)


def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def line(df, label):
    if len(df) == 0:
        print(f"  {label:<34} 无样本"); return
    ok = int(df["ok20"].sum()); n = len(df)
    print(f"  {label:<34} n={n:>3}  命中 {ok}/{n:<3} {ok/n*100:5.1f}%  下界 {wil(ok,n)*100:5.1f}%   "
          f"均T+20 {df['ret20'].mean()*100:+6.2f}%  最差浮亏 {df['mdd20'].min()*100:+7.2f}%   "
          f"T+60 {(df['ret60']>0).sum()}/{n}")


RAW = {b: F.build_board_raw(b) for b in C.BOARD_ORDER}
DP = signals({b: pd.DataFrame({"score": score(RAW[b], make_anchors(RAW[b], "trail")),
                               "close": RAW[b]["close"]}) for b in C.BOARD_ORDER})
PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
LT = signals({nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "trail")), "close": raw["close"]})
              for nm, raw in PAN.items()})
LE = signals({nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "expand")), "close": raw["close"]})
              for nm, raw in PAN.items()})

print("=" * 136)
print("① 生产 6 板块 (n=33) · 共振定义扫描")
print("=" * 136)
line(DP, "无过滤（对照）")
for col in ("reso_d0", "reso_d1", "reso_mx", "lowN5", "lowN10"):
    for x in range(1, 7):
        line(DP[DP[col] >= x], f"{col} >= {x}")

print("\n" + "=" * 136)
print("② 长历史 8 宽基 · trailing-750 (n=77)")
print("=" * 136)
line(LT, "无过滤（对照）")
for col in ("reso_d0", "reso_d1", "reso_mx", "lowN5", "lowN10"):
    for x in range(1, 9):
        sub = LT[LT[col] >= x]
        if len(sub): line(sub, f"{col} >= {x}")

print("\n" + "=" * 136)
print("③ 长历史 · 扩展因果锚（含 2015-2016, n=35）")
print("=" * 136)
line(LE, "无过滤（对照）")
for col in ("reso_d1", "reso_mx", "lowN5"):
    for x in range(1, 9):
        sub = LE[LE[col] >= x]
        if len(sub): line(sub, f"{col} >= {x}")

print("\n" + "=" * 136)
print("④ 分年度（trailing-750，H=20 命中 / T+60 盈利）")
print("=" * 136)
for col, x in (("reso_d0", 2), ("reso_d1", 2), ("reso_d1", 3), ("reso_mx", 2), ("lowN5", 3), ("lowN10", 4)):
    sub = LT[LT[col] >= x]
    yrs = sorted(LT["date"].dt.year.unique())
    print(f"\n  {col} >= {x}   合计 n={len(sub)}")
    print(f"    {'年份':<8}" + "".join(f"{y:>10}" for y in yrs))
    a = ""
    for y in yrs:
        s = sub[sub["date"].dt.year == y]
        a += (f"{int(s['ok20'].sum())}/{len(s)}" if len(s) else "-").rjust(10)
    print(f"    {'H20命中':<8}" + a)
    b = ""
    for y in yrs:
        s = sub[sub["date"].dt.year == y]
        b += (f"{int((s['ret60']>0).sum())}/{len(s)}" if len(s) else "-").rjust(10)
    print(f"    {'T60盈利':<8}" + b)

print("\n" + "=" * 136)
print("⑤ 交付窗口（2023-09-20 起）· 生产 6 板块")
print("=" * 136)
DW = DP[DP["date"] >= W2]
line(DW, "无过滤")
for col, x in (("reso_d0", 2), ("reso_d1", 2), ("reso_d1", 3), ("reso_mx", 2), ("lowN5", 3)):
    line(DW[DW[col] >= x], f"{col} >= {x}")

print("\n" + "=" * 136)
print("⑥ 生产 6 板块 · 33 次信号明细（按 reso_d1 排序）")
print("=" * 136)
t = DP.copy(); t["date"] = t["date"].dt.strftime("%Y-%m-%d")
t["w20"] = t["ok20"].map({True: "√", False: "×"}); t["w60"] = (t["ret60"] > 0).map({True: "√", False: "×"})
print(t.sort_values(["reso_d1", "date"])[["board", "date", "reso_d0", "reso_d1", "lowN5", "lowN10",
                                           "gap_d1", "ret20", "mdd20", "ret60", "w20", "w60"]].to_string(index=False,
      formatters={"ret20": lambda x: f"{x*100:+6.2f}%", "mdd20": lambda x: f"{x*100:+6.2f}%",
                  "ret60": lambda x: f"{x*100:+7.2f}%", "gap_d1": lambda x: f"{x:+6.1f}"}))
