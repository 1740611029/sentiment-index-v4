"""顶部重做：用同样的工具箱（新信息源 + 横截面广度）救顶部。

现状：顶部信号（昨日 ≤100 且今日 >100）实测只有 2/9 = 22%，基本无效。
      2024-10 那批高分信号之后 20 天平均还涨 8.7% —— 是「踏空」，不是「见顶」。
      换过「从 100 上方回落」口径，同样 22%，不是口径问题。

这次的假设（与底部严格对称）：
  底部要「杠杆已出清」→ 融资余额分位低、破净率高
  顶部要「杠杆已挤满」→ 融资余额分位高、破净率极低
  2015 年 6 月那个真正的顶，融资余额 2.1 万亿是历史极值 —— 这正是分位能吃到的信号。

另外加一个横截面维度：top_reso = 当日有多少个板块分值 > 100（全场狂热广度）。

检验顺序不变：生产好看不算，必须过 2015（那一年才有真正的顶）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C
from senti import model as M
from senti import factors as F
import longhist

H, H2, COOL, TOL = 20, 60, 20, 0.03
TOP = 100.0
W2 = pd.Timestamp("2023-09-20")
LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
LO_Q, HI_Q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
EXCLUDE = {"CSI2000"}
KEYS = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20", "disp", "dd250"]
WIN, MINP = C.MODEL["anchor_window"], C.MODEL["anchor_min"]
CLIP_LO, CLIP_HI = -30.0, 140.0
MKT_DIR = os.path.join(C.DATA_DIR, "cache", "market")


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


def top_signals(panels):
    """顶部：昨日 ≤100 且今日 >100。返回 DataFrame。"""
    S = pd.DataFrame({nm: p["score"] for nm, p in panels.items()}).dropna(how="all")
    CL = pd.DataFrame({nm: p["close"] for nm, p in panels.items()}).reindex(S.index)
    ZH = (S > TOP)
    rows = []
    for nm in panels:
        v = S[nm].to_numpy(float); cl = CL[nm].to_numpy(float)
        n = len(v); last = -10 ** 9
        for i in range(1, n - H2):
            if np.isnan(v[i]) or np.isnan(v[i - 1]): continue
            if i - last < COOL: continue
            if v[i - 1] > TOP or v[i] <= TOP: continue     # 首次突破 100
            last = i
            c0 = cl[i]
            seg = cl[i + 1:i + H + 1]; seg2 = cl[i + 1:i + H2 + 1]
            rows.append({"board": nm, "date": S.index[i], "score": v[i],
                         "ret20": seg[-1] / c0 - 1, "run20": seg.max() / c0 - 1,
                         "ret60": seg2[-1] / c0 - 1, "run60": seg2.max() / c0 - 1,
                         # 顶部命中 = 之后确实跌了，且没怎么踏空
                         "ok20": bool(seg[-1] / c0 - 1 < 0 and seg.max() / c0 - 1 <= TOL),
                         "down20": bool(seg[-1] / c0 - 1 < 0),
                         "down60": bool(seg2[-1] / c0 - 1 < 0),
                         "top_reso": int(ZH.iloc[i].sum())})
    return pd.DataFrame(rows)


mg = pd.read_parquet(os.path.join(MKT_DIR, "margin.parquet")).set_index("date")["margin"].astype(float)
bn = pd.read_parquet(os.path.join(MKT_DIR, "bna.parquet")).set_index("date")["bna"].astype(float)
mg = mg[~mg.index.duplicated(keep="last")].sort_index().shift(1)
bn = bn[~bn.index.duplicated(keep="last")].sort_index().shift(1)
D = pd.DataFrame({"mg": mg, "bna": bn}).sort_index()
r250 = lambda x: x.rolling(250, min_periods=120)
D["mg_pct"] = r250(D["mg"]).rank(pct=True) * 100
D["mg_chg20"] = D["mg"] / D["mg"].shift(20) - 1
D["mg_chg60"] = D["mg"] / D["mg"].shift(60) - 1
D["bna_pct"] = r250(D["bna"]).rank(pct=True) * 100


def attach(df):
    out = df.copy()
    for f in ("mg_pct", "mg_chg20", "mg_chg60", "bna_pct"):
        out[f] = D[f].reindex(pd.DatetimeIndex(out["date"]), method="ffill").to_numpy()
    return out


def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def tline(df, label):
    if len(df) == 0:
        print(f"  {label:<34} 无样本"); return
    ok = int(df["ok20"].sum()); dn = int(df["down20"].sum()); n = len(df)
    print(f"  {label:<34} n={n:>3}  命中 {ok}/{n:<3} {ok/n*100:5.1f}%  下界 {wil(ok,n)*100:5.1f}%   "
          f"T+20下跌 {dn}/{n} {dn/n*100:5.1f}%   均T+20 {df['ret20'].mean()*100:+6.2f}%   "
          f"最大踏空 {df['run20'].max()*100:+6.2f}%   T+60下跌 {int(df['down60'].sum())}/{n}")


RAW = {b: F.build_board_raw(b) for b in C.BOARD_ORDER}
DP = attach(top_signals({b: pd.DataFrame({"score": score(RAW[b], make_anchors(RAW[b], "trail")),
                                          "close": RAW[b]["close"]}) for b in C.BOARD_ORDER}))
PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
LT = attach(top_signals({nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "trail")),
                                           "close": raw["close"]}) for nm, raw in PAN.items()}))
LE = attach(top_signals({nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "expand")),
                                           "close": raw["close"]}) for nm, raw in PAN.items()}))

print("=" * 136)
print("① 顶部信号现状（首破 100）")
print("=" * 136)
for nm, d in (("生产 6 板块", DP), ("长历史 8 宽基 trail", LT), ("长历史 8 宽基 expand", LE)):
    tline(d, f"　{nm}")
    tline(d[d["date"] >= W2], f"　{nm} · 交付窗口")

print("\n" + "=" * 136)
print("② 分年度（长历史 expand 锚）—— 2015 年才有真正的顶")
print("=" * 136)
years = sorted(LE["date"].dt.year.unique())
print(f"  {'':<16}" + "".join(f"{y:>9}" for y in years) + f"{'合计':>12}")
for lab, col in (("T+20 命中", "ok20"), ("T+20 下跌", "down20"), ("T+60 下跌", "down60")):
    cells = ""
    for y in years:
        s = LE[LE["date"].dt.year == y]
        cells += (f"{int(s[col].sum())}/{len(s)}" if len(s) else "-").rjust(9)
    cells += f"{int(LE[col].sum())}/{len(LE)}".rjust(12)
    print(f"  {lab:<16}" + cells)

print("\n" + "=" * 136)
print("③ 顶部信号逐条明细（长历史 expand，按年份）")
print("=" * 136)
t = LE.copy(); t["date"] = t["date"].dt.strftime("%Y-%m-%d")
print(t[["board", "date", "score", "ret20", "run20", "ret60", "mg_pct", "mg_chg20",
         "bna_pct", "top_reso", "ok20"]].to_string(index=False,
      formatters={"score": lambda x: f"{x:6.1f}", "ret20": lambda x: f"{x*100:+6.2f}%",
                  "run20": lambda x: f"{x*100:+6.2f}%", "ret60": lambda x: f"{x*100:+7.2f}%",
                  "mg_pct": lambda x: f"{x:5.1f}", "mg_chg20": lambda x: f"{x*100:+6.1f}%",
                  "bna_pct": lambda x: f"{x:5.1f}"}))

print("\n" + "=" * 136)
print("④ 门槛扫描：顶部也要「杠杆挤满」吗？")
print("=" * 136)
COND = {
    "融资余额分位 >= 70": lambda d: d["mg_pct"] >= 70,
    "融资余额分位 >= 80": lambda d: d["mg_pct"] >= 80,
    "融资余额分位 >= 90": lambda d: d["mg_pct"] >= 90,
    "加杠杆 20日 >= +3%": lambda d: d["mg_chg20"] >= 0.03,
    "加杠杆 20日 >= +5%": lambda d: d["mg_chg20"] >= 0.05,
    "加杠杆 60日 >= +8%": lambda d: d["mg_chg60"] >= 0.08,
    "破净分位 <= 40（估值贵）": lambda d: d["bna_pct"] <= 40,
    "破净分位 <= 25": lambda d: d["bna_pct"] <= 25,
    "狂热广度 top_reso>=2": lambda d: d["top_reso"] >= 2,
    "狂热广度 top_reso>=3": lambda d: d["top_reso"] >= 3,
}
print("　（对照）无过滤")
for nm, d in (("生产", DP), ("长历史trail", LT), ("长历史expand", LE)):
    tline(d, f"　{nm}")
for lab, fn in COND.items():
    print(f"\n──── {lab} ────")
    for nm, d in (("生产", DP), ("长历史trail", LT), ("长历史expand", LE)):
        tline(d[fn(d)], f"　{nm}")

print("\n" + "=" * 136)
print("⑤ 组合（顶部的对称假说：杠杆挤满 + 全场狂热）")
print("=" * 136)
for lab, fn in (("杠杆分位>=80 且 破净分位<=40", lambda d: (d["mg_pct"] >= 80) & (d["bna_pct"] <= 40)),
                ("杠杆分位>=80 且 reso>=2", lambda d: (d["mg_pct"] >= 80) & (d["top_reso"] >= 2)),
                ("加杠杆>=3% 且 杠杆分位>=80", lambda d: (d["mg_chg20"] >= 0.03) & (d["mg_pct"] >= 80)),
                ("三项全满足", lambda d: (d["mg_pct"] >= 80) & (d["bna_pct"] <= 40) & (d["top_reso"] >= 2))):
    print(f"\n──── {lab} ────")
    for nm, d in (("生产", DP), ("长历史trail", LT), ("长历史expand", LE)):
        tline(d[fn(d)], f"　{nm}")
