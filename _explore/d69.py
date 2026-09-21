"""新信息源：融资余额 + 破净占比，能不能区分「投降」和「强平」？

动机：现有模型全是价量。价量在「流动性危机」和「真恐慌底」上长得一模一样
（2024-01-23 中证2000 浮亏 −23.61% 就是栽在这里）。
要分开两者，必须引入价量之外的信息：

  融资余额（沪深合计）
    真投降 = 情绪先崩，杠杆随后慢慢退 → 余额下降平缓、且已接近尾声
    强平   = 杠杆先崩，被动平仓       → 余额断崖、且仍在加速
  破净占比
    破净率飙升 = 市场给的价格已经低于净资产 = 估值侧的投降

时序纪律（重要）：两融数据次日早上才公布 → 全部 shift(1)。
破净率依赖季报净资产 + 当日收盘价 → 同样 shift(1)。
不 shift 就是前视，这个坑前面已经踩过一次。

检验顺序（铁律）：生产好看不算，必须过 2015-2016。
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
            rows.append({"board": nm, "date": S.index[i],
                         "ret20": seg[-1] / c0 - 1, "mdd20": seg.min() / c0 - 1,
                         "ret60": seg2[-1] / c0 - 1,
                         "ok20": bool(seg[-1] / c0 - 1 > 0 and seg.min() / c0 - 1 >= -TOL),
                         "reso": int(Z.iloc[i].sum())})
    return pd.DataFrame(rows)


# ---------------- 市场级特征（全部 shift(1)，因果） ----------------
mg = pd.read_parquet(os.path.join(MKT_DIR, "margin.parquet")).set_index("date")["margin"].astype(float)
bn = pd.read_parquet(os.path.join(MKT_DIR, "bna.parquet")).set_index("date")["bna"].astype(float)
mg = mg[~mg.index.duplicated(keep="last")].sort_index()
bn = bn[~bn.index.duplicated(keep="last")].sort_index()
D = pd.DataFrame({"mg": mg, "bna": bn}).sort_index()
D["mg"] = D["mg"].shift(1)      # 次日才公布
D["bna"] = D["bna"].shift(1)
r250 = lambda x: x.rolling(250, min_periods=120)
D["mg_pct"] = r250(D["mg"]).rank(pct=True) * 100
D["mg_chg5"] = D["mg"] / D["mg"].shift(5) - 1
D["mg_chg20"] = D["mg"] / D["mg"].shift(20) - 1
D["mg_chg60"] = D["mg"] / D["mg"].shift(60) - 1
D["mg_pos"] = D["mg"] / r250(D["mg"]).max()
D["mg_acc"] = D["mg_chg5"] - D["mg_chg20"] / 4.0     # 负 = 仍在加速下降
D["bna_pct"] = r250(D["bna"]).rank(pct=True) * 100
D["bna_chg20"] = D["bna"] / D["bna"].shift(20) - 1
FEATS = ["mg_pct", "mg_chg5", "mg_chg20", "mg_chg60", "mg_pos", "mg_acc", "bna", "bna_pct", "bna_chg20"]


def attach(df):
    out = df.copy()
    for f in FEATS:
        out[f] = D[f].reindex(pd.DatetimeIndex(out["date"]), method="ffill").to_numpy()
    return out


def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def line(df, label):
    if len(df) == 0:
        print(f"  {label:<30} 无样本"); return
    ok = int(df["ok20"].sum()); n = len(df)
    print(f"  {label:<30} n={n:>3}  命中 {ok}/{n:<3} {ok/n*100:5.1f}%  下界 {wil(ok,n)*100:5.1f}%   "
          f"均T+20 {df['ret20'].mean()*100:+6.2f}%  最差浮亏 {df['mdd20'].min()*100:+7.2f}%   "
          f"T+60 {(df['ret60']>0).sum()}/{n}")


# ---------------- 面板 ----------------
RAW = {b: F.build_board_raw(b) for b in C.BOARD_ORDER}
DP = attach(signals({b: pd.DataFrame({"score": score(RAW[b], make_anchors(RAW[b], "trail")),
                                      "close": RAW[b]["close"]}) for b in C.BOARD_ORDER}))
PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
LT = attach(signals({nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "trail")),
                                       "close": raw["close"]}) for nm, raw in PAN.items()}))
LE = attach(signals({nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "expand")),
                                       "close": raw["close"]}) for nm, raw in PAN.items()}))

print("=" * 132)
print("① 特征覆盖检查（不能有大片 NaN）")
print("=" * 132)
for nm, d in (("生产", DP), ("长历史trail", LT), ("长历史expand", LE)):
    print(f"  {nm:<14} n={len(d)}  " + "  ".join(f"{f}:{d[f].notna().mean()*100:.0f}%" for f in FEATS))

print("\n" + "=" * 132)
print("② 分离度：命中组 vs 失败组（|均值差| / 合并标准差）—— 越大越有判别力")
print("=" * 132)
for nm, d in (("生产 33", DP), ("长历史trail 77", LT), ("长历史expand 35", LE)):
    a = d[d["ok20"]]; b = d[~d["ok20"]]
    print(f"\n  【{nm}】 命中 {len(a)} / 失败 {len(b)}")
    res = []
    for f in FEATS + ["reso"]:
        x, y = a[f].dropna(), b[f].dropna()
        if len(x) < 2 or len(y) < 2: continue
        sd = np.sqrt(((len(x)-1)*x.var() + (len(y)-1)*y.var()) / (len(x)+len(y)-2))
        res.append((abs(x.mean()-y.mean())/sd if sd > 1e-12 else 0.0, f, x.mean(), y.mean()))
    for s, f, xm, ym in sorted(res, reverse=True):
        print(f"    {f:<12} 分离度 {s:5.2f}   命中组 {xm:+9.4f}   失败组 {ym:+9.4f}")

print("\n" + "=" * 132)
print("③ 门槛扫描（必须同时改善 生产 与 长历史expand(含2015-2016)）")
print("=" * 132)
QS = [0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85]
for f in FEATS:
    pool = pd.concat([DP[f], LE[f]]).dropna()
    if len(pool) < 20: continue
    print(f"\n──── {f} ────")
    base_dp, base_le = DP.dropna(subset=[f]), LE.dropna(subset=[f])
    line(base_dp, "　生产 无过滤"); line(base_le, "　长历史expand 无过滤")
    for q in QS:
        thr = pool.quantile(q)
        for op, lab in (("ge", ">="), ("le", "<=")):
            sd = base_dp[base_dp[f] >= thr] if op == "ge" else base_dp[base_dp[f] <= thr]
            se = base_le[base_le[f] >= thr] if op == "ge" else base_le[base_le[f] <= thr]
            if len(sd) < 3 or len(se) < 3: continue
            line(sd, f"　生产  {f}{lab}{thr:+.4f}")
            line(se, f"　长历史{lab[0]}  {f}{lab}{thr:+.4f}")

print("\n" + "=" * 132)
print("④ 两个失败日的市场读数（2024-01-23 中小盘危机 vs 同日大盘命中）")
print("=" * 132)
sub = DP[DP["date"] >= "2024-01-15"][DP["date"] <= "2024-02-08"] if False else \
      DP[(DP["date"] >= "2024-01-15") & (DP["date"] <= "2024-02-08")]
t = sub.copy(); t["date"] = t["date"].dt.strftime("%Y-%m-%d")
print(t[["board", "date", "ret20", "mdd20", "reso", "mg_chg20", "mg_acc", "mg_pct",
         "bna", "bna_pct", "ok20"]].to_string(index=False,
      formatters={"ret20": lambda x: f"{x*100:+6.2f}%", "mdd20": lambda x: f"{x*100:+6.2f}%",
                  "mg_chg20": lambda x: f"{x*100:+6.2f}%", "mg_acc": lambda x: f"{x*100:+6.2f}%",
                  "mg_pct": lambda x: f"{x:5.1f}", "bna": lambda x: f"{x*100:5.2f}%",
                  "bna_pct": lambda x: f"{x:5.1f}"}))

print("\n" + "=" * 132)
print("⑤ 2015-2016 那段的读数（准入门槛）")
print("=" * 132)
for y in (2015, 2016):
    s = LE[LE["date"].dt.year == y]
    if not len(s): continue
    print(f"  {y} 年 n={len(s)} 命中 {int(s['ok20'].sum())}   "
          f"mg_chg20 均值 {s['mg_chg20'].mean()*100:+.1f}%   mg_acc 均值 {s['mg_acc'].mean()*100:+.2f}%   "
          f"bna_pct 均值 {s['bna_pct'].mean():.1f}")
    hi = s[s["mg_chg20"] <= s["mg_chg20"].median()]
    lo = s[s["mg_chg20"] > s["mg_chg20"].median()]
    line(hi, f"　{y} 去杠杆更快的一半")
    line(lo, f"　{y} 去杠杆更慢的一半")
