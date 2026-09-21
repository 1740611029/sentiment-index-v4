"""横截面共振（d65 的确认版）。

d65 的发现：
  生产 33 次信号里，reso（信号日当天 score ≤ 0 的板块数，含自己）
    reso = 1（只有自己在恐慌） 3 次 → 命中 1/3，其中就包括 2024-01-23 的 CSI2000 和 CSI1000
    reso >= 2（至少还有一个别的板块也在 ≤0） 8 次 → 8/8 = 100%，最差浮亏 −2.43%，T+60 18/18
  长历史 8 宽基 trailing-750：全部 62.3% → reso>=2 为 72.2%，最差浮亏 −25.87% → −8.83%，
    T+60 盈利 18/18 = 100%；2015 年 4/12 → reso>=2 的 4/4。

语义解释得通：
  reso 高 = 多个板块同时极度恐慌 = **系统性投降** → 真底
  reso 低 = 只有自己恐慌 = **局部流动性危机**（板块自身的问题） → 还会跌
  2024-01-23 就是后者：中小盘被砸，大盘没事。

d66 要回答三件事：
  ① 门槛取多少（reso 绝对数 vs frac 比例——两个样本集的板块数不同，必须统一）
  ② 2020 年 reso=2 那 3 次为什么全失败（唯一红灯）
  ③ 这个条件到底能不能写进模型，还是只能当「信号标签」
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
            snap = S.iloc[i]; others = snap.drop(nm).dropna()
            reso = int((snap <= THR).sum())
            rows.append({
                "board": nm, "date": S.index[i],
                "ret20": seg[-1] / c0 - 1, "mdd20": seg.min() / c0 - 1,
                "ret60": seg2[-1] / c0 - 1, "mdd60": seg2.min() / c0 - 1,
                "ok20": bool(seg[-1] / c0 - 1 > 0 and seg.min() / c0 - 1 >= -TOL),
                "reso": reso, "frac": reso / max(1, int(snap.notna().sum())),
                "gap": float(v[i] - others.median()) if len(others) else np.nan,
                "rank": int((snap < v[i]).sum()), "ntot": int(snap.notna().sum()),
            })
    return pd.DataFrame(rows)


def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def line(df, label):
    if len(df) == 0:
        print(f"  {label:<30} 无样本"); return
    ok = int(df["ok20"].sum()); n = len(df)
    print(f"  {label:<30} n={n:>3}  命中 {ok}/{n:<3} {ok/n*100:5.1f}%  下界 {wil(ok,n)*100:5.1f}%   "
          f"均T+20 {df['ret20'].mean()*100:+6.2f}%   最差浮亏 {df['mdd20'].min()*100:+7.2f}%   "
          f"T+60盈利 {(df['ret60']>0).sum()}/{n} {(df['ret60']>0).mean()*100:5.1f}%")


# ---------- 面板 ----------
RAW = {b: F.build_board_raw(b) for b in C.BOARD_ORDER}
DP = signals({b: pd.DataFrame({"score": score(RAW[b], make_anchors(RAW[b], "trail")),
                               "close": RAW[b]["close"]}) for b in C.BOARD_ORDER})
PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
LT = signals({nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "trail")), "close": raw["close"]})
              for nm, raw in PAN.items()})
LE = signals({nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "expand")), "close": raw["close"]})
              for nm, raw in PAN.items()})

print("=" * 130)
print("① 门槛扫描：reso（绝对个数） vs frac（比例）—— 两个样本集板块数不同，必须用 frac 统一")
print("=" * 130)
for name, D in (("生产 6 板块", DP), ("长历史 8 宽基·trail", LT), ("长历史 8 宽基·扩展锚", LE)):
    print(f"\n  【{name}】  全部 n={len(D)}")
    line(D, "　无过滤")
    for x in range(1, 7):
        line(D[D["reso"] >= x], f"　reso >= {x}")
    for f in (0.15, 0.25, 0.34, 0.5, 0.6):
        line(D[D["frac"] >= f - 1e-9], f"　frac >= {f:.2f}")
    for g in (-99, -12, -8, -6):
        sub = D[(D["reso"] >= 2) & (D["gap"] >= g)]
        line(sub, f"　reso>=2 且 gap>={g}")

print("\n" + "=" * 130)
print("② 2020 年那 3 次 reso=2 为什么全失败（唯一红灯）")
print("=" * 130)
sub = LT[(LT["date"].dt.year == 2020) & (LT["reso"] >= 2)]
print(sub[["board", "date", "ret20", "mdd20", "ret60", "reso", "frac", "gap", "rank", "ok20"]].to_string(index=False))
print("\n  同期（2020 年）其它信号：")
print(LT[LT["date"].dt.year == 2020][["board", "date", "ret20", "mdd20", "reso", "gap", "ok20"]].to_string(index=False))

print("\n" + "=" * 130)
print("③ 生产 6 板块 · 全部 33 次信号明细（按 reso 排序）")
print("=" * 130)
t = DP.copy()
t["date"] = t["date"].dt.strftime("%Y-%m-%d")
t["win"] = t["ok20"].map({True: "√", False: "×"})
print(t.sort_values(["reso", "date"])[["board", "date", "reso", "frac", "gap", "rank",
                                       "ret20", "mdd20", "ret60", "win"]].to_string(index=False,
      formatters={"ret20": lambda x: f"{x*100:+6.2f}%", "mdd20": lambda x: f"{x*100:+6.2f}%",
                  "ret60": lambda x: f"{x*100:+6.2f}%", "frac": lambda x: f"{x:.2f}",
                  "gap": lambda x: f"{x:+6.1f}"}))
