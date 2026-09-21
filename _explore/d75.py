"""d75：把「同一次市场恐慌」聚成一个事件，重新算命中率。

动机（d74 发现）：近 3 年 17 次底部信号只落在 5 个日期上
（2023-10-24、2024-01-23、2024-02-01、2025-04-08、2026-03-24），
每次多个板块同日触发。
→ 那么「17/19 = 89.5%」里有多少是**独立样本**？如果每个板块只是同一次市场恐慌的重复计数，
  有效样本量其实是事件数（5），不是信号数（17）。这会显著改变置信区间。

做法：
  ① 按日期把跨板块的信号聚类（相邻 ≤5 交易日算同一事件）
  ② 事件级结果 = 该事件内所有板块 T+20 收益的均值；事件算赢 = 均值 > 0
  ③ 事件级随机基线：随机挑一天，取同样多个板块的 T+20 均值 > 0 的概率
  ④ 长历史 8 宽基同样处理，把事件样本量做大
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C
from senti import model as M
from senti import factors as F
import longhist

H, H2, COOL, TOL = 20, 60, 20, 0.03
W2 = pd.Timestamp("2023-09-20")
LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
LO_Q, HI_Q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
EXCLUDE = {"CSI2000"}
KEYS = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20", "disp", "dd250"]
WIN, MINP = C.MODEL["anchor_window"], C.MODEL["anchor_min"]
CLIP_LO, CLIP_HI = -30.0, 140.0
rng = np.random.default_rng(20260921)


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


def signals(panels, thr=0.0):
    S = pd.DataFrame({nm: p["score"] for nm, p in panels.items()}).dropna(how="all")
    CL = pd.DataFrame({nm: p["close"] for nm, p in panels.items()}).reindex(S.index)
    rows = []
    for nm in panels:
        v = S[nm].to_numpy(float); cl = CL[nm].to_numpy(float)
        n = len(v); last = -10 ** 9
        for i in range(1, n - H2):
            if np.isnan(v[i]) or np.isnan(v[i - 1]): continue
            if i - last < COOL: continue
            if v[i - 1] > thr or v[i] <= v[i - 1]: continue
            last = i
            c0 = cl[i]
            seg = cl[i + 1:i + H + 1]; seg2 = cl[i + 1:i + H2 + 1]
            rows.append({"board": nm, "date": S.index[i], "pos": i,
                         "ret20": seg[-1] / c0 - 1, "mdd20": seg.min() / c0 - 1,
                         "ret60": seg2[-1] / c0 - 1,
                         "ok20": bool(seg[-1] / c0 - 1 > 0 and seg.min() / c0 - 1 >= -TOL)})
    return pd.DataFrame(rows)


def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def cluster(df, gap=5):
    """把跨板块信号按交易日位置聚类：相邻 ≤ gap 个交易日算同一事件"""
    if len(df) == 0: return []
    d = df.sort_values("pos").copy()
    ev = []; cur = [d.iloc[0]]
    for r in d.iloc[1:].itertuples():
        if r.pos - cur[-1]["pos"] <= gap:
            cur.append({"board": r.board, "date": r.date, "pos": r.pos,
                        "ret20": r.ret20, "mdd20": r.mdd20, "ret60": r.ret60, "ok20": r.ok20})
        else:
            ev.append(cur); cur = [{"board": r.board, "date": r.date, "pos": r.pos,
                                    "ret20": r.ret20, "mdd20": r.mdd20, "ret60": r.ret60, "ok20": r.ok20}]
    ev.append(cur)
    return ev


def event_rows(ev):
    out = []
    for g in ev:
        r20 = np.mean([x["ret20"] for x in g])
        m20 = np.mean([x["mdd20"] for x in g])
        r60 = np.mean([x["ret60"] for x in g])
        boards = sorted({x["board"] for x in g})
        days = sorted({x["date"] for x in g})
        out.append({"start": days[0], "n": len(g), "nboard": len(boards),
                    "boards": ",".join(boards),
                    "ret20": r20, "mdd20": m20, "ret60": r60,
                    "win": bool(r20 > 0),
                    "win60": bool(r60 > 0),
                    "allok": bool(all(x["ok20"] for x in g)),
                    "nok": sum(1 for x in g if x["ok20"])})
    return pd.DataFrame(out)


def event_baseline(panels, k_sizes, nsim=6000):
    """事件级基线：随机挑一天，随机取 k 个板块，T+20 均值 > 0 的概率"""
    names = list(panels.keys())
    CL = pd.DataFrame({nm: panels[nm]["close"] for nm in names}).dropna(how="all")
    cl = CL.to_numpy(float); n = len(cl)
    idx = rng.integers(0, n - H - 1, nsim)
    wins = []
    for k in k_sizes:
        w = 0
        for i in idx:
            bs = rng.choice(len(names), k, replace=False)
            r = np.mean([cl[i + H, b] / cl[i, b] - 1 for b in bs])
            w += (r > 0)
        wins.append(w / len(idx) * 100)
    return np.mean(wins)


def report(tag, pan, thr=0.0):
    df = signals(pan, thr)
    if len(df) == 0:
        print(f"  {tag}: 无信号"); return
    ev = cluster(df)
    E = event_rows(ev)
    n = len(E); w = int(E["win"].sum()); w60 = int(E["win60"].sum())
    base = event_baseline(pan, sorted(set(E["nboard"])))
    print(f"\n【{tag}】信号 {len(df)} 次 → 聚成 {n} 个事件（每事件含 {E['nboard'].min()}~{E['nboard'].max()} 个板块）")
    print(f"  信号级命中率 {int(df['ok20'].sum())}/{len(df)} = {df['ok20'].mean()*100:.1f}%"
          f"（下界 {wil(int(df['ok20'].sum()),len(df))*100:.1f}%）   <-- 相关样本，会虚高")
    print(f"  ★ 事件级命中率 {w}/{n} = {w/n*100:.1f}%"
          f"（下界 {wil(w,n)*100:.1f}%）  随机基线 {base:.1f}%")
    print(f"  事件级 T+60 盈利 {w60}/{n} = {w60/n*100:.1f}%　"
          f"事件均 T+20 {E['ret20'].mean()*100:+.2f}%　最差事件 {E['ret20'].min()*100:+.2f}%"
          f"　最深浮亏 {E['mdd20'].min()*100:+.2f}%")
    print(f"  {'起始日':<12}{'板块数':>6}{'信号数':>6}{'T+20均值':>10}{'T+60均值':>10}{'最深':>9}{'判定':>6}")
    for r in E.itertuples():
        print(f"  {str(r.start)[:10]:<12}{r.nboard:>6}{r.n:>6}{r.ret20*100:>9.2f}%"
              f"{r.ret60*100:>9.2f}%{r.mdd20*100:>8.2f}%{'✔' if r.win else '✘':>6}")
    return E


print("=" * 118)
print("d75 · 信号聚类成「市场级事件」后，命中率还剩多少？（有效样本量检验）")
print("=" * 118)

RAW = {b: F.build_board_raw(b) for b in C.BOARD_ORDER}
PP = {b: pd.DataFrame({"score": score(RAW[b], make_anchors(RAW[b], "trail")),
                       "close": RAW[b]["close"]}) for b in C.BOARD_ORDER}
PP = {b: p[p.index >= W2] for b, p in PP.items()}
report("生产 6 板块 2023-09-20 起", PP)

PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
LP = {nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "trail")), "close": raw["close"]})
      for nm, raw in PAN.items()}
EP = {nm: pd.DataFrame({"score": score(raw, make_anchors(raw, "expand")), "close": raw["close"]})
      for nm, raw in PAN.items()}
report("长历史 8 宽基 2013~2026 · trail 锚", LP)
report("长历史 8 宽基 2013~2026 · 扩展锚", EP)

print("\n" + "=" * 118)
print("结论用的一句话")
print("=" * 118)
