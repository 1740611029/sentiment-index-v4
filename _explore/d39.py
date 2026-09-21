"""d38 的 A1 vs A2 混了「面板长度不一致」：交付板块指数只到 2023-06，
额外板块我从 2022-06 抓的。这里把所有 11 个板块**统一截到同一起点**再比，
干净地隔离「标定样本长度」这一个变量。

可用起点受限于：个股指标缓存 2022-12 起、交付板块指数缓存 2023-06 起。
所以能比的起点只有 2023-06-01 与 2023-09-20。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors, model

OUT = os.path.join(C.DATA_DIR, "cache", "valid")
EXTRA_NAME = {"SZ50": "上证50", "CSI100": "中证100", "CSI500": "中证500",
              "CSI800": "中证800", "SH180": "上证180"}
H, TOL, COOL = 20, 0.03, 20
EVAL_START = pd.Timestamp(C.BACKTEST_START)


def build_extra(key, idx, codes, stock_ind):
    sub = stock_ind[stock_ind["code"].isin(set(codes))]
    if sub.empty: return None
    g = sub.groupby("date", sort=True)
    breadth = pd.DataFrame({
        "b20": g["above_ma20"].mean(), "b60": g["above_ma60"].mean(), "r5": g["up5"].mean(),
        "nh": g["is_nh60"].mean() - g["is_nl60"].mean(),
        "lim": g["any_lu5"].mean() - g["any_ld5"].mean(),
        "amt": g["amount"].sum(), "disp": g["ret5"].std(), "n": g["above_ma20"].size()})
    breadth.index = pd.to_datetime(breadth.index)
    idx = idx.set_index("date")
    df = pd.DataFrame(index=idx.index)
    df["close"] = idx["close"].astype(float)
    df = df.join(breadth, how="left")
    df.loc[df["n"] < 20, ["b20", "b60", "r5", "nh", "lim"]] = np.nan
    amt = df["amt"].fillna(idx["amount"].astype(float))
    df["amt_pct"] = amt.rolling(250, min_periods=60).rank(pct=True)
    close = df["close"]
    df["rsi"] = factors._rsi(close, 14)
    df["bias"] = close / close.rolling(60).mean() - 1.0
    df["ret20"] = close.pct_change(20)
    return df.drop(columns=["amt", "n"])


def score_of(raw):
    lo_q, hi_q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
    lc, hc = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
    pm = model._pct_map
    num = pd.Series(0.0, index=raw.index); den = 0.0
    for k, w in model.L_WEIGHTS.items():
        num = num + pm(raw[k], lo_q, hi_q, lc, hc) * w; den += w
    L = num / den
    T = (pm(raw["bias"], lo_q, hi_q, lc, hc) * model.T_WEIGHTS["bias"]
         + pm(raw["ret20"], lo_q, hi_q, lc, hc) * model.T_WEIGHTS["ret20"]
         + pm(raw["rsi"], lo_q, hi_q, lc, hc) * model.T_WEIGHTS["rsi"]
         + raw["amt_pct"] * 100.0 * model.T_WEIGHTS["amt"])
    U = model.W_L * L + (1 - model.W_L) * T
    cfb = (((raw["amt_pct"].astype(float) - 0.60) / 0.40).clip(0, 1).fillna(0.0)
           * ((12.0 - L) / 12.0).clip(0, 1).fillna(0.0)).fillna(0.0)
    dd250 = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    f_disp = (100.0 - pm(raw["disp"], lo_q, hi_q, lc, hc)).fillna(0.0)
    f_dd = (100.0 - pm(dd250, lo_q, hi_q, lc, hc)).fillna(0.0)
    cfg = (((f_dd - 90.0) / 12.0).clip(0, 1) * ((f_disp - 75.0) / 12.0).clip(0, 1)).fillna(0.0)
    return (U - 25.0 * cfb - 30.0 * cfg).clip(-30.0, 140.0)


def entries(s, c, thr=-6.0):
    dates = list(s.index)
    s = s.to_numpy(float); cl = c.to_numpy(float)
    out = []; last = -10 ** 9; n = len(s)
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if s[i - 1] <= thr and s[i] > s[i - 1]:
            last = i
            if dates[i] < EVAL_START: continue
            c0 = cl[i]; seg = cl[i + 1:i + H + 1]
            out.append((dates[i], seg[-1] / c0 - 1, seg.min() / c0 - 1, s[i]))
    return out


def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


print("载入 ...", flush=True)
stock_ind = data.build_stock_indicators()
RAW = {}
for b in C.BOARD_ORDER:
    raw = factors.build_board_raw(b, stock_ind)
    idx = data.load_index(b).set_index("date")
    for col in ("open", "high", "low"):
        raw[col] = idx[col].reindex(raw.index)
    RAW[(C.BOARDS[b]["name"], "交付")] = raw
for key, name in EXTRA_NAME.items():
    fi = os.path.join(OUT, f"{key}_index.parquet"); fc = os.path.join(OUT, f"{key}_cons.json")
    if not (os.path.exists(fi) and os.path.exists(fc)): continue
    idx = pd.read_parquet(fi)
    if len(idx) == 0: continue
    r = build_extra(key, idx, json.load(open(fc, encoding="utf-8")), stock_ind)
    if r is not None and len(r) > 200:
        RAW[(name, "验证")] = r
print("  板块", len(RAW), flush=True)

print("\n" + "=" * 124)
print("统一截断点 → 只隔离「标定样本长度」")
print("=" * 124)
print(f"  {'标定起点':<14}{'每板块行数':>10}{'样本':>5}{'命中':>6}{'命中率':>8}"
      f"{'未被套':>7}{'未被套率':>9}{'均T+20':>9}{'最差回撤':>10}{'威尔逊下界':>10}")

for start in ("2023-09-20", "2023-06-01"):
    S = pd.Timestamp(start)
    rows = []
    nrow = None
    for (nm, grp), raw in RAW.items():
        if raw.index.min() > S: continue
        sub = raw[raw.index >= S]
        nrow = len(sub)
        sc = score_of(sub)
        for d, r, m, v in entries(sc, sub["close"]):
            rows.append((nm, d, r, m, v, grp))
    ok = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
    nt = sum(1 for x in rows if x[3] >= -TOL)
    print(f"  {start:<14}{nrow:>10}{len(rows):>5}{ok:>6}{ok/len(rows)*100:>7.1f}%{nt:>7}"
          f"{nt/len(rows)*100:>8.1f}%{np.mean([x[2] for x in rows])*100:>+8.2f}%"
          f"{min(x[3] for x in rows)*100:>+9.2f}%{wilson(ok,len(rows))*100:>9.1f}%")
    # 分组
    for g in ("交付", "验证"):
        sub = [x for x in rows if x[5] == g]
        if not sub: continue
        o = sum(1 for x in sub if x[2] > 0 and x[3] >= -TOL)
        print(f"        └ {g}板块 {o}/{len(sub)} = {o/len(sub)*100:.1f}%")
    LAST = rows

print("\n两组信号重合度：")
A = set()
S = pd.Timestamp("2023-09-20")
for (nm, grp), raw in RAW.items():
    sub = raw[raw.index >= S]
    for d, r, m, v in entries(score_of(sub), sub["close"]):
        A.add((nm, d))
B = {(x[0], x[1]) for x in LAST}
print(f"  起点2023-09-20: {len(A)} 个；起点2023-06-01: {len(B)} 个；重合 {len(A & B)} 个")

print("\n起点 2023-06-01 的未成立条目：")
for x in sorted(LAST, key=lambda z: z[3]):
    if not (x[2] > 0 and x[3] >= -TOL):
        print(f"  {x[0]:<8} {str(x[1].date())}  当日分{x[4]:6.1f}  T+20 {x[2]*100:+7.2f}%  最深 {x[3]*100:+7.2f}%")
