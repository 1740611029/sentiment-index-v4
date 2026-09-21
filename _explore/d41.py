"""长历史（2013~2026，约 13 年）验证：把入场规则放到 2015 / 2018 / 2022 等熊市里跑。

目的有二：
 1) 让 p2/p98 分位锚点在 13 年样本上稳定，检验 −6 阈值是否可移植
 2) 拿到别的熊市样本，看规则是不是只在 2023-2026 这一段灵

数据源拼接：新浪(2013~2023-06, qfq) × 常数比率 → 对齐 v1/v2 缓存(2022-09~今)。
广度用累加器计算，不建大表。
成分股用**当前清单**回溯 —— 有幸存者偏差，已在结果中标注。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors

DEST = os.path.join(C.DATA_DIR, "cache", "stocks_hist")
IDEST = os.path.join(C.DATA_DIR, "cache", "index_hist")
H, TOL, COOL = 20, 0.03, 20
START = pd.Timestamp("2014-01-01")   # 前 1 年作滚动窗口预热

BOARDS = {   # key -> (名称, sina指数文件, 成分来源)
    "HS300":   ("沪深300", "HS300",   "board"),
    "CSI1000": ("中证1000", "CSI1000", "board"),
    "CSI2000": ("中证2000", "CSI2000", "board"),
    "CHINEXT": ("创业板", "CHINEXT", "board"),
    "SZ50":    ("上证50", "SZ50", "valid"),
    "CSI100":  ("中证100", "CSI100", "valid"),
    "CSI500":  ("中证500", "CSI500", "valid"),
    "CSI800":  ("中证800", "CSI800", "valid"),
    "SH180":   ("上证180", "SH180", "valid"),
}


def load_merged(code):
    """新浪(乘常数比率对齐) + v1/v2 缓存"""
    p = os.path.join(DEST, f"{code}.parquet")
    if not os.path.exists(p):
        return data.load_stock(code)
    a = pd.read_parquet(p)
    a["date"] = pd.to_datetime(a["date"])
    b = data.load_stock(code)
    if b is None or len(b) == 0:
        return a.sort_values("date").reset_index(drop=True)
    b["date"] = pd.to_datetime(b["date"])
    lo, hi = max(a["date"].min(), b["date"].min()), min(a["date"].max(), b["date"].max())
    ratio = 1.0
    if lo < hi:
        A = a[(a["date"] >= lo) & (a["date"] <= hi)].set_index("date")["close"].astype(float)
        B = b[(b["date"] >= lo) & (b["date"] <= hi)].set_index("date")["close"].astype(float)
        j = pd.concat([A, B], axis=1, keys=["s", "v"]).dropna()
        if len(j) >= 30:
            r = (j["v"] / j["s"]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(r) and r.mean() and (r.std() / r.mean()) < 0.02:
                ratio = float(r.median())
    a2 = a.copy()
    if abs(ratio - 1.0) > 1e-6:
        for c in ("open", "high", "low", "close"):
            if c in a2.columns:
                a2[c] = a2[c] * ratio
    cut = b["date"].min()
    a2 = a2[a2["date"] < cut]
    out = pd.concat([a2, b], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


print("准备板块 ...", flush=True)
BC = {}
for k, (name, ifile, src) in BOARDS.items():
    fp = os.path.join(IDEST, f"{ifile}.parquet")
    if not os.path.exists(fp):
        print(f"  {name} 无指数文件，跳过"); continue
    idx = pd.read_parquet(fp)
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if src == "board":
        codes = set(data.load_universe(k))
    else:
        cf = os.path.join(C.DATA_DIR, "cache", "valid", f"{ifile}_cons.json")
        if not os.path.exists(cf):
            print(f"  {name} 无成分文件，跳过"); continue
        codes = set(json.load(open(cf, encoding="utf-8")))
    idx = idx[idx["date"] >= START]
    BC[k] = dict(name=name, idx=idx, codes=codes,
                 pos={d: i for i, d in enumerate(idx["date"])})
print("  板块:", len(BC), flush=True)

ACC = {k: dict(
    b20=np.zeros(len(v["idx"])), b60=np.zeros(len(v["idx"])), r5=np.zeros(len(v["idx"])),
    nh=np.zeros(len(v["idx"])), nl=np.zeros(len(v["idx"])),
    lu=np.zeros(len(v["idx"])), ld=np.zeros(len(v["idx"])),
    amt=np.zeros(len(v["idx"])), s1=np.zeros(len(v["idx"])),
    s2=np.zeros(len(v["idx"])), cnt=np.zeros(len(v["idx"])))
    for k, v in BC.items()}

allcodes = sorted(set().union(*[v["codes"] for v in BC.values()]))
have = {f[:-8] for f in os.listdir(DEST) if f.endswith(".parquet")}
todo = [c for c in allcodes if c in have or True]
print(f"  需处理股票 {len(todo)}（其中已下载 {len([c for c in todo if c in have])}）", flush=True)

N = len(BOARDS)
done = 0
for ci, code in enumerate(todo, 1):
    try:
        df = load_merged(code)
        if df is None or len(df) < 70 or "close" not in df:
            continue
        ind = data.compute_stock_indicators(df, code)
        ind["date"] = pd.to_datetime(ind["date"])
        for k, v in BC.items():
            if code not in v["codes"]:
                continue
            p = ind["date"].map(v["pos"])
            m = p.notna().to_numpy()
            if not m.any():
                continue
            pos_i = p[m].astype(int).to_numpy()
            n = len(v["idx"])
            g = ACC[k]
            g["cnt"] += np.bincount(pos_i, minlength=n)
            for col, key in (("above_ma20", "b20"), ("above_ma60", "b60"), ("up5", "r5"),
                             ("is_nh60", "nh"), ("is_nl60", "nl"),
                             ("any_lu5", "lu"), ("any_ld5", "ld")):
                g[key] += np.bincount(pos_i, weights=ind[col].to_numpy()[m], minlength=n)
            g["amt"] += np.bincount(pos_i, weights=ind["amount"].to_numpy()[m], minlength=n)
            rv = ind["ret5"].to_numpy()[m]
            rv = np.where(np.isnan(rv), 0.0, rv)
            g["s1"] += np.bincount(pos_i, weights=rv, minlength=n)
            g["s2"] += np.bincount(pos_i, weights=rv * rv, minlength=n)
        done += 1
    except Exception:
        continue
    if ci % 500 == 0:
        print(f"    ... {ci}/{len(todo)}  有效 {done}", flush=True)
print(f"  个股处理完成，有效 {done}\n", flush=True)


def score_of(raw):
    lo_q, hi_q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
    lc, hc = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
    from senti import model
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
            c0 = cl[i]; seg = cl[i + 1:i + H + 1]
            out.append((dates[i], seg[-1] / c0 - 1, seg.min() / c0 - 1, s[i]))
    return out


def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


print("计算分值并跑规则 ...", flush=True)
ROWS = []
for k, v in BC.items():
    g = ACC[k]; idx = v["idx"]; n = len(idx)
    cnt = g["cnt"]
    ok = cnt >= 20
    def safe(a):
        out = np.full(n, np.nan)
        out[ok] = a[ok] / cnt[ok]
        return out
    raw = pd.DataFrame(index=idx["date"])
    raw["close"] = idx["close"].astype(float).to_numpy()
    raw["b20"] = safe(g["b20"]); raw["b60"] = safe(g["b60"]); raw["r5"] = safe(g["r5"])
    raw["nh"] = safe(g["nh"] - g["nl"]); raw["lim"] = safe(g["lu"] - g["ld"])
    amt = np.where(ok, g["amt"], np.nan)
    raw["amt_series"] = amt
    var = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        var[ok] = (g["s2"][ok] - g["s1"][ok] ** 2 / cnt[ok]) / np.maximum(cnt[ok] - 1, 1)
    raw["disp"] = np.sqrt(np.where(var > 0, var, np.nan))
    raw["amt_pct"] = pd.Series(amt, index=raw.index).rolling(250, min_periods=60).rank(pct=True)
    close = raw["close"]
    raw["rsi"] = factors._rsi(close, 14)
    raw["bias"] = close / close.rolling(60).mean() - 1.0
    raw["ret20"] = close.pct_change(20)
    raw = raw.drop(columns=["amt_series"])
    raw = raw[raw.index >= START]
    sc = score_of(raw)
    ev = entries(sc, raw["close"])
    for d, r, m, s0 in ev:
        ROWS.append((v["name"], d, r, m, s0))
    print(f"  {v['name']:<8} 样本 {len(idx)} 天  信号 {len(ev)} 个  "
          f"分值区间 {sc.min():.0f}~{sc.max():.0f}", flush=True)

print("\n" + "=" * 122)
print(f"长历史验证  {START.date()} ~ 今   规则：昨 ≤ −6 且今日回升 → 今日收盘买"
      f"（T+{H}，容差 {TOL:.0%}，冷却 {COOL} 交易日）")
print("=" * 122)
ok = sum(1 for x in ROWS if x[2] > 0 and x[3] >= -TOL)
nt = sum(1 for x in ROWS if x[3] >= -TOL)
print(f"  合计  样本 {len(ROWS)}   命中 {ok} = {ok/len(ROWS)*100:.1f}%   "
      f"未被套 {nt} = {nt/len(ROWS)*100:.1f}%   均T+20 {np.mean([x[2] for x in ROWS])*100:+.2f}%   "
      f"均最深 {np.mean([x[3] for x in ROWS])*100:+.2f}%   最差 {min(x[3] for x in ROWS)*100:+.2f}%   "
      f"威尔逊下界 {wilson(ok,len(ROWS))*100:.1f}%")

print("\n分年度：")
print(f"  {'年份':<6}{'样本':>5}{'命中':>6}{'命中率':>8}{'未被套':>7}{'未被套率':>9}{'均T+20':>9}{'最差回撤':>10}")
for y in sorted({x[1].year for x in ROWS}):
    sub = [x for x in ROWS if x[1].year == y]
    o = sum(1 for x in sub if x[2] > 0 and x[3] >= -TOL)
    n2 = sum(1 for x in sub if x[3] >= -TOL)
    print(f"  {y:<6}{len(sub):>5}{o:>6}{o/len(sub)*100:>7.1f}%{n2:>7}{n2/len(sub)*100:>8.1f}%"
          f"{np.mean([x[2] for x in sub])*100:>+8.2f}%{min(x[3] for x in sub)*100:>+9.2f}%")

print("\n分板块：")
for nm in sorted({x[0] for x in ROWS}):
    sub = [x for x in ROWS if x[0] == nm]
    o = sum(1 for x in sub if x[2] > 0 and x[3] >= -TOL)
    n2 = sum(1 for x in sub if x[3] >= -TOL)
    print(f"  {nm:<10}{o}/{len(sub):<4} 命中 {o/len(sub)*100:5.1f}%   未被套 {n2}/{len(sub)}   "
          f"均T+20 {np.mean([x[2] for x in sub])*100:+6.2f}%")
