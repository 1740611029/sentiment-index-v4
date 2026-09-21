"""在扩充样本（6 交付板块 + 5 额外宽基 = 11 个板块）上重扫入场阈值。
目标：看 −6 在 41 个样本里还是不是最优点，还是说更深的阈值能把 2 个擦边失败也清掉。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"; os.environ["no_proxy"] = "*"
import numpy as np, pandas as pd
from senti import config as C, data, factors, model
from importlib.machinery import SourceFileLoader

D34 = SourceFileLoader("d34", os.path.join(os.path.dirname(os.path.abspath(__file__)), "d34.py"))

OUT = os.path.join(C.DATA_DIR, "cache", "valid")
EXTRA_NAME = {"SZ50": "上证50", "CSI100": "中证100", "CSI500": "中证500",
              "CSI800": "中证800", "SH180": "上证180", "CSIDIV": "中证红利"}
H, TOL, COOL = 20, 0.03, 20


def build_panel(key, idx, codes, stock_ind):
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
    pm = lambda v: model._pct_map(v, lo_q, hi_q, lc, hc)
    num = pd.Series(0.0, index=raw.index); den = 0.0
    for k, w in model.L_WEIGHTS.items():
        num = num + pm(raw[k]) * w; den += w
    L = num / den
    T = (pm(raw["bias"]) * model.T_WEIGHTS["bias"] + pm(raw["ret20"]) * model.T_WEIGHTS["ret20"]
         + pm(raw["rsi"]) * model.T_WEIGHTS["rsi"] + raw["amt_pct"] * 100.0 * model.T_WEIGHTS["amt"])
    U = model.W_L * L + (1 - model.W_L) * T
    cfb = (((raw["amt_pct"].astype(float) - 0.60) / 0.40).clip(0, 1).fillna(0.0)
           * ((12.0 - L) / 12.0).clip(0, 1).fillna(0.0)).fillna(0.0)
    dd250 = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    f_disp = (100.0 - pm(raw["disp"])).fillna(0.0)
    f_dd = (100.0 - pm(dd250)).fillna(0.0)
    cfg = (((f_dd - 90.0) / 12.0).clip(0, 1) * ((f_disp - 75.0) / 12.0).clip(0, 1)).fillna(0.0)
    return (U - 25.0 * cfb - 30.0 * cfg).clip(-30.0, 140.0)


print("载入 ...", flush=True)
stock_ind = data.build_stock_indicators()
panels = model.build_all()
SER = {}
for b in C.BOARD_ORDER:
    p = panels[b]
    p = p[p.index >= pd.Timestamp(C.BACKTEST_START)]
    SER[C.BOARDS[b]["name"]] = (p["score"].to_numpy(float), p["close"].to_numpy(float),
                                list(p.index), "交付")
for key, name in EXTRA_NAME.items():
    fi = os.path.join(OUT, f"{key}_index.parquet"); fc = os.path.join(OUT, f"{key}_cons.json")
    if not (os.path.exists(fi) and os.path.exists(fc)): continue
    idx = pd.read_parquet(fi)
    if len(idx) == 0: continue
    codes = json.load(open(fc, encoding="utf-8"))
    raw = build_panel(key, idx, codes, stock_ind)
    if raw is None: continue
    raw = raw[raw.index >= pd.Timestamp(C.BACKTEST_START)]
    if len(raw) < 200: continue
    sc = score_of(raw)
    SER[name] = (sc.to_numpy(float), raw["close"].to_numpy(float), list(raw.index), "验证")
print("  板块数:", len(SER), flush=True)


def run(thr, group=None):
    out = []
    for nm, (s, c, dates, grp) in SER.items():
        if group and grp != group: continue
        n = len(s); last = -10 ** 9
        for i in range(1, n - H):
            if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
            if i - last < COOL: continue
            if s[i - 1] <= thr and s[i] > s[i - 1]:
                last = i
                c0 = c[i]; seg = c[i + 1:i + H + 1]
                out.append((nm, dates[i], seg[-1] / c0 - 1, seg.min() / c0 - 1, s[i]))
    return out


def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


print("\n" + "=" * 118)
print("阈值重扫（全部 11 个板块；昨 ≤ X 且今日回升 → 今日收盘买）")
print("=" * 118)
print(f"  {'X':>5} {'样本':>5} {'命中':>7} {'命中率':>7} {'未被套':>7} {'未被套率':>8} "
      f"{'均T+20':>8} {'均最深':>8} {'威尔逊下界':>10}")
for x in [-3, -4, -5, -6, -7, -8, -9, -10, -12, -15]:
    r = run(float(x))
    if not r: continue
    ok = sum(1 for e in r if e[2] > 0 and e[3] >= -TOL)
    nt = sum(1 for e in r if e[3] >= -TOL)
    print(f"  {x:>5} {len(r):>5} {ok:>7} {ok/len(r)*100:>6.1f}% {nt:>7} "
          f"{nt/len(r)*100:>7.1f}% {np.mean([e[2] for e in r])*100:>+7.2f}% "
          f"{np.mean([e[3] for e in r])*100:>+7.2f}% {wilson(ok,len(r))*100:>9.1f}%")

print("\n" + "=" * 118)
print("X = −6 的 41 个样本里，未成立的条目")
print("=" * 118)
for e in run(-6.0):
    if not (e[2] > 0 and e[3] >= -TOL):
        print(f"  {e[0]:<8} {str(e[1].date())}  当日分{e[4]:6.1f}  T+20 {e[2]*100:+7.2f}%  最深 {e[3]*100:+7.2f}%")

print("\n" + "=" * 118)
print("最差 5 次（按最深回撤排序）")
print("=" * 118)
for e in sorted(run(-6.0), key=lambda z: z[3])[:5]:
    print(f"  {e[0]:<8} {str(e[1].date())}  当日分{e[4]:6.1f}  T+20 {e[2]*100:+7.2f}%  最深 {e[3]*100:+7.2f}%")
