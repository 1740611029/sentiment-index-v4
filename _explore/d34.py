"""样本扩充验证：把入场规则「昨 ≤ −6 且今日回升」放到更多板块上跑。

动机：原 6 板块只有 20 个样本，100% 的威尔逊下界约 84%，不足以断言。
这里额外加 6 个宽基指数（新浪日线 + 中证成分股），把样本翻上去。
个股日线复用现有 stock_ind 缓存，不需要下载新数据。

注意：新增板块只用于**验证**，不改变交付的 6 个板块。
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"; os.environ["no_proxy"] = "*"
import numpy as np, pandas as pd
import akshare as ak
from senti import config as C, data, factors, model

OUT = os.path.join(C.DATA_DIR, "cache", "valid")
os.makedirs(OUT, exist_ok=True)

EXTRA = {
    "SZ50":   ("sh000016", "000016", "上证50"),
    "CSI100": ("sh000903", "000903", "中证100"),
    "CSI500": ("sh000905", "000905", "中证500"),
    "CSI800": ("sh000906", "000906", "中证800"),
    "SH180":  ("sh000010", "000010", "上证180"),
    "CSIDIV": ("sh000922", "000922", "中证红利"),
}

H, TOL, COOL, ENTRY_THR = 20, 0.03, 20, -6.0


def fetch():
    """抓指数日线 + 成分股，缓存到本地"""
    got = {}
    for key, (sina, code, name) in EXTRA.items():
        fi = os.path.join(OUT, f"{key}_index.parquet")
        fc = os.path.join(OUT, f"{key}_cons.json")
        if os.path.exists(fi) and os.path.exists(fc):
            got[key] = (pd.read_parquet(fi), json.load(open(fc, encoding="utf-8")), name)
            print(f"  {name:<8} 已缓存 指数{len(got[key][0])}行 成分{len(got[key][1])}只")
            continue
        try:
            d = ak.stock_zh_index_daily(symbol=sina)
            d["date"] = pd.to_datetime(d["date"])
            d = d[d["date"] >= pd.Timestamp("2022-06-01")].reset_index(drop=True)
            if "amount" not in d.columns:
                d["amount"] = d["volume"] * d["close"]
            d.to_parquet(fi, index=False)
            c = ak.index_stock_cons(symbol=code)
            codes = sorted(str(x).zfill(6) for x in c["品种代码"].tolist())
            json.dump(codes, open(fc, "w", encoding="utf-8"))
            got[key] = (d, codes, name)
            print(f"  {name:<8} 抓取成功 指数{len(d)}行 成分{len(codes)}只")
        except Exception as e:
            print(f"  {name:<8} FAIL {type(e).__name__}: {str(e)[:70]}")
        time.sleep(0.3)
    return got


def build_panel(key, idx, codes, stock_ind):
    """与 factors.build_board_raw 同构，但用外部传入的指数/成分"""
    sub = stock_ind[stock_ind["code"].isin(set(codes))]
    if sub.empty:
        return None
    g = sub.groupby("date", sort=True)
    breadth = pd.DataFrame({
        "b20": g["above_ma20"].mean(), "b60": g["above_ma60"].mean(),
        "r5": g["up5"].mean(),
        "nh": g["is_nh60"].mean() - g["is_nl60"].mean(),
        "lim": g["any_lu5"].mean() - g["any_ld5"].mean(),
        "amt": g["amount"].sum(), "disp": g["ret5"].std(),
        "n": g["above_ma20"].size(),
    })
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
    df["high"] = idx["high"].astype(float)
    df["low"] = idx["low"].astype(float)
    return df.drop(columns=["amt", "n"])


def score_of(raw):
    """与 model.build 完全一致"""
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


def live_entries(s, c):
    """昨 ≤ ENTRY_THR 且今日回升 → 今日收盘买；入场后 20 交易日冷却"""
    dates = list(s.index)          # 必须先取索引，再转 numpy
    s = s.to_numpy(dtype=float); cl = c.to_numpy(dtype=float)
    out = []; last = -10 ** 9; n = len(s)
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if s[i - 1] <= ENTRY_THR and s[i] > s[i - 1]:
            last = i
            c0 = cl[i]; seg = cl[i + 1:i + H + 1]
            out.append((dates[i], seg[-1] / c0 - 1, seg.min() / c0 - 1, s[i]))
    return out


print("抓数据 ...", flush=True)
got = fetch()
print("\n载入个股指标缓存 ...", flush=True)
stock_ind = data.build_stock_indicators()
print("done\n", flush=True)

print("=" * 118)
print(f"入场规则：昨 ≤ {ENTRY_THR:.0f} 且今日回升 → 今日收盘买（T+{H}，容差 {TOL:.0%}，冷却 {COOL} 交易日）")
print("=" * 118)

rows = []
# 原 6 板块
panels = model.build_all()
for b in C.BOARD_ORDER:
    p = panels[b]
    p2 = p[p.index >= pd.Timestamp(C.BACKTEST_START)]
    for d, r, m, sc in live_entries(p2["score"], p2["close"]):
        rows.append((C.BOARDS[b]["name"], d, r, m, sc, "原6板块"))
# 新增板块
for key, (idx, codes, name) in got.items():
    if idx is None or len(idx) == 0:
        print(f"  {name} 指数数据为空，跳过"); continue
    raw = build_panel(key, idx, codes, stock_ind)
    if raw is None:
        print(f"  {name} 无成分股数据"); continue
    raw = raw[raw.index >= pd.Timestamp(C.BACKTEST_START)]
    if len(raw) < 200:
        print(f"  {name} 历史不足({len(raw)}行)"); continue
    sc_ser = score_of(raw)
    for d, r, m, sc in live_entries(sc_ser, raw["close"]):
        rows.append((name, d, r, m, sc, "新增验证"))

ok = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
nt = sum(1 for x in rows if x[3] >= -TOL)
print(f"  合计   命中 {ok}/{len(rows)} = {ok/len(rows)*100:.1f}%   "
      f"未被套 {nt}/{len(rows)} = {nt/len(rows)*100:.1f}%   "
      f"均T+20 {np.mean([x[2] for x in rows])*100:+.2f}%   "
      f"均最深 {np.mean([x[3] for x in rows])*100:+.2f}%")

for grp in ("原6板块", "新增验证"):
    sub = [x for x in rows if x[5] == grp]
    if not sub: continue
    o = sum(1 for x in sub if x[2] > 0 and x[3] >= -TOL)
    n2 = sum(1 for x in sub if x[3] >= -TOL)
    print(f"    {grp}  命中 {o}/{len(sub)} = {o/len(sub)*100:.1f}%   "
          f"未被套 {n2}/{len(sub)} = {n2/len(sub)*100:.1f}%   "
          f"均T+20 {np.mean([x[2] for x in sub])*100:+.2f}%")

print("\n分板块：")
names = []
for x in rows:
    if x[0] not in names: names.append(x[0])
for nm in names:
    sub = [x for x in rows if x[0] == nm]
    o = sum(1 for x in sub if x[2] > 0 and x[3] >= -TOL)
    n2 = sum(1 for x in sub if x[3] >= -TOL)
    print(f"  {nm:<10} {o}/{len(sub)} 命中   未被套 {n2}/{len(sub)}   "
          f"均T+20 {np.mean([x[2] for x in sub])*100:+6.2f}%")

print("\n未成立的条目：")
bad = [x for x in rows if not (x[2] > 0 and x[3] >= -TOL)]
if not bad:
    print("  （无）")
for x in bad:
    print(f"  {x[0]:<10} {str(x[1].date())}  当日分{x[4]:6.1f}  T+20 {x[2]*100:+7.2f}%  最深 {x[3]*100:+7.2f}%")

# Wilson 置信下界
def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n
    d = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (ctr - half) / d
print(f"\n威尔逊 95% 置信下界：原 6 板块 {wilson(20,20)*100:.1f}%   "
      f"全部 {len(rows)} 个样本 {wilson(ok,len(rows))*100:.1f}%")
