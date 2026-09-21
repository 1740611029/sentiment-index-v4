"""公平版前视检验：两种标定口径覆盖同一段评估期。

d36 的因果版只有 16 个样本，是因为 rolling(750, min_periods=250) 需要 250 天预热，
把 2023-10~2024-09 那段（信号最密集）全砍掉了 —— 那是预热期造成的，不是前视偏差。

本脚本：
  - 面板起点提前到数据允许的最早日（约 2022-12），仅作标定预热
  - 两种口径都只在 EVAL_START 之后统计信号，保证评估期一致
  - 因果窗口缩短为 rolling(250, min_periods=150)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors, model

OUT = os.path.join(C.DATA_DIR, "cache", "valid")
EXTRA_NAME = {"SZ50": "上证50", "CSI100": "中证100", "CSI500": "中证500",
              "CSI800": "中证800", "SH180": "上证180"}
H, TOL, COOL = 20, 0.03, 20
EVAL_START = pd.Timestamp(C.BACKTEST_START)   # 只统计此日之后的信号
W_CAL, MIN_CAL = 250, 150                     # 因果标定窗口（缩短以尽快可用）


def pmap_causal(v, lo_q, hi_q, lc, hc):
    x = v.astype(float)
    lo = x.rolling(W_CAL, min_periods=MIN_CAL).quantile(lo_q / 100.0)
    hi = x.rolling(W_CAL, min_periods=MIN_CAL).quantile(hi_q / 100.0)
    return (100.0 * (x - lo) / (hi - lo)).clip(lc, hc)


def pmap_full(v, lo_q, hi_q, lc, hc):
    return model._pct_map(v, lo_q, hi_q, lc, hc)


def raw_of_board(b, stock_ind):
    """交付板块：拿完整历史（不做 BACKTEST_START 过滤）作为预热"""
    raw = factors.build_board_raw(b, stock_ind)
    idx = data.load_index(b).set_index("date")
    for col in ("open", "high", "low"):
        raw[col] = idx[col].reindex(raw.index)
    return raw


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


def score_of(raw, pm):
    lo_q, hi_q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
    lc, hc = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
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
    """只在 EVAL_START 之后计数；预热段照常让规则跑（保持状态连续）"""
    dates = list(s.index)
    s = s.to_numpy(float); cl = c.to_numpy(float)
    out = []; last = -10 ** 9; n = len(s)
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if s[i - 1] <= thr and s[i] > s[i - 1]:
            last = i
            if dates[i] < EVAL_START:
                continue
            c0 = cl[i]; seg = cl[i + 1:i + H + 1]
            out.append((dates[i], seg[-1] / c0 - 1, seg.min() / c0 - 1, s[i]))
    return out


def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


print("载入 ...", flush=True)
stock_ind = data.build_stock_indicators()
BOARDS = {}
for b in C.BOARD_ORDER:
    BOARDS[C.BOARDS[b]["name"]] = (raw_of_board(b, stock_ind), "交付")
for key, name in EXTRA_NAME.items():
    fi = os.path.join(OUT, f"{key}_index.parquet"); fc = os.path.join(OUT, f"{key}_cons.json")
    if not (os.path.exists(fi) and os.path.exists(fc)): continue
    idx = pd.read_parquet(fi)
    if len(idx) == 0: continue
    raw = build_panel(key, idx, json.load(open(fc, encoding="utf-8")), stock_ind)
    if raw is not None and len(raw) > 200:
        BOARDS[name] = (raw, "验证")
print("  板块:", len(BOARDS), flush=True)
for nm, (raw, _) in list(BOARDS.items())[:3]:
    print(f"    面板起点 {raw.index.min().date()}  行数 {len(raw)}")

SC = {}
for nm, (raw, grp) in BOARDS.items():
    SC[nm] = (score_of(raw, pmap_full), score_of(raw, pmap_causal), raw["close"], grp)
print("\n计算完成\n", flush=True)

print("=" * 122)
print(f"公平对比：面板含预热段，两种口径都在 {EVAL_START.date()} 之后统计信号")
print("=" * 122)
print(f"  {'口径':<18}{'样本':>5}{'命中':>6}{'命中率':>8}{'未被套':>7}{'未被套率':>9}"
      f"{'均T+20':>9}{'均最深':>9}{'最差回撤':>10}{'威尔逊下界':>10}")
RESULT = {}
for tag, pick in (("全样本标定(前视)", 0), ("因果滚动标定", 1)):
    rows = []
    for nm, (raw, grp) in BOARDS.items():
        for d, r, m, v in entries(SC[nm][pick], SC[nm][2]):
            rows.append((nm, d, r, m, v, grp))
    RESULT[tag] = rows
    ok = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
    nt = sum(1 for x in rows if x[3] >= -TOL)
    print(f"  {tag:<18}{len(rows):>5}{ok:>6}{ok/len(rows)*100:>7.1f}%{nt:>7}"
          f"{nt/len(rows)*100:>8.1f}%{np.mean([x[2] for x in rows])*100:>+8.2f}%"
          f"{np.mean([x[3] for x in rows])*100:>+8.2f}%{min(x[3] for x in rows)*100:>+9.2f}%"
          f"{wilson(ok,len(rows))*100:>9.1f}%")

rows = RESULT["因果滚动标定"]
print("\n因果版未成立条目：")
bad = [x for x in rows if not (x[2] > 0 and x[3] >= -TOL)]
if not bad: print("  （无）")
for x in bad:
    print(f"  {x[0]:<8} {str(x[1].date())}  当日分{x[4]:6.1f}  T+20 {x[2]*100:+7.2f}%  最深 {x[3]*100:+7.2f}%")

print("\n因果版分板块：")
for nm in BOARDS:
    sub = [x for x in rows if x[0] == nm]
    if not sub: continue
    o = sum(1 for x in sub if x[2] > 0 and x[3] >= -TOL)
    print(f"  {nm:<10} {o}/{len(sub)}  均T+20 {np.mean([x[2] for x in sub])*100:+6.2f}%")

a = {(x[0], x[1]) for x in RESULT["全样本标定(前视)"]}
b = {(x[0], x[1]) for x in rows}
print(f"\n两口径信号重合度：因果版 {len(b)} 个中有 {len(a & b)} 个也出现在全样本版")
print(f"  只在全样本版出现：{len(a - b)} 个")
