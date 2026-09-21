"""隔离 d37 里被搅在一起的三个变量，定位 95% → 69% 到底是谁造成的。

变量：
  V1 面板起点   726 行(2023-09-20起，生产口径) vs 803 行(2023-06-01起，含预热)
  V2 标定估计量 全样本分位 vs 因果扩展窗口 vs 因果滚动250
  V3 预热期是否消耗冷却额度（预热段触发的信号会不会吃掉后面的冷却）

配置：
  A1 726 + 全样本   —— 生产现状（d35 = 95.1%）
  A2 803 + 全样本   —— 只改 V1（d37 的全样本版，69.2%）
  A3 803 + 全样本 + 预热不算冷却  —— 去掉 V3
  B1 803 + 扩展窗口（真因果，用截至当日全部历史）
  B2 803 + 滚动250
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


def pm_full(v, lo_q, hi_q, lc, hc):
    return model._pct_map(v, lo_q, hi_q, lc, hc)


def pm_expand(v, lo_q, hi_q, lc, hc, mp=150):
    x = v.astype(float)
    lo = x.expanding(min_periods=mp).quantile(lo_q / 100.0)
    hi = x.expanding(min_periods=mp).quantile(hi_q / 100.0)
    return (100.0 * (x - lo) / (hi - lo)).clip(lc, hc)


def pm_roll(v, lo_q, hi_q, lc, hc, w=250, mp=150):
    x = v.astype(float)
    lo = x.rolling(w, min_periods=mp).quantile(lo_q / 100.0)
    hi = x.rolling(w, min_periods=mp).quantile(hi_q / 100.0)
    return (100.0 * (x - lo) / (hi - lo)).clip(lc, hc)


def raw_full(b, stock_ind):
    raw = factors.build_board_raw(b, stock_ind)
    idx = data.load_index(b).set_index("date")
    for col in ("open", "high", "low"):
        raw[col] = idx[col].reindex(raw.index)
    return raw


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


def entries(s, c, thr=-6.0, cooldown_in_warmup=True):
    dates = list(s.index)
    s = s.to_numpy(float); cl = c.to_numpy(float)
    out = []; last = -10 ** 9; n = len(s)
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if s[i - 1] <= thr and s[i] > s[i - 1]:
            if not cooldown_in_warmup and dates[i] < EVAL_START:
                continue                      # 预热段信号完全不占位
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
RAW = {}
for b in C.BOARD_ORDER:
    RAW[C.BOARDS[b]["name"]] = raw_full(b, stock_ind)
for key, name in EXTRA_NAME.items():
    fi = os.path.join(OUT, f"{key}_index.parquet"); fc = os.path.join(OUT, f"{key}_cons.json")
    if not (os.path.exists(fi) and os.path.exists(fc)): continue
    idx = pd.read_parquet(fi)
    if len(idx) == 0: continue
    r = build_extra(key, idx, json.load(open(fc, encoding="utf-8")), stock_ind)
    if r is not None and len(r) > 200:
        RAW[name] = r
print("  板块", len(RAW), " 面板起点", min(r.index.min() for r in RAW.values()).date(), flush=True)

# 预计算各标定口径的分值
S = {}
for nm, raw in RAW.items():
    S[nm] = {"full": score_of(raw, pm_full),
             "expand": score_of(raw, pm_expand),
             "roll": score_of(raw, pm_roll)}
print("  分值计算完成\n", flush=True)


def run(cfg_name, key, trim_to_eval, cooldown_in_warmup):
    rows = []
    for nm, raw in RAW.items():
        sc_all = S[nm][key]
        cl_all = raw["close"]
        if trim_to_eval:                       # 只保留评估段（726 行口径）
            sc = sc_all[sc_all.index >= EVAL_START]
            cl = cl_all[cl_all.index >= EVAL_START]
            # 注意：trim 后百分位仍是全样本算的（与生产一致）
            sc = score_of(raw[raw.index >= EVAL_START], pm_full) if key == "full" else sc
            cl = raw["close"][raw["close"].index >= EVAL_START]
        else:
            sc, cl = sc_all, cl_all
        for d, r, m, v in entries(sc, cl, cooldown_in_warmup=cooldown_in_warmup):
            rows.append((nm, d, r, m, v))
    return rows


CFG = [
    ("A1 726行+全样本(生产)", "full", True, True),
    ("A2 803行+全样本", "full", False, True),
    ("A3 803行+全样本(预热不吃冷却)", "full", False, False),
    ("B1 803行+扩展窗口(真因果)", "expand", False, True),
    ("B2 803行+滚动250", "roll", False, True),
]

print("=" * 124)
print("隔离测试")
print("=" * 124)
print(f"  {'配置':<30}{'样本':>5}{'命中':>6}{'命中率':>8}{'未被套':>7}{'未被套率':>9}"
      f"{'均T+20':>9}{'最差回撤':>10}{'威尔逊下界':>10}")
for tag, key, trim, cw in CFG:
    rows = run(tag, key, trim, cw)
    ok = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
    nt = sum(1 for x in rows if x[3] >= -TOL)
    print(f"  {tag:<30}{len(rows):>5}{ok:>6}{ok/len(rows)*100:>7.1f}%{nt:>7}"
          f"{nt/len(rows)*100:>8.1f}%{np.mean([x[2] for x in rows])*100:>+8.2f}%"
          f"{min(x[3] for x in rows)*100:>+9.2f}%{wilson(ok,len(rows))*100:>9.1f}%")
