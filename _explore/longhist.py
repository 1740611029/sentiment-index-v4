"""长历史（2013~2026）原始因子面板构建，带 parquet 缓存。

数据源拼接：新浪(2013~2023-06, qfq) × 常数比率 → 对齐 v1/v2 缓存(2022-09~今)。
广度用累加器计算（np.bincount），不建 1400 万行的大表。
成分股用**当前清单**回溯 —— 有幸存者偏差。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors

DEST = os.path.join(C.DATA_DIR, "cache", "stocks_hist")
IDEST = os.path.join(C.DATA_DIR, "cache", "index_hist")
OUT = os.path.join(C.DATA_DIR, "cache", "longhist")
os.makedirs(OUT, exist_ok=True)

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
    """新浪段(乘常数比率对齐) + v1/v2 缓存"""
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


def _boards():
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
        BC[k] = dict(name=name, idx=idx, codes=codes,
                     pos={d: i for i, d in enumerate(idx["date"])})
    return BC


def build(force=False):
    """返回 {board_key: raw DataFrame}，带缓存。"""
    cached = {k: os.path.join(OUT, f"{k}.parquet") for k in BOARDS}
    if not force and all(os.path.exists(v) for v in cached.values() if True) \
            and len([1 for v in cached.values() if os.path.exists(v)]) == len(BOARDS):
        return {k: pd.read_parquet(v) for k, v in cached.items()}

    print("准备板块 ...", flush=True)
    BC = _boards()
    print("  板块:", len(BC), flush=True)

    ACC = {k: dict(
        b20=np.zeros(len(v["idx"])), b60=np.zeros(len(v["idx"])), r5=np.zeros(len(v["idx"])),
        nh=np.zeros(len(v["idx"])), nl=np.zeros(len(v["idx"])),
        lu=np.zeros(len(v["idx"])), ld=np.zeros(len(v["idx"])),
        amt=np.zeros(len(v["idx"])), s1=np.zeros(len(v["idx"])),
        s2=np.zeros(len(v["idx"])), cnt=np.zeros(len(v["idx"])))
        for k, v in BC.items()}

    allcodes = sorted(set().union(*[v["codes"] for v in BC.values()]))
    print(f"  需处理股票 {len(allcodes)}", flush=True)

    done = 0
    for ci, code in enumerate(allcodes, 1):
        try:
            df = load_merged(code)
            if df is None or len(df) < 70 or "close" not in df:
                continue
            ind = data.compute_stock_indicators(df, code)
            ind["date"] = pd.to_datetime(ind["date"])
            hit = [k for k, v in BC.items() if code in v["codes"]]
            if not hit:
                continue
            for k in hit:
                v = BC[k]
                p = ind["date"].map(v["pos"])
                m = p.notna().to_numpy()
                if not m.any():
                    continue
                pos_i = p[m].astype(int).to_numpy()
                n = len(v["idx"])
                g = ACC[k]
                g["cnt"] += np.bincount(pos_i, minlength=n)
                w = m
                for col, key in (("above_ma20", "b20"), ("above_ma60", "b60"), ("up5", "r5"),
                                 ("is_nh60", "nh"), ("is_nl60", "nl"),
                                 ("any_lu5", "lu"), ("any_ld5", "ld")):
                    g[key] += np.bincount(pos_i, weights=ind[col].to_numpy()[w], minlength=n)
                g["amt"] += np.bincount(pos_i, weights=ind["amount"].to_numpy()[w], minlength=n)
                rv = np.where(np.isnan(ind["ret5"].to_numpy()[w]), 0.0, ind["ret5"].to_numpy()[w])
                g["s1"] += np.bincount(pos_i, weights=rv, minlength=n)
                g["s2"] += np.bincount(pos_i, weights=rv * rv, minlength=n)
            done += 1
        except Exception:
            continue
        if ci % 500 == 0:
            print(f"    ... {ci}/{len(allcodes)}  有效 {done}", flush=True)
    print(f"  个股处理完成，有效 {done}\n", flush=True)

    res = {}
    for k, v in BC.items():
        g = ACC[k]; idx = v["idx"]; n = len(idx)
        cnt = g["cnt"]; ok = cnt >= 20

        def safe(a):
            out = np.full(n, np.nan)
            out[ok] = a[ok] / cnt[ok]
            return out

        raw = pd.DataFrame(index=pd.DatetimeIndex(idx["date"]))
        raw["close"] = idx["close"].astype(float).to_numpy()
        raw["b20"] = safe(g["b20"]); raw["b60"] = safe(g["b60"]); raw["r5"] = safe(g["r5"])
        raw["nh"] = safe(g["nh"] - g["nl"]); raw["lim"] = safe(g["lu"] - g["ld"])
        amt = np.where(ok, g["amt"], np.nan)
        var = np.full(n, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            var[ok] = (g["s2"][ok] - g["s1"][ok] ** 2 / cnt[ok]) / np.maximum(cnt[ok] - 1, 1)
        raw["disp"] = np.sqrt(np.where(var > 0, var, np.nan))
        raw["amt_series"] = amt
        raw["amt_pct"] = pd.Series(amt, index=raw.index).rolling(250, min_periods=60).rank(pct=True)
        close = raw["close"]
        raw["rsi"] = factors._rsi(close, 14)
        raw["bias"] = close / close.rolling(60).mean() - 1.0
        raw["ret20"] = close.pct_change(20)
        raw = raw.drop(columns=["amt_series"])
        res[v["name"]] = raw
        raw.to_parquet(cached[k])
        print(f"  {v['name']:<8} {len(raw)} 行  已存 {os.path.basename(cached[k])}", flush=True)
    return res


if __name__ == "__main__":
    build(force=True)
