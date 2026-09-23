"""数据层：加载指数日线、成分股清单、个股日线（长历史 + v1/v2 缓存），
并预计算个股级指标长表（供各板块广度因子复用）。

个股长历史来自新浪前复权（锚定 2023-06），与 v1/v2 缓存（锚定当前）的复权基准不同。
qfq 调整是乘性的，所以用重叠期求一个常数比率对齐；比率不稳定（期间有分红送转）时
放弃该股的长历史，只用缓存段。
"""
from __future__ import annotations

import os
import glob
import warnings

import numpy as np
import pandas as pd

from . import config as C

warnings.filterwarnings("ignore")


MKT_DIR = os.path.join(C.CACHE_DIR, "market")
# 本项目自己抓的个股增量（`run.py refresh` 写入，见 fetch.py）。
# 上游 v1/v2 缓存是**只读复用**的、时间停在他们最后一次更新；
# 这一段是唯一会随时间前进的数据。
DELTA_DIR = os.path.join(C.CACHE_DIR, "stocks_delta")
BNA_PCT_MIN = 60.0     # 「估值已投降」的分位门槛，见 store.py 里的说明


def load_bna_pct() -> pd.Series | None:
    """破净股占比的 250 日因果分位（0~100，越大 = 破净越严重 = 估值侧越投降）。

    这是**价量之外**的信息源，用来区分「真投降」和「杠杆强平」：
    价量在两者上长得几乎一样，但估值侧不会说谎。

    为什么必须用**分位**而不是绝对值：破净率的绝对水平有强烈的时代偏移
    （2015 年最高只有 1.76%，2024 年能到 16.64%），用分位才能跨时代可比。

    时序：破净率依赖季报净资产 + 当日收盘价，保守起见整体 shift(1)。

    抓不到时返回 None（调用方必须能优雅降级）。
    """
    p = os.path.join(MKT_DIR, "bna.parquet")
    if not os.path.exists(p):
        return None
    try:
        s = pd.read_parquet(p).set_index("date")["bna"].astype(float)
        s = s[~s.index.duplicated(keep="last")].sort_index().shift(1)
        return s.rolling(250, min_periods=120).rank(pct=True) * 100.0
    except Exception:
        return None


def load_margin_pct() -> pd.Series | None:
    """沪深融资余额的 250 日因果分位（0~100，越大 = 杠杆越挤满）。

    与破净率严格对称：
      底部要「杠杆已出清」→ 看破净率分位（高 = 投降）
      顶部要「杠杆已挤满」→ 看融资余额分位（高 = 狂热）
    2015 年 6 月那个真正的顶，融资余额 2.1 万亿是历史极值，正是分位能吃到的信号。

    时序：两融数据次日早上才公布 → 整体 shift(1)。
    抓不到时返回 None。
    """
    p = os.path.join(MKT_DIR, "margin.parquet")
    if not os.path.exists(p):
        return None
    try:
        s = pd.read_parquet(p).set_index("date")["margin"].astype(float)
        s = s[~s.index.duplicated(keep="last")].sort_index().shift(1)
        return s.rolling(250, min_periods=120).rank(pct=True) * 100.0
    except Exception:
        return None


# ---------------------------------------------------------------- 指数
def load_index(board_key: str) -> pd.DataFrame:
    """返回 date/open/high/low/close/volume/amount，按日期升序。

    优先用 index_long（含锚点预热段）；没有则回退到 v2 缓存的 3 年段。
    """
    fl = os.path.join(C.CACHE_DIR, "index_long", f"{board_key}.parquet")
    f = fl if os.path.exists(fl) else os.path.join(
        C.UPSTREAM["v2_index"], f"{C.BOARDS[board_key]['index']}.parquet")
    df = pd.read_parquet(f)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------- 成分股
def load_universe(board_key: str) -> list[str]:
    f = os.path.join(C.UPSTREAM["v2_universe"], f"{C.BOARDS[board_key]['uni']}.parquet")
    df = pd.read_parquet(f)
    return sorted(df["code"].astype(str).str.zfill(6).unique().tolist())


# ---------------------------------------------------------------- 个股日线
def _stock_files() -> dict[str, tuple[str, str, str]]:
    """code -> (v1_path, v2_path, delta_path)，任一可为 None。

    v1 / v2 = 上游只读缓存；delta = 本项目 `run.py refresh` 抓到的最新增量。
    优先级由 load_stock 决定：delta > v2 > v1。
    """
    out: dict[str, tuple] = {}
    for tag, key in (("v1", "v1_stocks"), ("v2", "v2_stocks")):
        for p in glob.glob(os.path.join(C.UPSTREAM[key], "*.parquet")):
            code = os.path.splitext(os.path.basename(p))[0].zfill(6)
            a, b, c = out.get(code, (None, None, None))
            out[code] = (p, b, c) if tag == "v1" else (a, p, c)
    for p in glob.glob(os.path.join(DELTA_DIR, "*.parquet")):
        code = os.path.splitext(os.path.basename(p))[0].zfill(6)
        a, b, _ = out.get(code, (None, None, None))
        out[code] = (a, b, p)
    return out


HIST_DIR = os.path.join(C.CACHE_DIR, "stocks_hist")


def _load_hist(code: str):
    """新浪长历史（个股日线），不存在则返回 None。"""
    p = os.path.join(HIST_DIR, f"{code}.parquet")
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p)
    d["date"] = pd.to_datetime(d["date"])
    return d


def _ratio_to(base: pd.Series, other: pd.Series, min_rows: int = 5) -> float | None:
    """重叠期的常数比率 other → base（前复权基准不同，qfq 是乘性调整）。

    比率不稳定（期间有分红送转 / 源换了口径）时返回 None，调用方应放弃拼接。
    """
    j = pd.concat([base.rename("a"), other.rename("b")], axis=1).dropna()
    if len(j) < min_rows:
        return None
    r = (j["a"] / j["b"]).replace([np.inf, -np.inf], np.nan).dropna()
    if not len(r) or not r.mean() or (r.std() / abs(r.mean())) > 0.02:
        return None
    return float(r.median())


def load_stock(code: str, files: dict | None = None) -> pd.DataFrame:
    """合并 delta + v1 + v2 + 长历史，越新越优先。

    两种拼接都要做**常数比率对齐**，因为前复权基准不同：
      · 长历史（锚定 2023-06）→ 缓存口径
      · delta（锚定抓取当日）→ 缓存口径
    比率不稳定时放弃该段。delta 只追加「比缓存更新的日期」，不覆盖历史 ——
    覆盖会让已发布的分数随抓取日漂移。
    """
    files = files or _stock_files()
    v1, v2, dl = files.get(code, (None, None, None))
    frames = []
    for p in (v1, v2):
        if p and os.path.exists(p):
            d = pd.read_parquet(p)
            d["date"] = pd.to_datetime(d["date"])
            frames.append(d)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    else:
        df = pd.DataFrame()

    # ---- 本项目抓的增量：只追加，并按重叠期比率对齐到缓存口径 ----
    if dl and os.path.exists(dl):
        d = pd.read_parquet(dl)
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        if len(df) == 0:
            df = d
        else:
            cut = df["date"].max()
            r = _ratio_to(df.set_index("date")["close"].astype(float),
                          d.set_index("date")["close"].astype(float))
            if r is not None and abs(r - 1.0) > 1e-9:
                for c in ("open", "high", "low", "close"):
                    if c in d.columns:
                        d[c] = d[c] * r
            new = d[d["date"] > cut]
            if len(new):
                df = pd.concat([df, new], ignore_index=True)
                df = (df.sort_values("date").drop_duplicates("date", keep="last")
                        .reset_index(drop=True))

    if len(df) == 0:
        return df

    h = _load_hist(code)
    if h is None or len(h) == 0:
        return df
    cut = df["date"].min()

    # 重叠期求常数比率（qfq 是乘性调整）
    lo = max(h["date"].min(), df["date"].min())
    hi = min(h["date"].max(), df["date"].max())
    ratio = None
    if lo < hi:
        A = h[(h["date"] >= lo) & (h["date"] <= hi)].set_index("date")["close"].astype(float)
        B = df[(df["date"] >= lo) & (df["date"] <= hi)].set_index("date")["close"].astype(float)
        j = pd.concat([A, B], axis=1, keys=["h", "v"]).dropna()
        if len(j) >= 30:
            r = (j["v"] / j["h"]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(r) and r.mean() and (r.std() / r.mean()) < 0.02:
                ratio = float(r.median())
    if ratio is None:
        return df

    pre = h[h["date"] < cut].copy()
    if len(pre) == 0:
        return df
    if abs(ratio - 1.0) > 1e-6:
        for c in ("open", "high", "low", "close"):
            if c in pre.columns:
                pre[c] = pre[c] * ratio
    out = pd.concat([pre, df], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


# ---------------------------------------------------------------- 个股指标
def _limit_pct(code: str) -> float:
    """涨跌幅停阈值：创业板/科创板 20%，其余 10%。"""
    if code.startswith(("688", "689", "300", "301")):
        return 0.195
    return 0.095


def compute_stock_indicators(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """输入个股日线，输出个股级指标（与 df 等长）。"""
    c = df["close"].astype(float)
    v = df["amount"].astype(float)

    out = pd.DataFrame({"date": df["date"].values})
    out["ma20"] = c.rolling(20).mean().values
    out["ma60"] = c.rolling(60).mean().values

    out["above_ma20"] = (c > out["ma20"]).astype("float32").values
    out["above_ma60"] = (c > out["ma60"]).astype("float32").values

    out["ret1"] = c.pct_change().values
    out["ret5"] = c.pct_change(5).values
    out["ret20"] = c.pct_change(20).values

    hi60 = c.rolling(60).max()
    lo60 = c.rolling(60).min()
    out["is_nh60"] = (c >= hi60).astype("float32").values
    out["is_nl60"] = (c <= lo60).astype("float32").values

    lim = _limit_pct(code)
    up = (out["ret1"] >= lim).astype("float32")
    dn = (out["ret1"] <= -lim).astype("float32")
    lu = up.rolling(5).sum()      # 近 5 日涨停次数
    ld = dn.rolling(5).sum()
    # 预计算 0/1 标记，聚合时直接 mean 即可（避免 groupby.apply 慢路径）
    out["up5"] = (out["ret5"] > 0).astype("float32").values
    out["any_lu5"] = (lu > 0).astype("float32").values
    out["any_ld5"] = (ld > 0).astype("float32").values

    out["amount"] = v.values
    return out


def build_stock_indicators(force: bool = False) -> pd.DataFrame:
    """构建（并缓存）全市场个股指标长表：date, code, + 各指标列。"""
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    cache = os.path.join(C.CACHE_DIR, "stock_ind.parquet")
    if os.path.exists(cache) and not force:
        return pd.read_parquet(cache)

    files = _stock_files()
    frames = []
    for i, code in enumerate(sorted(files)):
        raw = load_stock(code, files)
        if raw is None or len(raw) < 70 or "close" not in raw:
            continue
        try:
            ind = compute_stock_indicators(raw, code)
        except Exception:
            continue
        ind["code"] = code
        frames.append(ind)
        if (i + 1) % 1000 == 0:
            print(f"  ... 个股指标 {i + 1}/{len(files)}")
    big = pd.concat(frames, ignore_index=True)
    # 只保留有 ma60 之后的行（前面无法用于广度统计）
    big = big[big["ma60"].notna()]
    # 锚点窗口只需有限预热，更早的历史只增加内存开销
    big = big[big["date"] >= pd.Timestamp(C.HIST_START)]
    for col in big.columns:
        if col not in ("date", "code"):
            big[col] = big[col].astype("float32")
    big.to_parquet(cache, index=False)
    print(f"个股指标长表已缓存: {cache}  rows={len(big)}  stocks={big['code'].nunique()}")
    return big
