"""数据抓取层：把 6 个板块指数 + 成分股并集的日线增量抓到本地。

⚠️ 这是 senti/ 里**唯一**会联网的模块。
   model / data / factors / store 全部只读本地缓存 —— 所以 `run.py update`
   **永远不可能推进数据日期**：它只是把已有数据重算一遍，`built_at` 会变、
   `last_date` 不变。要让页面日期往前走，只能跑 `run.py refresh`。

数据源（2026-09-23 本环境实测）
--------------------------------
| 用途 | 源 | 结论 |
|---|---|---|
| 上证 / 科创50 / 创业板 / 中证1000 / 沪深300 | 新浪 `ak.stock_zh_index_daily` | 与本地 index_long 重叠 35 天，偏差 **0.0000%** → 可直接追加 |
| 中证2000 | **没有可用的免费源** | 新浪与腾讯的 `sh000932` 都不是中证2000（腾讯 09-17→09-18 涨 0.429%、本地涨 1.647%、官方涨 1.774%） |
| 个股日线 | 新浪 `ak.stock_zh_a_daily` | 0.6s/只，自带 amount 列 |

网络可达性实测：`hq.sinajs.cn` / `web.ifzq.gtimg.cn` / `push2.eastmoney.com`（**仅快照**）/
`datacenter-web.eastmoney.com` / `www.csindex.com.cn` 通；
`push2his.eastmoney.com` 与 `push2.eastmoney.com` 的 **kline** 接口被拒（RemoteDisconnected）。

为什么中证2000 要自己合成
--------------------------
本地这条序列**本来就不是中证2000指数**。上游（情绪指标2）配置写着
`source: em / fallback: synthetic`，而 `emo/data/synthetic.py` 的口径是
「成分股日涨跌幅**均值** → 按各股涨跌停 clip → 链式累乘 → 基期 1000」。
实测本地序列与官方 932000（中证官网）的日收益相关 **0.9899**、
比值在 0.42~0.47 之间缓慢漂移（官方 09-18 收 3192.21，本地收 1496.94）——
这正是「等权合成 vs 市值加权」的特征。
所以这里**照同样口径续算**，而不是把官方点位接上去（接上去会瞬间跳 2 倍）。
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import config as C, data

DELTA_DIR = os.path.join(C.CACHE_DIR, "stocks_delta")
IDX_DIR = os.path.join(C.CACHE_DIR, "index_long")
STOCK_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]

# 有免费源的 5 个板块（board_key -> 新浪 symbol）
SINA_INDEX = {
    "SH": "sh000001", "STAR": "sh000688", "CHINEXT": "sz399006",
    "CSI1000": "sh000852", "HS300": "sh000300",
}
SYNTH_BOARD = "CSI2000"        # 无免费源，走等权合成续算
SYNTH_MIN_STOCKS = 200         # 与上游 emo/data/synthetic.py 一致
LIMIT_TOL = 0.003              # 与上游 provider.LIMIT_TOL 一致
DEFAULT_WORKERS = 8
DEFAULT_LOOKBACK = 120         # 日历日：覆盖「本地缓存末日 → 今天」并留足重叠期做复权校准


# ---------------------------------------------------------------- 环境 / 工具
def prep_env() -> None:
    """清掉代理环境变量。akshare 在新浪/东财上遇到代理会直接连不上。"""
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "all_proxy", "ALL_PROXY"):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def _sym(code: str) -> str:
    return ("sh" if code.startswith("6") else "sz") + code


def limit_pct(code: str) -> float:
    """涨跌停幅度。与上游 emo/data/provider.limit_pct 一致（含北交所 10% 兜底）。"""
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return 0.10
    return 0.10


def needed_codes() -> list[str]:
    """6 个板块成分股的并集 —— 广度因子与中证2000 合成都需要它们。"""
    s: set[str] = set()
    for b in C.BOARD_ORDER:
        s |= set(data.load_universe(b))
    return sorted(s)


# ---------------------------------------------------------------- 指数
def fetch_indices(log=print) -> dict:
    """5 个有免费源的板块：只追加「比本地更新的日期」，绝不覆盖已有历史。

    为什么只追加：本地 index_long 的历史段是拼接/合成出来的（见
    `_explore/build_index_long.py`），拿新浪全序列去覆盖会把那段冲掉。
    追加前先做一次**衔接校验**：新浪在「本地最后一天」的值必须与本地一致，
    偏差 > 0.5% 就说明数据源换了口径 → 跳过并报警，而不是硬接。
    """
    prep_env()
    import akshare as ak

    out: dict[str, dict] = {}
    for b, sym in SINA_INDEX.items():
        fp = os.path.join(IDX_DIR, f"{b}.parquet")
        loc = pd.read_parquet(fp)
        loc["date"] = pd.to_datetime(loc["date"])
        loc = loc.sort_values("date").reset_index(drop=True)
        last = loc["date"].max()
        last_close = float(loc.loc[loc["date"] == last, "close"].iloc[0])
        try:
            d = ak.stock_zh_index_daily(symbol=sym)
        except Exception as e:
            out[b] = {"status": "fail", "err": f"{type(e).__name__}: {e}"}
            log(f"  {b:<8} 抓取失败 {type(e).__name__}")
            continue
        d["date"] = pd.to_datetime(d["date"])
        if "amount" not in d.columns:
            d["amount"] = d["volume"] * d["close"]
        chk = d[d["date"] == last]
        dev = None
        if len(chk):
            dev = abs(float(chk["close"].iloc[0]) / last_close - 1.0)
            if dev > 0.005:
                out[b] = {"status": "skip", "dev": dev}
                log(f"  {b:<8} ⚠️ 衔接校验不过：{last.date()} 新浪 "
                    f"{float(chk['close'].iloc[0]):.2f} vs 本地 {last_close:.2f}"
                    f"（偏差 {dev:.2%}）→ 跳过")
                continue
        new = d[d["date"] > last][STOCK_COLS].copy()
        if len(new) == 0:
            out[b] = {"status": "uptodate", "last": str(last.date())}
            log(f"  {b:<8} 已是最新（{last.date()}）")
            continue
        merged = pd.concat([loc[STOCK_COLS], new], ignore_index=True)
        merged = (merged.sort_values("date").drop_duplicates("date", keep="last")
                        .reset_index(drop=True))
        merged.to_parquet(fp, index=False)
        out[b] = {"status": "ok", "added": len(new),
                  "last": str(merged["date"].max().date()), "dev": dev}
        log(f"  {b:<8} +{len(new)} 行 → {merged['date'].max().date()}"
            f"（衔接偏差 {(dev or 0):.4%}）")
    return out


def synth_returns(frames: dict[str, pd.DataFrame], start) -> pd.Series:
    """成分股日涨跌幅的**截面均值**（按各股涨跌停 clip），照上游口径。

    上游注释特别强调：必须用**均值**不能用中位数 —— A股个股日涨跌幅右偏
    （少数大涨、多数小跌），中位数长期系统性为负（实测沪深300 三年样本中位数合成
    −53%、均值合成 +59.8%，后者与真实指数日收益相关 0.954）。
    也不要截尾：等权组合的收益本就等于成分股涨跌幅的算术平均。
    """
    rows = []
    for c, d in frames.items():
        if d is None or len(d) < 2:
            continue
        x = d[["date", "close"]].copy()
        x["code"] = c
        rows.append(x)
    if not rows:
        return pd.Series(dtype=float)
    ps = pd.concat(rows, ignore_index=True)
    ps["ret"] = ps.groupby("code")["close"].pct_change()
    lim = ps["code"].map(limit_pct) + LIMIT_TOL
    ps["ret"] = ps["ret"].clip(-lim, lim)
    agg = ps.groupby("date")["ret"].mean().fillna(0.0).sort_index()
    return agg[agg.index >= pd.Timestamp(start)]


def extend_csi2000(frames: dict[str, pd.DataFrame], cap, log=print) -> dict:
    """按等权合成口径把中证2000 续算到 `cap`（含）。

    只追加、不重算历史 —— 历史段是上游合成的，重算会因成分股清单变化而漂移。
    `cap` 取其它 5 个板块里最新的那个交易日：保证 6 个板块的日期轴一致，
    否则中证2000 会单独多出一天（个股数据比指数接口早一天更新）。
    """
    fp = os.path.join(IDX_DIR, f"{SYNTH_BOARD}.parquet")
    loc = pd.read_parquet(fp)
    loc["date"] = pd.to_datetime(loc["date"])
    loc = loc.sort_values("date").reset_index(drop=True)
    last = loc["date"].max()

    uni = set(data.load_universe(SYNTH_BOARD))
    sub = {c: d for c, d in frames.items() if c in uni}
    if len(sub) < SYNTH_MIN_STOCKS:
        log(f"  {SYNTH_BOARD:<8} ⚠️ 有效成分股 {len(sub)} < {SYNTH_MIN_STOCKS} → 跳过")
        return {"status": "too-few", "n": len(sub)}

    agg = synth_returns(sub, start=last)
    agg = agg[(agg.index > last) & (agg.index <= pd.Timestamp(cap))]
    if len(agg) == 0:
        log(f"  {SYNTH_BOARD:<8} 已是最新（{last.date()}）")
        return {"status": "uptodate", "last": str(last.date())}

    lvl = float(loc["close"].iloc[-1])
    rows = []
    for dt, r in agg.items():
        lvl *= (1.0 + float(r))
        day = [d for d in sub.values() if dt in set(d["date"])]
        amt = float(sum(float(d.loc[d["date"] == dt, "amount"].iloc[0])
                        for d in day if "amount" in d.columns)) if day else np.nan
        vol = float(sum(float(d.loc[d["date"] == dt, "volume"].iloc[0])
                        for d in day if "volume" in d.columns)) if day else np.nan
        rows.append({"date": dt, "open": lvl, "high": lvl, "low": lvl, "close": lvl,
                     "volume": vol, "amount": amt})
    new = pd.DataFrame(rows)[STOCK_COLS]
    merged = pd.concat([loc[STOCK_COLS], new], ignore_index=True)
    merged = (merged.sort_values("date").drop_duplicates("date", keep="last")
                    .reset_index(drop=True))
    merged.to_parquet(fp, index=False)
    log(f"  {SYNTH_BOARD:<8} +{len(new)} 行 → {merged['date'].max().date()}"
        f"（等权合成，{len(sub)} 只成分股，收 {lvl:.2f}）")
    return {"status": "ok", "added": len(new), "stocks": len(sub),
            "last": str(merged["date"].max().date())}


# ---------------------------------------------------------------- 个股
def fetch_stocks(codes: list[str], lookback: int = DEFAULT_LOOKBACK,
                 workers: int = DEFAULT_WORKERS, log=print):
    """抓个股日线增量 → `data/cache/stocks_delta/{code}.parquet`。

    抓的是「最近 lookback 个日历日」这一整段（不是只抓新增那天），因为：
      · 前复权基准会随除权除息漂移，需要重叠期来算常数比率（见 data.load_stock）
      · 顺带修掉上游缓存里可能存在的当日错值

    **不写上游缓存**（`D:\\情绪指标{,2}\\data\\cache\\stocks` 是只读复用的）。
    返回 (frames, stats)：frames 供中证2000 合成直接用，避免再读一遍盘。
    """
    prep_env()
    import akshare as ak

    os.makedirs(DELTA_DIR, exist_ok=True)
    end = pd.Timestamp.today()
    start = (end - pd.Timedelta(days=lookback)).strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")

    lock = threading.Lock()
    stat = {"ok": 0, "empty": 0, "fail": 0}
    frames: dict[str, pd.DataFrame] = {}
    fails: list[str] = []
    t0 = time.time()
    n = len(codes)

    def one(code: str):
        for attempt in range(3):
            try:
                d = ak.stock_zh_a_daily(symbol=_sym(code), start_date=start,
                                        end_date=end_s, adjust="qfq")
                if d is None or len(d) == 0:
                    return code, None, "empty"
                keep = [c for c in STOCK_COLS if c in d.columns]
                d = d[keep].copy()
                if "amount" not in d.columns:
                    d["amount"] = d["volume"] * d["close"]
                d["date"] = pd.to_datetime(d["date"])
                d = d.sort_values("date").drop_duplicates("date", keep="last")
                d.to_parquet(os.path.join(DELTA_DIR, f"{code}.parquet"), index=False)
                return code, d.reset_index(drop=True), "ok"
            except Exception:
                time.sleep(0.4 * (attempt + 1))
        return code, None, "fail"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, c) for c in codes]
        for i, f in enumerate(as_completed(futs), 1):
            code, d, st = f.result()
            with lock:
                stat[st] += 1
                if d is not None:
                    frames[code] = d
                if st == "fail":
                    fails.append(code)
            if i % 400 == 0 or i == n:
                el = time.time() - t0
                log(f"  {i}/{n}  ok={stat['ok']} empty={stat['empty']} "
                    f"fail={stat['fail']}  {el:.0f}s  剩余约 {el / i * (n - i):.0f}s")
    if fails:
        with open(os.path.join(DELTA_DIR, "_fails.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(fails, f)
    return frames, stat


def _align_boards(log=print):
    """把 6 个板块的日期轴对齐到最短的那个。

    为什么必须对齐：`store.resonance()` 把 6 条分值放进一张宽表按公共日期轴求和
    （`S <= ENTRY_THR).sum(axis=1)`），某个板块缺一天，那天该板块就是 NaN、
    比较结果恒为 False → **共振数会少算**（例如 6 个板块齐跌却只报 5）。
    共振数直接决定 S/A/B 等级，所以宁可截齐也不能让它偏小。

    各指数接口的发布时间不一致（实测 2026-09-23 17:55：
    上证 / 沪深300 已出 09-23，而创业板还停在 09-22），
    所以**晚上跑 refresh 经常只能更新到前一天** —— 这是正常的，不是坏了。

    截断掉的只是刚追加的最新 1~2 天，下一次 refresh 会自动补回来（幂等，不丢数据）。
    """
    last = {}
    for b in C.BOARD_ORDER:
        d = pd.read_parquet(os.path.join(IDX_DIR, f"{b}.parquet"))
        last[b] = pd.to_datetime(d["date"]).max()
    cap = min(last.values())
    if len(set(last.values())) == 1:
        return cap, {}
    behind = {b: v for b, v in last.items() if v > cap}
    log(f"  ⚠️ 各板块指数接口进度不一致，本次只能更新到 {cap.date()}：")
    for b in C.BOARD_ORDER:
        if last[b] > cap:
            log(f"     {C.BOARDS[b]['name']:<8} 接口已有 {last[b].date()}"
                f"（本地先截到 {cap.date()}）")
        elif last[b] == cap:
            log(f"     {C.BOARDS[b]['name']:<8} 接口最新 {last[b].date()} ← 最慢的，以它为准")
    for b in C.BOARD_ORDER:
        if last[b] > cap:
            fp = os.path.join(IDX_DIR, f"{b}.parquet")
            d = pd.read_parquet(fp)
            d["date"] = pd.to_datetime(d["date"])
            d[d["date"] <= cap].to_parquet(fp, index=False)
    log("     → 下次 refresh 会自动补上这几天，不需要其他操作。")
    return cap, {b: str(v.date()) for b, v in last.items()}


# ---------------------------------------------------------------- 编排
def refresh(lookback: int = DEFAULT_LOOKBACK, workers: int = DEFAULT_WORKERS,
            skip_stocks: bool = False, log=print) -> dict:
    """抓数 → 写本地增量缓存 → 对齐 6 板块日期轴。**不重建面板**（那是 run.py 的事）。"""
    prep_env()
    frames: dict[str, pd.DataFrame] = {}
    stat = {"ok": 0, "empty": 0, "fail": 0}

    if skip_stocks:
        log("① 个股日线：已跳过（--index-only）")
        frames, _ = _load_delta_frames()
    else:
        log(f"① 抓个股日线增量（{lookback} 个日历日，{workers} 线程）...")
        frames, stat = fetch_stocks(needed_codes(), lookback, workers, log)

    log("② 抓指数增量（新浪，5 个板块）...")
    idx = fetch_indices(log)

    log("③ 续算中证2000（等权合成）...")
    caps = [v["last"] for v in idx.values() if v.get("last")]
    cap = max(caps) if caps else str(pd.Timestamp.today().date())
    csi = extend_csi2000(frames, cap=cap, log=log)

    log("④ 对齐 6 板块日期轴 ...")
    cap, before = _align_boards(log)
    log(f"  数据最新交易日：{cap.date()}")

    return {"stocks": stat, "index": idx, "csi2000": csi,
            "cap": str(cap.date()), "before_align": before}


def _load_delta_frames() -> tuple[dict[str, pd.DataFrame], dict]:
    """读已有的 stocks_delta（--index-only 模式下给中证2000 合成用）。"""
    frames = {}
    for p in os.listdir(DELTA_DIR) if os.path.isdir(DELTA_DIR) else []:
        if not p.endswith(".parquet"):
            continue
        code = os.path.splitext(p)[0]
        d = pd.read_parquet(os.path.join(DELTA_DIR, p))
        d["date"] = pd.to_datetime(d["date"])
        frames[code] = d
    return frames, {"ok": len(frames)}
