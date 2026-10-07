"""盘中预览：手动触发的「假设现在收盘」分值计算。

红线（铁律八）：本模块只在内存里拼「临时今天」，**绝不写 data/ 下任何文件**。
盘中值只是预览：不产生信号、不进事件统计、不触发冷却。

口径：与正式面板完全相同的模型公式与因果锚 —— 直接复用
model / swing / swing2 / swing3 的 build()，区别只在注入的数据：
  ① 指数序列末尾多一行「临时今天」（快照价/量/额；中证2000 由个股快照等权合成）
  ② 个股指标长表末尾多每个成分股的「临时今天」行
两个注入都通过临时替换模块属性完成（data.load_index / factors.build_board_raw），
在锁内执行、finally 恢复，不会影响正式的 refresh/update 链路。

「临时今天」的个股指标行不再逐只读 parquet，而是从 stock_ind 长表的 ret1 链
反推最近 60 个收盘价（以快照昨收为锚），镜像 data.compute_stock_indicators
的单日逻辑；窗口内缺数据（停牌、新股）的股票当天自动缺席广度统计。
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pandas as pd

from . import config as C, data, factors, fetch, model, swing, swing2, swing3

TTL_SEC = 60                  # 进程内缓存：60 秒内重复请求直接复用
STOCK_TAIL = 60               # 个股反推窗口（ma60 / 60 日新高新低需要 60 天）

_cache = {"at": 0.0, "payload": None}
_lock = threading.Lock()
_stock_ind = {"mtime": 0.0, "df": None}


def _load_stock_ind() -> pd.DataFrame:
    """读个股指标长表缓存（按文件 mtime 在进程内复用，避免每次点击都读盘）。"""
    fp = os.path.join(C.CACHE_DIR, "stock_ind.parquet")
    mt = os.path.getmtime(fp) if os.path.exists(fp) else 0.0
    if _stock_ind["df"] is None or _stock_ind["mtime"] != mt:
        _stock_ind["df"] = data.build_stock_indicators()
        _stock_ind["mtime"] = mt
    return _stock_ind["df"]


def _temp_indices(idx_snap: pd.DataFrame, stk_snap: pd.DataFrame,
                  day: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """各板块指数序列 + 末尾一行「临时今天」（内存对象，不落盘）。

    day 取快照的交易日（不是自然日今天）：假期里点刷新，快照是哪个交易日，
    临时行就落在哪个交易日 —— 缓存里已有的同一天会被替换而不是重复。
    """
    snap = idx_snap.set_index("board")
    frames: dict[str, pd.DataFrame] = {}
    for b in fetch.EM_INDEX:
        loc = data.load_index(b)
        loc = loc[loc["date"] < day]
        r = snap.loc[b]
        row = {"date": day, "open": r["open"], "high": r["high"], "low": r["low"],
               "close": r["close"], "volume": r["volume"], "amount": r["amount"]}
        frames[b] = pd.concat([loc, pd.DataFrame([row])], ignore_index=True)

    # 中证2000：成分股快照涨跌幅均值（按涨跌停 clip）→ 链式累乘，与 fetch.extend_csi2000 同口径
    b = fetch.SYNTH_BOARD
    loc = data.load_index(b)
    loc = loc[loc["date"] < day]
    uni = set(data.load_universe(b))
    sub = stk_snap[stk_snap["code"].isin(uni)]
    if len(sub) >= fetch.SYNTH_MIN_STOCKS:
        lim = sub["code"].map(fetch.limit_pct) + fetch.LIMIT_TOL
        r = float(sub["chg"].clip(-lim, lim).mean())
        lvl = float(loc["close"].iloc[-1]) * (1.0 + r)
        row = {"date": day, "open": lvl, "high": lvl, "low": lvl, "close": lvl,
               "volume": np.nan, "amount": float(sub["amount"].sum())}
        loc = pd.concat([loc, pd.DataFrame([row])], ignore_index=True)
    # 成分股不足时不追加（盘中值退化为缓存末日值，比错值好）
    frames[b] = loc
    return frames


def _today_stock_rows(stock_ind: pd.DataFrame, stk_snap: pd.DataFrame,
                      day: pd.Timestamp) -> pd.DataFrame:
    """每个成分股的「临时今天」指标行（镜像 data.compute_stock_indicators 单日逻辑）。

    从 stock_ind 的 ret1 链反推最近 STOCK_TAIL 个收盘价（锚 = 快照昨收），
    再与快照价合并计算。窗口内任何一天缺数据的股票整只跳过（NaN 传播），
    与 build_board_raw 的「样本 <20 才置 NaN」质量门配合。
    """
    dates = np.sort(stock_ind["date"].unique())[-STOCK_TAIL:]
    sub = stock_ind[stock_ind["date"] >= dates[0]]
    R = sub.pivot(index="date", columns="code", values="ret1").sort_index()
    R = R.reindex(dates)

    snap = stk_snap.set_index("code")
    common = R.columns.intersection(snap.index)
    if len(common) < 20:
        raise RuntimeError(f"快照与指标长表交集过小: {len(common)}")
    R = R[common]
    prev = snap.loc[common, "prev"].to_numpy(dtype=float)

    # 反推收盘价：close_d = prev / prod_{e>d}(1+ret1_e)，最后一天 = prev
    G = 1.0 + R.to_numpy(dtype=float)
    P = np.cumprod(G[::-1], axis=0)[::-1]          # 后缀积 P_d = prod_{e>=d} G_e
    Pnxt = np.vstack([P[1:], np.ones((1, P.shape[1]))])
    closes = prev[None, :] / Pnxt                  # T×N，T-1 日 = prev
    C60 = np.vstack([closes, snap.loc[common, "close"].to_numpy(float)[None, :]])

    with np.errstate(invalid="ignore"):
        ma20 = np.nanmean(C60[-20:], axis=0)
        ma60 = np.nanmean(C60[-60:], axis=0)
        hi60 = np.nanmax(C60[-60:], axis=0)
        lo60 = np.nanmin(C60[-60:], axis=0)
    today_c = C60[-1]
    ret5 = today_c / C60[-6] - 1.0
    ret20 = today_c / C60[-21] - 1.0
    ret1 = snap.loc[common, "chg"].to_numpy(dtype=float)

    lim = np.array([data._limit_pct(c) for c in common])
    Rh = R.to_numpy(dtype=float)[-4:]              # t-4 .. t-1 的 ret1
    up5 = np.nansum((Rh >= lim[None, :]), axis=0) + (ret1 >= lim) > 0
    dn5 = np.nansum((Rh <= -lim[None, :]), axis=0) + (ret1 <= -lim) > 0

    # 窗口不完整（停牌/新股/长表缺尾）的股票当天缺席
    ok = ~np.isnan(C60).any(axis=0)

    out = pd.DataFrame({
        "date": day, "code": common,
        "ma20": ma20, "ma60": ma60,
        "above_ma20": (today_c > ma20).astype(float),
        "above_ma60": (today_c > ma60).astype(float),
        "ret1": ret1, "ret5": ret5, "ret20": ret20,
        "is_nh60": (today_c >= hi60).astype(float),
        "is_nl60": (today_c <= lo60).astype(float),
        "up5": (ret5 > 0).astype(float),
        "any_lu5": up5.astype(float), "any_ld5": dn5.astype(float),
        "amount": snap.loc[common, "amount"].to_numpy(dtype=float),
    })
    return out[ok].reset_index(drop=True)


def _compute() -> dict:
    idx_snap, ts_i = fetch.fetch_index_snapshot()
    stk_snap, ts_s = fetch.fetch_stock_snapshot(fetch.needed_codes())
    day = pd.Timestamp(ts_i.split(" ")[0])

    frames = _temp_indices(idx_snap, stk_snap, day)
    stock_ind = _load_stock_ind()
    if stock_ind["date"].max() < day:
        extra = _today_stock_rows(stock_ind, stk_snap, day)
        stock_ind = pd.concat([stock_ind, extra], ignore_index=True)

    # 临时替换两个注入点；build_board_raw 做按板块备忘（model 与 swing 各调一次，省一半聚合）
    orig_index, orig_raw = data.load_index, factors.build_board_raw
    raw_memo: dict[str, pd.DataFrame] = {}
    data.load_index = lambda b: frames[b]
    factors.build_board_raw = lambda b, s=None: raw_memo.setdefault(b, orig_raw(b, stock_ind))
    try:
        boards: dict[str, dict] = {}
        for b in C.BOARD_ORDER:
            row = {}
            fns = {
                "senti": lambda: model.build(b, stock_ind)["score"].iloc[-1],
                "swing": lambda: swing.build(b, stock_ind)["swing"].iloc[-1],
                "swing2": lambda: swing2.build(b)["swing2"].iloc[-1],
                "swing3": lambda: swing3.build(b)["swing3"].iloc[-1],
            }
            for k, fn in fns.items():
                try:
                    v = float(fn())
                    row[k] = round(v, 1) if np.isfinite(v) else None
                except Exception:
                    row[k] = None        # 单模型失败不影响其它模型
            boards[b] = row
    finally:
        data.load_index, factors.build_board_raw = orig_index, orig_raw

    # live = 点击时正处在交易时段（快照日=今天 且 09:25~15:05）。
    # 节假日/收盘后点击为 False，前端据此一直显示占位符「—」。
    now = pd.Timestamp.now()
    mins = now.hour * 60 + now.minute
    live = (day == now.normalize()) and (9 * 60 + 25 <= mins <= 15 * 60 + 5)
    return {"ts": (ts_s or ts_i)[5:], "day": str(day.date()),
            "live": bool(live), "boards": boards}


def get_intraday() -> dict:
    """60 秒进程内缓存：重复点击不重抓、不重算（cached 标记供前端提示）。"""
    with _lock:
        if _cache["payload"] is not None and time.time() - _cache["at"] < TTL_SEC:
            return {**_cache["payload"], "cached": True}
        payload = _compute()
        _cache.update({"at": time.time(), "payload": payload})
        return {**payload, "cached": False}
