"""d86 —— 个股层面的**短期**广度因子。

为什么做这个：项目里已经验证过一次 —— SENTI-1 的底部命中率从 66% 提到 89%，
靠的不是指数技术面，而是**成分股广度**。广度是"市场内部有多虚"，
指数技术面只是"价格跌了多少"，前者信息量大得多。

现有 stock_ind 只有 5 日/20 日/60 日尺度的广度，对 4~10 天的小波段太粗。
这里补 1 日 / 3 日 / 10 日尺度的：
  dn1  当日下跌个股占比
  dn3  近 3 日下跌个股占比
  b5   站上 MA5 占比
  b10  站上 MA10 占比
  dn1run 连跌 3 天以上的个股占比（"普跌且持续"）

实现：stock_ind 里没有 close，只有 ret1。用 (1+ret1).cumprod() 还原相对价格序列，
MA 只关心相对形状，不需要真实价格。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C, data

CACHE = os.path.join(C.CACHE_DIR, "breadth_short.parquet")


def build(force=False) -> pd.DataFrame:
    """返回长表 date, code, dn1, dn3, b5, b10, deep3（0/1 标记与占比）。"""
    if os.path.exists(CACHE) and not force:
        return pd.read_parquet(CACHE)
    si = data.build_stock_indicators()
    si = si.sort_values(["code", "date"]).reset_index(drop=True)

    p = (1.0 + si["ret1"].astype(float)).groupby(si["code"]).cumprod()
    si = si.assign(_p=p)
    g = si.groupby("code")["_p"]
    si["ma5"] = g.transform(lambda s: s.rolling(5).mean())
    si["ma10"] = g.transform(lambda s: s.rolling(10).mean())
    si["ret3"] = g.transform(lambda s: s.pct_change(3))

    out = pd.DataFrame({
        "date": si["date"], "code": si["code"],
        "dn1": (si["ret1"] < 0).astype("float32"),
        "dn3": (si["ret3"] < 0).astype("float32"),
        "b5": (si["_p"] > si["ma5"]).astype("float32"),
        "b10": (si["_p"] > si["ma10"]).astype("float32"),
    })
    out = out[out["b10"].notna()]
    out.to_parquet(CACHE, index=False)
    print(f"短期广度长表已缓存: rows={len(out)}")
    return out


def board_breadth(board_key: str, wide: pd.DataFrame | None = None) -> pd.DataFrame:
    """聚合为板块每日短期广度。"""
    if wide is None:
        wide = build()
    codes = set(data.load_universe(board_key))
    sub = wide[wide["code"].isin(codes)]
    g = sub.groupby("date", sort=True)
    b = pd.DataFrame({
        "dn1": g["dn1"].mean(),
        "dn3": g["dn3"].mean(),
        "b5": g["b5"].mean(),
        "b10": g["b10"].mean(),
    })
    b.index = pd.to_datetime(b.index)
    return b


if __name__ == "__main__":
    import time
    t0 = time.time()
    w = build()
    print(f"耗时 {time.time()-t0:.0f}s")
    for b in C.BOARD_ORDER:
        bb = board_breadth(b, w)
        print(f"{C.BOARDS[b]['name']:<9} rows={len(bb)}  {bb.index.min().date()} ~ {bb.index.max().date()}")
    print()
    bb = board_breadth("STAR", w).loc["2026-08-20":]
    print("科创板短期广度（近段）:")
    print((bb * 100).round(1).to_string())
