"""外部数据源：全市场融资融券（东财数据中心）。

融资净买入是「真实杠杆」数据：恐慌底往往伴随融资盘被迫去杠杆（净买入大幅为负），
这是价格/广度因子看不到的信息。
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from . import config as C

CACHE = os.path.join(C.CACHE_DIR, "margin.parquet")


def fetch_margin(force: bool = False) -> pd.DataFrame:
    """返回 date, rzye(融资余额), rzjme(融资净买入), rzjme5, rzjme10, rzyezb(占流通市值比)。"""
    if os.path.exists(CACHE) and not force:
        return pd.read_parquet(CACHE)

    s = requests.Session()
    s.trust_env = False
    rows = []
    page = 1
    while True:
        url = ("https://datacenter-web.eastmoney.com/api/data/v1/get"
               "?reportName=RPTA_RZRQ_LSHJ&columns=ALL&sortColumns=dim_date&sortTypes=-1"
               f"&pageSize=500&pageNumber={page}&source=WEB&client=WEB")
        r = s.get(url, timeout=30)
        js = r.json()
        res = js.get("result") or {}
        data = res.get("data") or []
        if not data:
            break
        rows.extend(data)
        total = res.get("count", 0)
        if len(rows) >= total:
            break
        page += 1
        if page > 20:
            break

    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["DIM_DATE"])
    keep = {"RZYE": "rzye", "RZJME": "rzjme", "RZJME5D": "rzjme5",
            "RZJME10D": "rzjme10", "RZYEZB": "rzyezb"}
    d = d[["date"] + list(keep)].rename(columns=keep)
    for c in keep.values():
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    d.to_parquet(CACHE, index=False)
    print(f"融资融券历史已缓存: {CACHE}  rows={len(d)}  {d['date'].min().date()} ~ {d['date'].max().date()}")
    return d


def margin_factors(d: pd.DataFrame) -> pd.DataFrame:
    """由融资余额派生情绪因子（无未来函数，只用当日及历史）。"""
    out = pd.DataFrame(index=d["date"])
    bal = d.set_index("date")["rzye"]
    net = d.set_index("date")["rzjme"]
    net5 = d.set_index("date")["rzjme5"]
    net10 = d.set_index("date")["rzjme10"]

    # 净买入 / 余额 —— 归一化，避免绝对值随市值增长
    out["mg_net"] = (net / bal).reindex(out.index)
    out["mg_net5"] = (net5 / bal).reindex(out.index)
    out["mg_net10"] = (net10 / bal).reindex(out.index)
    # 余额 20 日变化（去杠杆速度）
    out["mg_bal20"] = (bal / bal.shift(20) - 1).reindex(out.index)
    # 余额在自身 250 日区间的位置
    out["mg_pos"] = bal.rolling(250, min_periods=60).rank(pct=True).reindex(out.index)
    return out
