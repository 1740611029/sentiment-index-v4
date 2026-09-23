"""s31 —— 第二十三轮：探查「价量之外」的新信息源在本环境的可达性。

为什么需要这一步
--------------
前三轮（因子广扫 → 第 4 模型收口 → 三层口径）已经把**价量因子**这条路走到收敛：
46 个价量因子 × 多确认层 × 多阈值，没有一个值得收编。
继续在价量里变换只是重排噪声。要再有增量，必须引入**价量之外的信息源**。

候选（都是「情绪/杠杆/估值」侧，理论上与 7~10 天波段底相关）
  1. 期权隐含波动率 QVIX（50ETF / 300ETF / 500ETF / 中证1000）—— 最直接的「恐慌定价」
  2. 国债收益率 / 股债利差（ERP = 沪深300 股息率 − 10Y 国债）
  3. 北向资金（注意：2024-08 起沪深港通改为按季披露，日频可能已断）
  4. AH 股溢价指数（恒生沪深港通 AH 溢价）
  5. 全 A 估值（PE/PB 分位）
  6. 破净股占比（项目已在用 bna_pct）
  7. 新成立基金份额 / 新增投资者（低频，可能只有月度）

本环境已知的网络限制（AGENTS.md）
  · 东财 push2 连不上；东财 datacenter-web 可用（margin 就是从那取的）
  · 新浪接口可用
所以逐个探测，记录「可达 / 不可达 / 起始日期 / 行数」，再决定测哪些。
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import akshare as ak                                # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s31_probe.csv")


def probe(name, fn, **kw):
    t0 = time.time()
    try:
        d = fn(**kw)
        dt = time.time() - t0
        if d is None or len(d) == 0:
            print(f"  ✘ {name:<34} 返回空")
            return None
        cols = list(d.columns)[:8]
        # 找日期列
        dcol = None
        for c in d.columns:
            if "date" in str(c).lower() or "日期" in str(c):
                dcol = c
                break
        span = ""
        if dcol is not None:
            v = pd.to_datetime(d[dcol], errors="coerce")
            span = f"{v.min().date()} ~ {v.max().date()}"
        print(f"  ✔ {name:<34} rows={len(d):<6} {dt:>5.1f}s  {span}")
        print(f"     列: {cols}")
        return {"name": name, "rows": len(d), "secs": round(dt, 1),
                "span": span, "cols": "|".join(map(str, cols))}
    except Exception as e:
        print(f"  ✘ {name:<34} {type(e).__name__}: {str(e)[:70]}")
        return {"name": name, "rows": 0, "secs": round(time.time() - t0, 1),
                "span": "", "cols": f"ERR {type(e).__name__}"}


CAND = [
    # ---- 1. 期权隐含波动率 ----
    ("QVIX 50ETF", lambda: ak.index_option_50etf_qvix()),
    ("QVIX 300ETF", lambda: ak.index_option_300etf_qvix()),
    ("QVIX 500ETF", lambda: ak.index_option_500etf_qvix()),
    ("QVIX 创业板ETF", lambda: ak.index_option_100etf_qvix()),
    ("QVIX 中证1000", lambda: ak.index_option_1000index_qvix()),
    ("QVIX 上证50指数", lambda: ak.index_option_50index_qvix()),
    ("QVIX 沪深300指数", lambda: ak.index_option_300index_qvix()),
    # ---- 2. 利率 / 股债 ----
    ("中美国债收益率", lambda: ak.bond_zh_us_rate(start_date="20130101")),
    ("LPR 报价", lambda: ak.macro_china_lpr()),
    # ---- 3. 北向 / 港股通 ----
    ("北向资金历史", lambda: ak.stock_hsgt_hist_em(symbol="北向资金")),
    ("港股通历史", lambda: ak.stock_hsgt_hist_em(symbol="港股通沪")),
    # ---- 4. AH 溢价 ----
    ("AH 溢价指数", lambda: ak.stock_zh_ah_spot_em()),
    # ---- 5. 全A估值 ----
    ("全A 估值(乐咕)", lambda: ak.stock_a_ttm_lyr()),
    ("A股平均市盈率", lambda: ak.stock_a_pe()),
    # ---- 6. 破净（项目已用） ----
    ("破净股统计", lambda: ak.stock_below_net_asset_statistics()),
    # ---- 7. 低频情绪 ----
    ("新成立基金份额", lambda: ak.fund_new_found_em()),
]


def main():
    print("=" * 96)
    print("外部数据源可达性探测")
    print("=" * 96)
    rows = []
    for name, fn in CAND:
        r = probe(name, fn)
        if r:
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n结果已存 {OUT}")
    ok = [r for r in rows if r["rows"] > 0]
    print(f"\n可达 {len(ok)}/{len(CAND)}：")
    for r in ok:
        print(f"  {r['name']:<34}{r['rows']:>7} 行   {r['span']}")


if __name__ == "__main__":
    main()
