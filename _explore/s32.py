"""s32 —— 第二十四轮：接入 QVIX 期权隐含波动率（第一个「价量之外」的信息源）

为什么是它
--------
前三轮把价量因子走到收敛（46 个因子 × 多确认层 × 多阈值，无一值得收编）。
QVIX（期权隐含波动率指数）是**市场对未来的恐慌定价**，
不是价格的函数 —— 它含有价量因子原理上拿不到的信息：
同样跌 5%，隐含波动率飙到 40 和稳在 18，是完全不同的两种市场。

数据（s31 探测）
  7 条序列，2015-02-09 ~ 2026-09-21，各 2817 行（东财 datacenter 可达）
  50ETF / 300ETF / 500ETF / 创业板ETF / 中证1000 / 上证50指数 / 沪深300指数
  **覆盖 2015-06 与 2016-01 两次崩盘** → 能过本项目唯一的历史否决关。

本脚本只做「取数 + 质检 + 建因子」，不做评估（评估在 s33）
A. 拉取并缓存（长表：date, code, close）
B. 数据质检：缺失、零值、异常跳变、与指数价格的相关性是否合理
C. 板块 → QVIX 序列映射
D. 建四个候选因子（全部走因果锚，无未来函数）：
     qvix_pct    QVIX 水平的因果锚分位（高 = 恐慌定价高）
     qvix_chg5   QVIX 的 5 日变化
     qvix_spike  QVIX / MA20 − 1（恐慌放大）
     vrp         QVIX − 20 日已实现波动率（方差风险溢价，真正的「恐慌溢价」）
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s8                                          # noqa: E402

import akshare as ak                               # noqa: E402

CACHE = os.path.join(s8.C.CACHE_DIR, "qvix.parquet")

SERIES = {
    "SH50ETF": ("QVIX 50ETF", ak.index_option_50etf_qvix),
    "SH300ETF": ("QVIX 300ETF", ak.index_option_300etf_qvix),
    "SH500ETF": ("QVIX 500ETF", ak.index_option_500etf_qvix),
    "CHINEXT": ("QVIX 创业板ETF", ak.index_option_100etf_qvix),
    "CSI1000": ("QVIX 中证1000", ak.index_option_1000index_qvix),
    "SH50IDX": ("QVIX 上证50指数", ak.index_option_50index_qvix),
    "HS300IDX": ("QVIX 沪深300指数", ak.index_option_300index_qvix),
}

# 板块 → QVIX 序列（选最贴近的标的）
BOARD_QVIX = {
    "SH": "SH50IDX", "CHINEXT": "CHINEXT", "CSI1000": "CSI1000", "HS300": "HS300IDX",
    "CSI500": "SH500ETF", "CSI800": "HS300IDX", "SH180": "SH50IDX",
    "SZ50": "SH50IDX", "CSI100": "HS300IDX",
    "STAR": "CHINEXT", "CSI2000": "CSI1000",
}


def fetch(force: bool = False) -> pd.DataFrame:
    if os.path.exists(CACHE) and not force:
        return pd.read_parquet(CACHE)
    rows = []
    for code, (name, fn) in SERIES.items():
        for attempt in range(3):
            try:
                d = fn()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  ✘ {name}: {type(e).__name__} {e}")
                    d = None
                time.sleep(1.0)
        if d is None or len(d) == 0:
            continue
        d = d.rename(columns={"date": "date", "close": "close"})[["date", "close"]]
        d["date"] = pd.to_datetime(d["date"])
        d["code"] = code
        rows.append(d)
        print(f"  ✔ {name:<18}{len(d):>6} 行  "
              f"{d['date'].min().date()} ~ {d['date'].max().date()}")
    out = pd.concat(rows, ignore_index=True).sort_values(["code", "date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out.to_parquet(CACHE, index=False)
    print(f"已缓存 {CACHE}  rows={len(out)}")
    return out


def wide(d: pd.DataFrame) -> pd.DataFrame:
    W = d.pivot(index="date", columns="code", values="close").sort_index()
    # 零值是脏数据（非交易日被写成 0），必须置 NaN，否则收益率出现 inf
    return W.replace(0.0, np.nan)


def main():
    print("=" * 100)
    print("A. 拉取 QVIX")
    print("=" * 100)
    d = fetch()
    W = wide(d)

    print("\n" + "=" * 100)
    print("B. 数据质检")
    print("=" * 100)
    print("  ⚠️ 各序列的起始日 = **对应期权合约的实际上市日**，不是数据缺失：")
    print("     50ETF 2015-02-09 / 300ETF 2019-12-23 / 中证1000 2022-07-22 /")
    print("     500ETF 与创业板ETF 2022-09-19 / 上证50指数 2023-01-03")
    print("     → **只有 50ETF QVIX 有完整长历史**（能过 2015-2016 关）\n")
    print(f"  {'序列':<12}{'有效行':>7}{'起始':>13}{'结束':>13}{'置零为NaN':>10}"
          f"{'最小':>8}{'中位':>8}{'最大':>8}{'最大单日跳变':>13}")
    for c in W.columns:
        v = W[c].dropna()
        jump = (v / v.shift(1) - 1).abs().max()
        print(f"  {c:<12}{len(v):>7}{str(v.index.min().date()):>13}"
              f"{str(v.index.max().date()):>13}{int((d[d.code == c]['close'] == 0).sum()):>10}"
              f"{v.min():>8.1f}{v.median():>8.1f}{v.max():>8.1f}{jump:>12.1%}")

    print("\n  合理性检查：QVIX 应与「指数 20 日实现波动」正相关")
    E = s8.Evaluator()
    print(f"  {'板块':<10}{'QVIX序列':<10}{'与20日实现波动相关':>20}{'样本':>8}")
    for b in s8.BOARDS:
        code = BOARD_QVIX.get(b)
        if code not in W.columns:
            continue
        q = W[code]
        c = E.CL[b]
        idx = q.dropna().index.intersection(c.index)
        if len(idx) < 100:
            continue
        rv = c.pct_change().rolling(20).std().reindex(idx)
        print(f"  {s8.NAME[b]:<10}{code:<10}{q.reindex(idx).corr(rv):>19.2f}{len(idx):>8}")

    print("\n" + "=" * 100)
    print("C. 建因子（因果锚，无未来函数）")
    print("=" * 100)
    print("  说明：非空率低是因为只有 50ETF 序列有长历史，其余只覆盖近 3~6 年；")
    print("       评估阶段必须按「该序列自身有数据的区间」分别统计，不能混在一起。")
    F = {}
    for b in s8.BOARDS:
        code = BOARD_QVIX.get(b)
        if code not in W.columns:
            continue
        q = W[code].reindex(E.CL[b].index).ffill()
        F[b] = pd.DataFrame(index=E.CL[b].index)
        F[b]["qvix"] = q
        F[b]["qvix_pct"] = s8.pm(q)                                   # 水平分位
        F[b]["qvix_chg5"] = q / q.shift(5) - 1                        # 5 日变化
        F[b]["qvix_spike"] = q / q.rolling(20).mean() - 1             # 相对 20 日均值
        rv = E.CL[b].pct_change().rolling(20).std() * np.sqrt(252) * 100
        F[b]["vrp"] = q - rv                                          # 隐含 − 实现
    for k in ("qvix", "qvix_pct", "qvix_chg5", "qvix_spike", "vrp"):
        vals = pd.concat([F[b][k] for b in F], axis=1)
        print(f"  {k:<12}非空率 {vals.notna().mean().mean():>6.1%}"
              f"　范围 [{vals.min().min():>7.2f}, {vals.max().max():>7.2f}]")

    # 缓存因子（长表）
    long = []
    for b, df in F.items():
        t = df.reset_index().rename(columns={"index": "date"})
        t["board"] = b
        long.append(t)
    L = pd.concat(long, ignore_index=True)
    p = os.path.join(s8.C.CACHE_DIR, "qvix_factors.parquet")
    L.to_parquet(p, index=False)
    print(f"\n因子已缓存 {p}  rows={len(L)}")

    print("\n" + "=" * 100)
    print("D. 与价量因子的相关性（确认它是新信息）")
    print("=" * 100)
    print(f"  {'板块':<10}{'qvix_pct vs pos':>18}{'qvix_pct vs macd':>19}"
          f"{'vrp vs atr14':>16}{'vrp vs vol20':>15}{'样本':>8}")
    for b in s8.BOARDS:
        if b not in F:
            continue
        df = E.D[b]
        pv = {nm: s8.FACTORS[nm][0](df).reindex(F[b].index)
              for nm in ("pos", "macd", "atr14", "vol20")}
        n = int(F[b]["qvix_pct"].notna().sum())
        print(f"  {s8.NAME[b]:<10}{F[b]['qvix_pct'].corr(pv['pos']):>17.2f}"
              f"{F[b]['qvix_pct'].corr(pv['macd']):>19.2f}"
              f"{F[b]['vrp'].corr(pv['atr14']):>16.2f}"
              f"{F[b]['vrp'].corr(pv['vol20']):>15.2f}{n:>8}")
    print("\n  注：vrp（隐含−实现波动溢价）与 vol20 相关 −0.6~−0.86，说明它主要由")
    print("      「实现波动」驱动；qvix_pct 与 pos/macd 相关 −0.4~0.0，是较独立的信息。")
    print("\n  ⚠️ 因果锚预热问题：50ETF QVIX 从 2015-02 起，若用 rolling(750, min_periods=500)")
    print("     做锚，前 500 个有效值全 NaN → **2015-06 与 2016-01 两次崩盘会看不见**")
    print("     （与 AGENTS.md 铁律四那个「分位的分位」bug 同源）。")
    print("     所以评估阶段以 **QVIX 绝对水平** 为主口径 —— 隐含波动率本身就是")
    print("     年化百分比的绝对刻度（类似 VIX>30 即恐慌），跨年代可比，不需要锚。")


if __name__ == "__main__":
    main()
