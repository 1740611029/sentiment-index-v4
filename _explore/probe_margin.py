"""探测「新信息源」的可得性。

目标：找一个能区分「投降」和「强平」的数据。
2024-01-23 那次失败的定性是杠杆强平（雪球敲入 / DMA 爆仓 / 两融平仓），
这类事件在价量上和真正的恐慌底几乎一样 —— 但**融资余额**会给出完全不同的读数：
  真投降   = 情绪先崩，杠杆随后慢慢退，融资余额下降但没断崖
  强平     = 杠杆先崩，融资余额断崖式下降（被动平仓，不是主动投降）

候选：
  ① 融资余额（沪深两市合计/分市场）日频 —— 最直接的杠杆指标
  ② 融券余额 —— 做空压力
  ③ 破净率（市净率<1 的个股占比）—— 需要个股 PB，看有没有
  ④ 新增投资者数量（月频，太粗，仅备案）
"""
import sys, traceback
import akshare as ak
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

CAND = [
    ("stock_margin_sse", lambda: ak.stock_margin_sse(start_date="20130101", end_date="20260920")),
    ("stock_margin_szse", lambda: ak.stock_margin_szse(date="20240123")),
    ("stock_margin_balance_sse", lambda: ak.stock_margin_balance_sse(start_date="20130101", end_date="20260920")),
]

for name, fn in CAND:
    try:
        d = fn()
        print("=" * 110)
        print(f"✓ {name}   shape={d.shape}")
        print("  列：", list(d.columns))
        print(d.head(3).to_string())
        print("  尾部：")
        print(d.tail(2).to_string())
    except Exception as e:
        print("=" * 110)
        print(f"✗ {name}   {type(e).__name__}: {e}")

print("\n" + "=" * 110)
print("akshare 里含 margin / 融资 的接口：")
print([x for x in dir(ak) if "margin" in x.lower() or "rzrq" in x.lower()])
print("\n含 net_asset / pb / 破净 的接口：")
print([x for x in dir(ak) if "net_asset" in x.lower() or x.lower().endswith("_pb")])
print("\n含 account / 开户 的接口：")
print([x for x in dir(ak) if "account" in x.lower() or "investor" in x.lower()])
