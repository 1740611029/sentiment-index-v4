"""抓取「新信息源」：市场级杠杆与投降指标。

选择理由：现有模型全是价量数据，而价量在「流动性危机」和「真恐慌底」上长得一模一样
（2024-01-23 中证2000 浮亏 −23.61% 就是栽在这里）。
能区分两者的东西必须来自价量之外：

  ① 融资余额（沪 + 深合计）   —— 杠杆水位与去杠杆速度
     真投降：杠杆先慢慢退，情绪后崩 → 融资余额下降平缓
     强平  ：杠杆先崩，被动平仓     → 融资余额断崖
  ② 破净股占比                 —— 全市场估值侧的投降程度
     破净率飙升 = 市场愿意给的价格低于净资产 = 深度投降

关键的时序纪律：
  两融数据次日早上才公布 → 信号日 t 只能用 t-1 的值（shift(1)）。
  破净率同理（依赖季报净资产 + 当日收盘价）→ 也 shift(1)。
  不 shift 就是前视，这个坑前面已经踩过一次了。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import akshare as ak

OUT = r"D:\情绪指标4\data\cache\market"
os.makedirs(OUT, exist_ok=True)

got = {}

# ---------- ① 融资余额 ----------
try:
    sh = ak.macro_china_market_margin_sh()
    sh = sh.rename(columns={"日期": "date", "融资余额": "margin_sh",
                            "融资买入额": "buy_sh", "融券余额": "short_sh"})
    sh["date"] = pd.to_datetime(sh["date"])
    sh = sh[["date", "margin_sh", "buy_sh", "short_sh"]].dropna(subset=["date"])
    sh.to_parquet(os.path.join(OUT, "margin_sh.parquet"), index=False)
    got["margin_sh"] = (len(sh), str(sh["date"].min().date()), str(sh["date"].max().date()))
except Exception as e:
    got["margin_sh"] = f"FAIL {e}"

try:
    sz = ak.macro_china_market_margin_sz()
    sz = sz.rename(columns={"日期": "date", "融资余额": "margin_sz",
                            "融资买入额": "buy_sz", "融券余额": "short_sz"})
    sz["date"] = pd.to_datetime(sz["date"])
    sz = sz[["date", "margin_sz", "buy_sz", "short_sz"]].dropna(subset=["date"])
    sz.to_parquet(os.path.join(OUT, "margin_sz.parquet"), index=False)
    got["margin_sz"] = (len(sz), str(sz["date"].min().date()), str(sz["date"].max().date()))
except Exception as e:
    got["margin_sz"] = f"FAIL {e}"

# 合计
try:
    m = pd.merge(sh, sz, on="date", how="outer").sort_values("date")
    m["margin"] = m["margin_sh"].fillna(0) + m["margin_sz"].fillna(0)
    m.loc[m["margin"] <= 0, "margin"] = float("nan")
    m = m[["date", "margin", "margin_sh", "margin_sz"]].dropna(subset=["margin"])
    m.to_parquet(os.path.join(OUT, "margin.parquet"), index=False)
    got["margin(沪深合计)"] = (len(m), str(m["date"].min().date()), str(m["date"].max().date()))
except Exception as e:
    got["margin(沪深合计)"] = f"FAIL {e}"

# ---------- ② 破净股占比 ----------
try:
    b = ak.stock_a_below_net_asset_statistics()
    b = b.rename(columns={"below_net_asset_ratio": "bna",
                          "below_net_asset": "bna_n", "total_company": "n_co"})
    b["date"] = pd.to_datetime(b["date"])
    b = b[["date", "bna", "bna_n", "n_co"]]
    b.to_parquet(os.path.join(OUT, "bna.parquet"), index=False)
    got["bna(破净占比)"] = (len(b), str(b["date"].min().date()), str(b["date"].max().date()))
except Exception as e:
    got["bna(破净占比)"] = f"FAIL {e}"

print("=" * 96)
for k, v in got.items():
    print(f"  {k:<20} {v}")
print("\n关键区间抽查（沪深合计融资余额，单位：亿元）：")
m = pd.read_parquet(os.path.join(OUT, "margin.parquet"))
m["亿"] = m["margin"] / 1e8
for a, b2, lab in (("2015-06-01", "2015-09-30", "2015 杠杆牛破裂"),
                   ("2016-01-01", "2016-02-29", "2016 熔断"),
                   ("2024-01-15", "2024-02-08", "2024 中小盘流动性危机"),
                   ("2026-08-01", "2026-09-18", "最近")):
    s = m[(m["date"] >= a) & (m["date"] <= b2)]
    if len(s):
        print(f"  {lab:<22} {s['亿'].iloc[0]:>9.0f} → {s['亿'].iloc[-1]:>9.0f}   "
              f"区间变化 {(s['亿'].iloc[-1]/s['亿'].iloc[0]-1)*100:+.1f}%  最低 {s['亿'].min():.0f}")
print("\n破净占比抽查：")
bb = pd.read_parquet(os.path.join(OUT, "bna.parquet"))
bb["pct"] = bb["bna"] * 100
for a, b2, lab in (("2015-06-01", "2015-09-30", "2015"), ("2016-01-01", "2016-02-29", "2016"),
                   ("2024-01-15", "2024-02-08", "2024-01"), ("2025-04-01", "2025-04-15", "2025-04 关税"),
                   ("2026-08-01", "2026-09-18", "最近")):
    s = bb[(bb["date"] >= a) & (bb["date"] <= b2)]
    if len(s):
        print(f"  {lab:<22} {s['pct'].iloc[0]:>6.2f}% → {s['pct'].iloc[-1]:>6.2f}%   最高 {s['pct'].max():.2f}%")
