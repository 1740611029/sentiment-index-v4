"""排查两个数据疑点：1) CSI2000 指数口径 2) 创业板 lim 因子在 2015 年恒为 0"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors
import longhist

PAN = longhist.build()
print("重叠区间 close 一致性：")
for k in ("HS300", "CSI1000", "CSI2000", "CHINEXT"):
    prod = factors.build_board_raw(k)
    lp = PAN[k]
    j = prod[["close"]].join(lp[["close"]], how="inner", lsuffix="_p", rsuffix="_l")
    a = j["close_p"].astype(float); b = j["close_l"].astype(float)
    print(f"  {k:<9} 重叠 {len(j):>4}  corr {a.corr(b):+.4f}  "
          f"中位相对差 {((a-b).abs()/a.abs()).median()*100:.4f}%  "
          f"末值 生产 {a.iloc[-1]:.2f} / 长历史 {b.iloc[-1]:.2f}")

print("\nCSI2000 指数来源对比：")
p2 = data.load_index("CSI2000"); p2["date"] = pd.to_datetime(p2["date"])
ih = pd.read_parquet(os.path.join(C.DATA_DIR, "cache", "index_hist", "CSI2000.parquet"))
ih["date"] = pd.to_datetime(ih["date"])
print(f"  生产   {len(p2)} 行  {p2['date'].min().date()} ~ {p2['date'].max().date()}  末值 {p2['close'].iloc[-1]:.2f}")
print(f"  长历史 {len(ih)} 行  {ih['date'].min().date()} ~ {ih['date'].max().date()}  末值 {ih['close'].iloc[-1]:.2f}")
m = p2.set_index("date")["close"].to_frame("p").join(ih.set_index("date")["close"].to_frame("l"), how="inner").dropna()
print(f"  重叠 {len(m)}  corr {m['p'].corr(m['l']):+.4f}")
print("  头："); print(m.head(3).to_string())
print("  尾："); print(m.tail(3).to_string())

print("\n创业板 lim 因子体检：")
raw = PAN["CHINEXT"]
for tag, sl in (("2015-07", ("2015-07-01", "2015-07-31")), ("2024-02", ("2024-01-15", "2024-02-29"))):
    sub = raw.loc[sl[0]:sl[1], ["lim", "b20", "disp"]]
    if not sub.empty:
        print(f"  {tag}  lim {sub['lim'].min():+.4f}~{sub['lim'].max():+.4f}")

# 直接看个股涨跌停判定阈值
codes = sorted(set(data.load_universe("CHINEXT")))[:3]
print(f"\n  创业板样本股 {codes}")
for c in codes:
    df = longhist.load_merged(c)
    if df is None or len(df) == 0:
        print(f"    {c} 无数据"); continue
    ind = data.compute_stock_indicators(df, c)
    s = ind.loc[(ind["date"] >= "2015-07-01") & (ind["date"] <= "2015-07-31")]
    if len(s):
        print(f"    {c} 2015-07  lu5={s['any_lu5'].sum():.0f} ld5={s['any_ld5'].sum():.0f}  "
              f"最小 ret1={s['ret1'].min():+.3f}")
    s2 = ind.loc[(ind["date"] >= "2024-01-15") & (ind["date"] <= "2024-02-29")]
    if len(s2):
        print(f"    {c} 2024-02  lu5={s2['any_lu5'].sum():.0f} ld5={s2['any_ld5'].sum():.0f}  "
              f"最小 ret1={s2['ret1'].min():+.3f}")

print("\n涨跌停判定阈值代码：")
import inspect
src = inspect.getsource(data.compute_stock_indicators)
for ln in src.splitlines():
    if "lu" in ln or "ld" in ln or "limit" in ln.lower() or "0.0" in ln and "%" not in ln:
        print("   ", ln.strip())
