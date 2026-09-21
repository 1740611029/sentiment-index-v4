"""抓取更长历史指数日线（仅用于验证顶部/底部逻辑的样本量，不进入展示回测）。"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, requests
import akshare as ak

OUT = r"D:\情绪指标4\data\cache\long_index"
os.makedirs(OUT, exist_ok=True)

SINA = {"SH": "sh000001", "HS300": "sh000300", "CSI1000": "sh000852",
        "STAR": "sh000688", "CHINEXT": "sz399006"}


def em(code_secid: str) -> pd.DataFrame:
    """东方财富历史 K 线：secid 形如 2.932000"""
    s = requests.Session(); s.trust_env = False
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid={code_secid}&fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56,f57"
           "&klt=101&fqt=0&beg=20100101&end=20500101")
    r = s.get(url, timeout=30)
    js = r.json()["data"]
    rows = [x.split(",") for x in js["klines"]]
    d = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume", "amount"])
    for c in d.columns[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = pd.to_datetime(d["date"])
    return d


got = {}
for k, sym in SINA.items():
    try:
        d = ak.stock_zh_index_daily(symbol=sym)
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"] >= pd.Timestamp("2015-01-01")]
        if "amount" not in d.columns:
            d["amount"] = d["volume"] * d["close"]
        d.to_parquet(os.path.join(OUT, f"{k}.parquet"), index=False)
        got[k] = (len(d), str(d["date"].min().date()), str(d["date"].max().date()))
        print(f"  {k:<9} sina  {got[k]}")
    except Exception as e:
        print(f"  {k:<9} FAIL {type(e).__name__}: {e}")

for k, sec in (("CSI2000", "2.932000"),):
    try:
        d = em(sec)
        d = d[d["date"] >= pd.Timestamp("2015-01-01")]
        d.to_parquet(os.path.join(OUT, f"{k}.parquet"), index=False)
        got[k] = (len(d), str(d["date"].min().date()), str(d["date"].max().date()))
        print(f"  {k:<9} eastmoney {got[k]}")
    except Exception as e:
        print(f"  {k:<9} FAIL {type(e).__name__}: {e}")

json.dump(got, open(os.path.join(OUT, "_meta.json"), "w"), indent=2)
print("\nsaved ->", OUT)
