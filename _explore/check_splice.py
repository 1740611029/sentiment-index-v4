"""验证能否用「重叠期常数比率」把新浪(锚定2023-06)与已有缓存(锚定2026-09)对齐。

qfq 前复权对价格是乘性调整，所以两段只差一个常数因子。
若重叠期内 close_v1v2 / close_sina 基本恒定（变异系数很小），就能安全拼接，
不必重新下载 4225 只。
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data

DEST = os.path.join(C.DATA_DIR, "cache", "stocks_hist")

codes = [f[:-8] for f in os.listdir(DEST) if f.endswith(".parquet")]
print("已下载股票:", len(codes))
random.seed(0)
sample = random.sample(codes, min(40, len(codes)))

rows = []
for c in sample:
    p = os.path.join(DEST, f"{c}.parquet")
    a = pd.read_parquet(p)
    a["date"] = pd.to_datetime(a["date"])
    b = data.load_stock(c)
    if b is None or len(b) == 0:
        continue
    b["date"] = pd.to_datetime(b["date"])
    # 重叠期
    lo = max(a["date"].min(), b["date"].min())
    hi = min(a["date"].max(), b["date"].max())
    if lo >= hi:
        continue
    A = a[(a["date"] >= lo) & (a["date"] <= hi)].set_index("date")["close"].astype(float)
    B = b[(b["date"] >= lo) & (b["date"] <= hi)].set_index("date")["close"].astype(float)
    j = pd.concat([A, B], axis=1, keys=["sina", "v12"]).dropna()
    if len(j) < 30:
        continue
    r = (j["v12"] / j["sina"]).replace([np.inf, -np.inf], np.nan).dropna()
    rows.append(dict(code=c, n=len(j), med=float(r.median()),
                     cv=float(r.std() / r.mean()) if r.mean() else np.nan,
                     p05=float(r.quantile(.05)), p95=float(r.quantile(.95))))

df = pd.DataFrame(rows)
print(f"\n可比对股票 {len(df)} 只，重叠期中位数 {int(df['n'].median())} 天")
pd.set_option("display.width", 160)
print(df.head(15).to_string(index=False))
print(f"\n比率中位数 median={df['med'].median():.4f}  范围 [{df['med'].min():.4f}, {df['med'].max():.4f}]")
print(f"比率变异系数 CV: 中位 {df['cv'].median():.5f}  最大 {df['cv'].max():.5f}")
print(f"\n判定：{'可以安全按常数比率拼接' if df['cv'].max() < 0.05 else '比率不稳定，不能拼接，需重下'}")
