"""检查 6 个交付板块的指数：v2 缓存 vs 新浪长历史，能否拼接。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data

IH = os.path.join(C.DATA_DIR, "cache", "index_hist")
MAP = {"SH": "SH", "STAR": "STAR", "CHINEXT": "CHINEXT",
       "CSI1000": "CSI1000", "CSI2000": "CSI2000", "HS300": "HS300"}

for bk, fn in MAP.items():
    base = data.load_index(bk)
    base["date"] = pd.to_datetime(base["date"])
    print(f"\n【{bk}】v2 缓存 {len(base)} 行  {base['date'].min().date()} ~ {base['date'].max().date()}"
          f"  末值 {base['close'].iloc[-1]:.2f}  列={list(base.columns)}")
    p = os.path.join(IH, f"{fn}.parquet")
    if not os.path.exists(p):
        print("   无长历史文件"); continue
    h = pd.read_parquet(p); h["date"] = pd.to_datetime(h["date"])
    print(f"   长历史 {len(h)} 行  {h['date'].min().date()} ~ {h['date'].max().date()}  末值 {h['close'].iloc[-1]:.2f}")
    j = base.set_index("date")["close"].to_frame("v").join(
        h.set_index("date")["close"].to_frame("h"), how="inner").dropna()
    if j.empty:
        print("   无重叠"); continue
    r = j["v"] / j["h"]
    print(f"   重叠 {len(j)} 天  corr={j['v'].corr(j['h']):+.4f}  比率 均值 {r.mean():.4f} "
          f"中位 {r.median():.4f} CV={r.std()/r.mean():.4f}")
    # 日收益一致性（更能说明是不是同一个指数）
    rv = j["v"].pct_change().dropna(); rh = j["h"].pct_change().dropna()
    m = pd.concat([rv, rh], axis=1).dropna()
    print(f"   日收益 corr={m.iloc[:,0].corr(m.iloc[:,1]):+.4f}  平均绝对差={ (m.iloc[:,0]-m.iloc[:,1]).abs().mean()*100:.4f}%")
