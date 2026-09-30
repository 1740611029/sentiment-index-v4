# -*- coding: utf-8 -*-
"""t2: 两融杠杆(融资余额)出清 对小波段并集的增益测试。

动机：这是 AGENTS §4 里没在「小波段 7 天尺度」测过、且项目已有数据、机制正交的
方向。逻辑：底部往往是「杠杆强平后出清」，融资余额快速下降可能标记真底。
（对照 AGENTS：SENTI-1 顶部曾用融资余额分位，但底部侧只在长尺度做过 bna 破净。）

纪律：全部 shift(1)（两融次日公布，因果）；因果 250 日分位；信号/事件/市场级三层；
分半复核；同环境基线。margin 数据止于 2026-09-17，之后信号无读数会被过滤掉。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"d:\情绪指标4")
import numpy as np
import pandas as pd
from senti import config as C, store, swing, swing2, swing3, data

mg = pd.read_parquet(r"d:\情绪指标4\data\cache\market\margin.parquet")
mg["date"] = pd.to_datetime(mg["date"])
bal = mg.set_index("date").sort_index()["margin"].astype(float)     # 融资余额(元)
buy = pd.read_parquet(r"d:\情绪指标4\data\cache\market\margin_sh.parquet")
buy["date"] = pd.to_datetime(buy["date"])
buys = buy.set_index("date").sort_index()["buy_sh"].astype(float)

# 因果特征（都 shift(1)）
chg5 = (bal / bal.shift(5) - 1.0).shift(1)          # 5日融资余额变化率(负=出清)
chg20 = (bal / bal.shift(20) - 1.0).shift(1)
pct250 = bal.rolling(250, min_periods=120).rank(pct=True).shift(1)

sp, s2p, s3p = swing.load(), swing2.load(), swing3.load()
bp = store.load()
sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
uev = {b: store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r) for b in C.BOARD_ORDER}

def rate(evs):
    d = [e for e in evs if e.get("ok") is not None]
    n = len(d); k = sum(1 for e in d if e["ok"])
    return n, (round(k/n*100, 1) if n else None)

def tiers(by, label):
    ns, rs = rate([e for v in by.values() for e in v])
    ne, re_ = rate(store.cluster_events(by))
    nm, rm = rate(store.market_events(by))
    print(f"{label:<42} 信号{ns:>4}/{rs}%  事件{ne:>4}/{re_}%  市场{nm:>3}/{rm}%")

def val(e, s):
    v = s.get(pd.Timestamp(e["date"]), np.nan)
    return np.nan if v is None else v

def filt(cond):
    out = {}
    for b, evs in uev.items():
        keep = [e for e in evs if cond(e)]
        if keep:
            out[b] = keep
    return out

print("=== 基线 ===")
tiers(uev, "小波段并集(不过滤)")

print("\n=== 融资余额快速下降(出清) 作为并集过滤器 ===")
for th in (-0.02, -0.04, -0.06, -0.08):
    tiers(filt(lambda e, t=th: val(e, chg5) < t), f"并集 & 5日融资变化<{th*100:.0f}%")
for th in (-0.05, -0.08, -0.12):
    tiers(filt(lambda e, t=th: val(e, chg20) < t), f"并集 & 20日融资变化<{th*100:.0f}%")

print("\n=== 融资余额分位低(整体杠杆出清) 作为过滤器 ===")
for pq in (0.10, 0.20, 0.30):
    tiers(filt(lambda e, t=pq: val(e, pct250) < t), f"并集 & 融资分位<{int(pq*100)}%")

print("\n=== 反向对照：融资上升期(杠杆回暖) 是否更好 ===")
for th in (0.0, 0.02, 0.04):
    tiers(filt(lambda e, t=th: val(e, chg5) >= t), f"并集 & 5日融资变化>={th*100:.0f}%")

# 覆盖度提醒
sig = [e for v in uev.values() for e in v]
have = sum(1 for e in sig if not np.isnan(val(e, chg5)))
print(f"\n(并集 {len(sig)} 个信号里，有 chg5 读数的 {have} 个；margin 止于 09-17)")

# ---------- 分半 ----------
mid = pd.Timestamp("2025-06-01")
print("\n=== 分半复核：5日融资变化<-4% ===")
for tag, sel in (("前半", lambda d: d < mid), ("后半", lambda d: d >= mid)):
    base = [e for v in uev.values() for e in v if sel(pd.Timestamp(e["date"]))]
    f = [e for v in uev.values() for e in v
         if sel(pd.Timestamp(e["date"])) and val(e, chg5) < -0.04]
    nb, rb = rate(base); nf, rf = rate(f)
    print(f"  {tag}: 不过滤 {nb}/{rb}%   地杠杆出清 {nf}/{rf}%")

# ---------- 同环境基线：任意日买HS300,按5日融资变化分桶 ----------
c300 = data.load_index("HS300").set_index("date").sort_index()["close"].astype(float)
c = c300.to_numpy(float); idx = c300.index; n = len(c); H, TOL = 7, 0.03
ch = chg5.reindex(idx)
rows = [(ch.iloc[i], c[i+H]/c[i]-1 > 0 and c[i+1:i+H+1].min()/c[i]-1 >= -TOL)
        for i in range(n-H) if not np.isnan(ch.iloc[i])]
df = pd.DataFrame(rows, columns=["chg", "ok"])
df["b"] = pd.cut(df["chg"], [-1, -.06, -.03, 0, .03, .06, 1])
print("\n=== 同环境基线：随机买HS300 按5日融资变化分桶 ===")
print(df.groupby("b", observed=True)["ok"].agg(["count", "mean"]).round(3))
