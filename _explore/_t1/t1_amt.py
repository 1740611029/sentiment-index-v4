# -*- coding: utf-8 -*-
"""t1: 「成交额地量」对小波段并集 / SENTI-1 的增益测试（稳健口径）。

用户猜想：沪市成交额 < 6000 亿可能是底部。

两个关键前置发现（先打印，供人判断）：
  A) 本项目缓存 data/cache/index_long/SH.parquet 的 amount 列单位不统一
     （09-18→09-21 有约 100 倍量级断裂，volume 却连续），所以**指数 amount 不能
     直接拿来卡绝对阈值**。这也是 factors.py 只用 amt 的分位、不用绝对值的原因。
  B) 绝对阈值本身是「制度代理」（AGENTS §24）：2015 年 vs 2024 年地量水平差数倍，
     固定 6000 亿会随时代漂移。所以主推**因果分位**口径。

改用「全市场成交额」= 逐股成交额(元, sina 单股可靠)汇总，稳定。
纪律：信号/事件/市场级三层；过滤结论做过半样本(前/后)与分半复核。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"d:\情绪指标4")
import numpy as np
import pandas as pd
from senti import config as C, store, swing, swing2, swing3, data

pd.set_option("display.width", 160)

# ---------- 发现 A：指数 amount 量级断裂 ----------
sh = data.load_index("SH").set_index("date").sort_index()
raw_amt = sh["amount"].astype(float)
r = (raw_amt / sh["volume"].astype(float)).dropna()
print("=== 指数 amount/volume 隐含均价：近 12 日 ===")
print(pd.DataFrame({"amt": raw_amt, "vol": sh["volume"],
                    "amt/vol": r}).tail(12).round(1))
print(f"隐含均价全历史 P50={r.median():.0f}  近端突变说明 amount 单位不可信\n")

# ---------- 稳健全市场成交额（元 -> 亿元） ----------
si = pd.read_parquet(r"d:\情绪指标4\data\cache\stock_ind.parquet")
mkt_amt = si.groupby("date")["amount"].sum() / 1e8      # 亿元
mkt_amt = mkt_amt.sort_index()
print("=== 全市场成交额(亿元, 逐股汇总) 分布 ===")
print("全历史 P5/P25/P50/P75/P95:",
      [round(mkt_amt.quantile(p), 0) for p in (.05, .25, .5, .75, .95)])
win = mkt_amt.loc[mkt_amt.index >= pd.Timestamp(C.BACKTEST_START)]
print("展示窗口 P5/P25/P50/P75/P95:",
      [round(win.quantile(p), 0) for p in (.05, .25, .5, .75, .95)])
print("展示窗口 <6000亿 天数占比:", f"{(win<6000).mean()*100:.1f}%",
      f"(<8000亿: {(win<8000).mean()*100:.1f}%)")

# 因果分位：只用过去 250 日，shift(1)
amt5 = mkt_amt.rolling(5).mean()
pq1 = mkt_amt.rolling(250, min_periods=120).rank(pct=True).shift(1)
pq5 = amt5.rolling(250, min_periods=120).rank(pct=True).shift(1)

# ---------- 现役信号 ----------
sp, s2p, s3p = swing.load(), swing2.load(), swing3.load()
bp = store.load()
sreso, s2reso, s3reso = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
uev = {b: store.union_events(b, sp[b], s2p[b], s3p[b], sreso, s2reso, s3reso)
       for b in C.BOARD_ORDER}
bev = {b: store.events(bp[b], store.resonance(bp), store.bna_series())
       for b in C.BOARD_ORDER}

def rate(evs):
    d = [e for e in evs if e.get("ok") is not None]
    n = len(d); k = sum(1 for e in d if e["ok"])
    return n, (round(k / n * 100, 1) if n else None)

def tiers(by_board, label):
    sig = [e for evs in by_board.values() for e in evs]
    ns, rs = rate(sig)
    ne, re_ = rate(store.cluster_events(by_board))
    nm, rm = rate(store.market_events(by_board))
    print(f"{label:<40} 信号{ns:>4}/{rs}%  事件{ne:>4}/{re_}%  市场{nm:>3}/{rm}%")

print("\n=== 基线 ===")
tiers(uev, "小波段并集(现役3模型)")
tiers(bev, "SENTI-1 底部")
print("随机基线:", round(float(np.mean([swing2.baseline(s2p[b]) for b in C.BOARD_ORDER])), 1))

def val_at(e, s):
    return s.get(pd.Timestamp(e["date"]), np.nan)

def filtered(by_board, s, thr, asc=True):
    out = {}
    for b, evs in by_board.items():
        keep = [e for e in evs if (val_at(e, s) < thr if asc else val_at(e, s) >= thr)]
        if keep:
            out[b] = keep
    return out

print("\n=== 地量(因果250日分位) 作为并集过滤器 ===")
tiers(uev, "并集(不过滤)")
for pq in (0.05, 0.10, 0.20, 0.30, 0.50):
    tiers(filtered(uev, pq1, pq), f"并集 & 日额分位<{int(pq*100)}%")
print("--- 5日均额分位 ---")
for pq in (0.10, 0.20, 0.30):
    tiers(filtered(uev, pq5, pq), f"并集 & 5日均额分位<{int(pq*100)}%")

print("\n=== 分半复核(前/后半 展示窗口)：地量过滤是否稳 ===")
mid = win.index[len(win)//2]
def half_tiers(s, thr, lab):
    for tag, sel in (("前半", lambda d: d < mid), ("后半", lambda d: d >= mid)):
        out = {}
        for b, evs in uev.items():
            out[b] = [e for e in evs if sel(pd.Timestamp(e["date"])) and val_at(e, s) < thr]
            out[b] = [e for e in out[b] if e]
        out = {k: v for k, v in out.items() if v}
        base = {}
        for b, evs in uev.items():
            base[b] = [e for e in evs if sel(pd.Timestamp(e["date"]))]
        nb, rb = rate([e for v in base.values() for e in v])
        nf, rf = rate([e for v in out.values() for e in v])
        print(f"  {lab} {tag}: 过滤前 {nb}/{rb}%  过滤后 {nf}/{rf}%")
half_tiers(pq1, 0.20, "日额分位<20%")

print("\n=== SENTI-1 同样测 ===")
tiers(bev, "SENTI-1(不过滤)")
for pq in (0.10, 0.20, 0.30):
    tiers(filtered(bev, pq1, pq), f"SENTI-1 & 日额分位<{int(pq*100)}%")

# ---------- 地量作为独立触发器（潜在新因子） ----------
def sim(close, trig, cool=4, H=7, TOL=0.03):
    c = close.to_numpy(float); idx = close.index; n = len(c)
    t = trig.reindex(idx).fillna(False).to_numpy(bool)
    out, last = [], -10**9
    start = pd.Timestamp(C.BACKTEST_START)
    for i in range(n):
        if idx[i] < start or not t[i] or i - last < cool:
            continue
        last = i
        if i + H <= n-1:
            seg = c[i+1:i+H+1]
            out.append({"date": str(idx[i].date()),
                        "ok": bool(seg[-1]/c[i]-1 > 0 and seg.min()/c[i]-1 >= -TOL)})
        else:
            out.append({"date": str(idx[i].date()), "ok": None})
    return out

print("\n=== 地量作为独立触发器(买沪深300 HS300) ===")
c300 = data.load_index("HS300").set_index("date").sort_index()["close"].astype(float)
for pq in (0.05, 0.10, 0.20, 0.30):
    n, r = rate(sim(c300, pq1 < pq)); print(f"  日额分位<{int(pq*100)}%: 信号{n} 命中{r}%")
print("--- 加『地量后分位首日抬升』确认(SWING-3 式拐点语义) ---")
for pq in (0.10, 0.20, 0.30):
    n, r = rate(sim(c300, (pq1 < pq) & (pq1 > pq1.shift(1))))
    print(f"  分位<{int(pq*100)}%且回升: 信号{n} 命中{r}%")

# ---------- 同环境基线 ----------
print("\n=== 同环境基线：任意一天买HS300，按当日额分位分桶 ===")
c = c300.to_numpy(float); idx = c300.index; n = len(c); H, TOL = 7, 0.03
qq = pq1.reindex(idx)
rows = [(qq.iloc[i], c[i+H]/c[i]-1 > 0 and c[i+1:i+H+1].min()/c[i]-1 >= -TOL)
        for i in range(n-H) if not np.isnan(qq.iloc[i])]
df = pd.DataFrame(rows, columns=["q", "ok"])
df["b"] = pd.cut(df["q"], [0, .1, .2, .3, .5, 1.0])
print(df.groupby("b", observed=True)["ok"].agg(["count", "mean"]).round(3))
