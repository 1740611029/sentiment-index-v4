"""诊断 3 个底部失败案例的形态特征，找能区分「下跌中继」与「真底」的信号。

候选特征（全部只用当日及历史数据）：
  dL5    = L − L.shift(5)            广度 5 日变化，>0 表示广度已停止恶化
  dL10   = L − L.shift(10)
  decel  = ret5 − ret20/4            跌速是否放缓（>0 = 最近跌得比过去 20 日平均慢）
  kpos   = (close−low)/(high−low)    收盘位置，1=收在最高，恐慌反包日接近 1
  kpos3  = 近 3 日 kpos 最大值
  dd250                               距 250 日高点回撤
  amt_pct                             成交额分位
  nlow   = 近 10 日 L 是否还在创新低
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors, model

MAIN_H = 20
TOL = 0.03
GAP = 20

print("building ...", flush=True)
stock_ind = data.build_stock_indicators()
panels = model.build_all(stock_ind)


def fwd(close, d, h=MAIN_H):
    i = close.index.get_loc(d)
    if i + h >= len(close):
        return None
    c0 = close.iloc[i]; seg = close.iloc[i + 1:i + h + 1]
    return seg.iloc[-1] / c0 - 1, seg.min() / c0 - 1, seg.max() / c0 - 1


def pick_bottom(score, thr=0.0, gap=GAP):
    s = score.dropna(); m = s < thr
    if not m.any():
        return []
    grp = (m != m.shift()).cumsum()
    picks = sorted({seg.idxmin() for _, seg in s[m].groupby(grp[m])})
    res = []
    for d in picks:
        if res and (d - res[-1]).days <= gap:
            continue
        res.append(d)
    return res


rows = []
for b in C.BOARD_ORDER:
    p = panels[b]
    L = p["L"]
    kpos = (p["close"] - p["low"]) / (p["high"] - p["low"]).replace(0, np.nan)
    ret5 = p["close"] / p["close"].shift(5) - 1
    ret20 = p["close"] / p["close"].shift(20) - 1
    dL5 = L - L.shift(5)
    dL10 = L - L.shift(10)
    decel = ret5 - ret20 / 4.0
    kpos3 = kpos.rolling(3).max()
    lmin10 = L.rolling(10).min()
    for d in pick_bottom(p["score"]):
        f = fwd(p["close"], d)
        if f is None:
            continue
        ok = f[0] > 0 and f[1] >= -TOL
        rows.append(dict(
            b=b, d=d, ok=ok, ret20=f[0] * 100, mdd=f[1] * 100,
            score=p["score"].loc[d], L=L.loc[d], dL5=dL5.loc[d], dL10=dL10.loc[d],
            decel=decel.loc[d] * 100, kpos=kpos.loc[d], kpos3=kpos3.loc[d],
            amt=p["amt_pct"].loc[d], dd=p["dd250"].loc[d] * 100,
            newlow=1.0 if L.loc[d] <= lmin10.loc[d] else 0.0,
        ))

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

print("\n" + "=" * 130)
print("全部底部事件形态（按是否命中排序）")
print("=" * 130)
print(f"{'板块':>8} {'日期':<11} {'':<5} {'20日%':>7} {'最深%':>7} {'分值':>6} "
      f"{'L':>6} {'dL5':>6} {'dL10':>7} {'跌速':>7} {'kpos':>5} {'kpos3':>6} {'amt':>5} {'dd250':>7} {'新低':>4}")
for _, r in df.sort_values(["ok", "d"]).iterrows():
    print(f"{C.BOARDS[r['b']]['name']:>8} {str(r['d'].date()):<11} {'OK  ' if r['ok'] else 'FAIL':<5} "
          f"{r['ret20']:+7.2f} {r['mdd']:+7.2f} {r['score']:+6.1f} {r['L']:+6.1f} "
          f"{r['dL5']:+6.1f} {r['dL10']:+7.1f} {r['decel']:+7.2f} {r['kpos']:5.2f} "
          f"{r['kpos3']:6.2f} {r['amt']:5.2f} {r['dd']:+7.1f} {r['newlow']:4.0f}")

print("\n" + "=" * 130)
print("分组统计（OK 组 vs FAIL 组）")
print("=" * 130)
for col in ["score", "L", "dL5", "dL10", "decel", "kpos", "kpos3", "amt", "dd", "newlow"]:
    a = df.loc[df["ok"], col]; c = df.loc[~df["ok"], col]
    print(f"  {col:>7}   OK  mean {a.mean():+8.2f}  [{a.min():+7.2f},{a.max():+7.2f}]   "
          f"FAIL mean {c.mean():+8.2f}  [{c.min():+7.2f},{c.max():+7.2f}]")
