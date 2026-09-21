"""风险端：加止损能换来什么？

信号层面已推到头（三次过滤尝试全部被长历史否决）。但最差一次回撤 −23.61% 很痛。
这里不改信号，只看**入场之后**的风控：盘中跌破入场价 X% 就离场。

对比：不止损 / 止损 −5% / −8% / −12%，持有 T+20 / T+40 / T+60。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, store

COOL, THR = store.COOL_TRADING, store.ENTRY_THR
HOR = [20, 40, 60]
STOPS = [None, 0.05, 0.08, 0.12]

P = store.load()


def signals(p, hold, stop):
    s = p["score"].to_numpy(float); cl = p["close"].to_numpy(float)
    n = len(s); out = []; last = -10 ** 9
    for i in range(1, n - hold):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if s[i - 1] <= THR and s[i] > s[i - 1]:
            last = i
            c0 = cl[i]
            r = None
            for j in range(i + 1, min(i + hold + 1, n)):
                dd = cl[j] / c0 - 1
                if stop is not None and dd <= -stop:
                    r = -stop; break
            if r is None:
                r = float(cl[min(i + hold, n - 1)] / c0 - 1)
            out.append({"date": p.index[i], "ret": r, "stopped": (stop is not None and r == -stop)})
    return out


rows = []
for b in C.BOARD_ORDER:
    for r in signals(P[b], max(HOR), None):
        r["board"] = C.BOARDS[b]["name"]; rows.append(r)
df = pd.DataFrame(rows)
print(f"共 {len(df)} 次信号\n")

print("=" * 112)
print("止损 × 持有期：收益分布")
print("=" * 112)
print(f"  {'持有期':<8}{'止损':<10}{'样本':>6}{'盈利占比':>10}{'均收益':>10}{'中位收益':>10}{'最差':>10}{'止损出场':>10}")
for hold in HOR:
    for stop in STOPS:
        rs = []
        for b in C.BOARD_ORDER:
            rs += signals(P[b], hold, stop)
        if not rs:
            continue
        ret = np.array([x["ret"] for x in rs])
        st = sum(1 for x in rs if x["stopped"])
        print(f"  T+{hold:<6}{('无' if stop is None else f'−{stop*100:.0f}%'):<10}{len(rs):>6}"
              f"{(ret > 0).mean()*100:>9.1f}%{ret.mean()*100:>+9.2f}%{np.median(ret)*100:>+9.2f}%"
              f"{ret.min()*100:>+9.2f}%{st:>10}")

print("\n" + "=" * 112)
print("逐信号：T+60 下 无止损 vs −8% 止损")
print("=" * 112)
print(f"  {'板块':<9}{'日期':<12}{'无止损':>12}{'−8%止损':>12}{'是否止损':>10}")
a = {b: signals(P[b], 60, None) for b in C.BOARD_ORDER}
c = {b: signals(P[b], 60, 0.08) for b in C.BOARD_ORDER}
for b in C.BOARD_ORDER:
    for x, y in zip(a[b], c[b]):
        print(f"  {C.BOARDS[b]['name']:<9}{str(x['date'].date()):<12}{x['ret']*100:>+11.2f}%"
              f"{y['ret']*100:>+11.2f}%{'是' if y['stopped'] else '':>10}")

print("\n" + "=" * 112)
print("小结")
print("=" * 112)
for stop in STOPS:
    rs = []
    for b in C.BOARD_ORDER:
        rs += signals(P[b], 60, stop)
    ret = np.array([x["ret"] for x in rs])
    print(f"  T+60  止损 {('无' if stop is None else f'−{stop*100:.0f}%'):<6}  "
          f"盈利占比 {(ret > 0).mean()*100:5.1f}%   均 {ret.mean()*100:+6.2f}%   "
          f"最差 {ret.min()*100:+6.2f}%   标准差 {ret.std()*100:5.2f}%")
