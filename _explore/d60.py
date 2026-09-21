"""换尺度看：信号到底在多长的窗口上成立？

现在的判定是固定 T+20。但「底部确认」本身是个更长尺度的事 ——
2024-01-23 中证2000 在 T+20 是 −6.79%，可如果放到 T+60 就涨回来了，
那这次「底部」判断其实是对的，只是进场早了。

本脚本对同一批信号算 T+10/20/40/60/90 的收益与最深回撤，看结论随尺度怎么变。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, store

COOL, THR, TOL = store.COOL_TRADING, store.ENTRY_THR, store.TOL
HOR = [10, 20, 40, 60, 90]

P = store.load()


def signals(p):
    s = p["score"].to_numpy(float); cl = p["close"].to_numpy(float)
    n = len(s); out = []; last = -10 ** 9
    for i in range(1, n - max(HOR)):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if s[i - 1] <= THR and s[i] > s[i - 1]:
            last = i
            c0 = cl[i]
            r = {"date": p.index[i], "score": float(s[i])}
            for h in HOR:
                seg = cl[i + 1:i + h + 1]
                r[f"ret{h}"] = float(seg[-1] / c0 - 1)
                r[f"mdd{h}"] = float(seg.min() / c0 - 1)
            out.append(r)
    return out


rows = []
for b in C.BOARD_ORDER:
    for r in signals(P[b]):
        r["board"] = C.BOARDS[b]["name"]; rows.append(r)
df = pd.DataFrame(rows)
print(f"共 {len(df)} 次信号（判定窗口最长 T+90，样本略少于 19）\n")

print("=" * 118)
print("命中率随持有期的变化（命中 = 期末收益 > 0 且 期间最深回撤 ≥ −3%）")
print("=" * 118)
print(f"  {'持有期':<10}{'命中':>10}{'命中率':>9}{'仅看收益>0':>12}{'未被套≥−3%':>12}{'均收益':>10}{'最差回撤':>10}")
for h in HOR:
    ok = ((df[f"ret{h}"] > 0) & (df[f"mdd{h}"] >= -TOL)).sum()
    pos = (df[f"ret{h}"] > 0).sum()
    nt = (df[f"mdd{h}"] >= -TOL).sum()
    print(f"  T+{h:<8}{ok:>4}/{len(df):<5}{ok/len(df)*100:>8.1f}%{pos/len(df)*100:>11.1f}%"
          f"{nt/len(df)*100:>11.1f}%{df[f'ret{h}'].mean()*100:>+9.2f}%{df[f'mdd{h}'].min()*100:>+9.2f}%")

print("\n" + "=" * 118)
print("逐信号：各持有期的收益 / 最深回撤")
print("=" * 118)
print(f"  {'板块':<9}{'日期':<12}" + "".join(f"{'T+'+str(h):>16}" for h in HOR))
for _, r in df.sort_values("date").iterrows():
    cells = ""
    for h in HOR:
        cells += f"{r[f'ret{h}']*100:>+7.1f}/{r[f'mdd{h}']*100:>6.1f}".rjust(16)
    print(f"  {r['board']:<9}{str(r['date'].date()):<12}" + cells)

print("\n" + "=" * 118)
print("两次「失败」放到更长尺度看")
print("=" * 118)
for _, r in df.iterrows():
    if r["ret20"] <= 0 or r["mdd20"] < -TOL:
        print(f"  {r['board']:<9}{str(r['date'].date())}  "
              + "  ".join(f"T+{h}: {r[f'ret{h}']*100:+6.2f}%（最深 {r[f'mdd{h}']*100:+6.2f}%）"
                          for h in HOR))
