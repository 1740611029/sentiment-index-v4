"""追剩下的 2 次失败：验证「只在恐慌放量时买」能不能把它们滤掉。

上一轮诊断里 5 次失败的 cf_b 全是 0 —— 没有放量的恐慌，一次都没成功过。
cf_b > 0 的语义很干净：成交额分位 > 0.60 且 L < 12，即「放量 + 超跌」= 恐慌投降。
没有成交量的恐慌不算投降。

本脚本：① 用当前模型(K_G=0)重列 19 次信号的特征
        ② 扫 cf_b 门槛
        ③ 拿去长历史（扩展锚 + trailing750 双口径）验证，看是否泛化
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, store, model as M

H, TOL, COOL = store.H, store.TOL, store.COOL_TRADING
THR = store.ENTRY_THR

P = store.load()


def signals(p, cf_lo=None):
    s = p["score"].to_numpy(float); cl = p["close"].to_numpy(float)
    cf = p["cf_b"].to_numpy(float)
    amt = p["amt_pct"].to_numpy(float)
    lo250 = p["close"].rolling(250, min_periods=60).min().to_numpy(float)
    n = len(s); out = []; last = -10 ** 9
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if not (s[i - 1] <= THR and s[i] > s[i - 1]): continue
        if cf_lo is not None and (np.isnan(cf[i]) or cf[i] <= cf_lo): continue
        last = i
        c0 = cl[i]; seg = cl[i + 1:i + H + 1]
        out.append(dict(date=p.index[i], ret=float(seg[-1] / c0 - 1), mdd=float(seg.min() / c0 - 1),
                        score=float(s[i]), prev=float(s[i - 1]), cf_b=float(cf[i]),
                        amt=float(amt[i]), L=float(p["L"].iloc[i]),
                        above_lo=float(cl[i] / lo250[i] - 1) if not np.isnan(lo250[i]) else np.nan))
    return out


def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def line(rows, label):
    if not rows:
        print(f"  {label:<24} 无样本"); return
    o = sum(1 for x in rows if x["ret"] > 0 and x["mdd"] >= -TOL)
    nt = sum(1 for x in rows if x["mdd"] >= -TOL)
    print(f"  {label:<24} n={len(rows):>3}  命中 {o}/{len(rows):<3} {o/len(rows)*100:5.1f}%   "
          f"未被套 {nt/len(rows)*100:5.1f}%   均T+20 {np.mean([x['ret'] for x in rows])*100:+6.2f}%   "
          f"最差 {min(x['mdd'] for x in rows)*100:+7.2f}%   下界 {wilson(o,len(rows))*100:5.1f}%")


print("=" * 128)
print("当前模型（K_G=0）· 19 次信号逐条特征")
print("=" * 128)
print(f"  {'✔/✘':<4}{'板块':<9}{'日期':<12}{'T+20':>8}{'最深':>9}{'分值':>8}{'昨值':>8}"
      f"{'cf_b':>8}{'amt':>7}{'L':>7}{'距250日低':>10}")
allrows = []
for b in C.BOARD_ORDER:
    for r in signals(P[b]):
        ok = "✔" if (r["ret"] > 0 and r["mdd"] >= -TOL) else "✘"
        r["board"] = C.BOARDS[b]["name"]; allrows.append(r)
        print(f"  {ok:<4}{C.BOARDS[b]['name']:<9}{str(r['date'].date()):<12}"
              f"{r['ret']*100:>+7.2f}%{r['mdd']*100:>+8.2f}%{r['score']:>8.1f}{r['prev']:>8.1f}"
              f"{r['cf_b']:>8.3f}{r['amt']:>7.3f}{r['L']:>7.1f}{r['above_lo']*100:>+9.2f}%")

print("\n【按 cf_b 分组】")
for lo, hi, tag in [(0.0, 9, "cf_b > 0（有放量）"), (None, 0.0, "cf_b = 0（无放量）")]:
    sub = [x for x in allrows if (x["cf_b"] > lo if lo is not None else x["cf_b"] <= hi)]
    line(sub, tag)

print("\n" + "=" * 128)
print("cf_b 门槛扫描（生产 6 板块）")
print("=" * 128)
for t in [None, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
    rows = []
    for b in C.BOARD_ORDER:
        rows += signals(P[b], cf_lo=t)
    line(rows, f"cf_b {'> '+str(t) if t is not None else '不限'}")
