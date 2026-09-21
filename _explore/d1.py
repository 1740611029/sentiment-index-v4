"""诊断 1：等权合成 H 的分布 + 校准 scale + 溢出区事件表现。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, model, backtest, factors

big = data.build_stock_indicators()
panels = model.build_all(big)

print("=" * 100)
print("【1】各板块 H 的分布（回测窗口内）")
print("=" * 100)
rows = []
for b in C.BOARD_ORDER:
    p = panels[b]
    s = p["H"].dropna()
    s = s[s.index >= pd.Timestamp(C.BACKTEST_START)]
    rows.append({"板块": C.BOARDS[b]["name"], "n": len(s), "mean": s.mean(), "std": s.std(),
                 "min": s.min(), "p1": s.quantile(.01), "p99": s.quantile(.99), "max": s.max()})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# 校准
allh = pd.concat([panels[b]["H"].dropna().loc[lambda x: x.index >= pd.Timestamp(C.BACKTEST_START)]
                  for b in C.BOARD_ORDER])
print("\n|H| 分位:", {q: round(float(np.percentile(np.abs(allh.values), q)), 3)
                      for q in (90, 95, 98, 99, 99.5, 99.9)})
sc99 = model.calibrate_scale(panels, 0.99)
sc995 = model.calibrate_scale(panels, 0.995)
sc999 = model.calibrate_scale(panels, 0.999)
print(f"scale(q=0.99)={sc99:.2f}  scale(q=0.995)={sc995:.2f}  scale(q=0.999)={sc999:.2f}  当前={C.MODEL['scale']}")

print("\n" + "=" * 100)
print("【2】单因子预测力诊断：各因子 z 极值区（|z|>2）后的 20 日方向")
print("=" * 100)
T = 20
for b in C.BOARD_ORDER:
    p = panels[b]
    p = p[p.index >= pd.Timestamp(C.BACKTEST_START)]
    close = p["close"].values
    pos = {d: i for i, d in enumerate(p.index)}
    line = [C.BOARDS[b]["name"]]
    for k in C.WEIGHTS:
        z = p[f"z_{k}"].dropna()
        lo = z[z <= -2].index
        hi = z[z >= 2].index
        def stat(idx, sign):
            if len(idx) == 0: return "  -   "
            r = []
            for d in idx:
                i = pos[d]
                if i + T < len(close):
                    r.append(close[i + T] / close[i] - 1)
            if not r: return "  -   "
            return f"{np.mean(r)*100:+5.1f}%/{len(r):>3}"
        line.append(f"{k}:{stat(lo,1)}|{stat(hi,-1)}")
    print("  ".join(line))
print("\n(格式 因子:低区均值收益/样本|高区均值收益/样本)")

print("\n" + "=" * 100)
print("【3】不同 scale 下溢出事件数与正确率")
print("=" * 100)
for sc in (14, 16, 18, 20, 22, 25, 28, 32):
    tot = {"bottom": [0, 0], "top": [0, 0]}
    for b in C.BOARD_ORDER:
        p = panels[b].copy()
        p["score"] = (50 + sc * p["H"]).clip(-15, 115)
        r = backtest.summarize(C.BOARDS[b]["name"], p)
        for k in ("bottom", "top"):
            tot[k][0] += r[f"n_{k}"]
            tot[k][1] += r[f"ok_{k}"]
    def f(t): return f"{t[1]}/{t[0]}" + (f"({t[1]/t[0]*100:.0f}%)" if t[0] else "")
    print(f"scale={sc:>3}  底部 {f(tot['bottom']):>12}   顶部 {f(tot['top']):>12}")
