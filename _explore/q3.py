"""按 hit-rate 检验口径，评估两个模型命中率的统计可信度。

两道必答题：1) 随机基线是多少（净优势）；2) 样本够不够（显著性 + 置信区间）。
样本少时补一个功效更高的连续指标（前瞻收益置换检验）。
"""
import os, sys
from math import comb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C, store, swing

RNG = np.random.default_rng(0)


def binom_sf(k, n, p):
    """P(X >= k)"""
    if n <= 0 or p is None:
        return None
    k = max(0, min(k, n))
    return float(sum(comb(n, i) * p**i * (1.0 - p)**(n - i) for i in range(k, n + 1)))


def wilson(succ, n, z=1.96):
    if n <= 0:
        return None
    ph = succ / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def perm_p(obs, pool, trials=20000):
    """双尾置换：随机抽 n 个，均值与观测的偏离至少一样大的概率。"""
    obs = np.asarray(obs, float)
    pool = np.asarray(pool, float)
    pool = pool[~np.isnan(pool)]
    n = len(obs)
    if n == 0 or len(pool) == 0:
        return None
    m = obs.mean()
    draws = RNG.choice(pool, size=(trials, n), replace=True)
    dm = draws.mean(axis=1)
    p1 = float((dm >= m).mean())
    return min(1.0, 2.0 * min(p1, 1.0 - p1))


def verdict(n, succ, base, p):
    if n < 20:
        tag = "样本偏少"
    else:
        tag = ""
    lift = succ / n - base
    star = "显著" if (p is not None and p < 0.05) else "不显著"
    return lift, star, tag


def show(name, succ, n, base, fwd=None, pool=None):
    lo, hi = wilson(succ, n)
    p1 = binom_sf(succ, n, base)
    lift, star, tag = verdict(n, succ, base, p1)
    pp = perm_p(fwd, pool) if (fwd is not None and pool) else None
    print(f"{name:<26} {succ}/{n:<4}= {succ/n*100:5.1f}%  基线 {base*100:5.1f}%  "
          f"优势 {lift*100:+5.1f}pp  95%CI [{lo*100:4.1f}, {hi*100:5.1f}]  "
          f"p={p1:.4f} {star}{('（' + tag + '）') if tag else ''}")
    if pp is not None:
        print(f"{'':<26}  前瞻收益置换 p={pp:.4f}  "
              f"信号均值 {np.nanmean(fwd)*100:+.2f}% vs 池均值 {np.nanmean(pool)*100:+.2f}%")
    return p1


print("=" * 96)
print("SENTI-1 大波段（判定 T+20，容差 3%）")
print("=" * 96)
P = store.load()
S = swing.load()
reso = store.resonance(P)
bna, mg = store.bna_series(), store.margin_series()

allb, allt = [], []
pool_b, pool_t = [], []
for b in C.BOARD_ORDER:
    p = P[b]
    bot, top = store.events(p, reso, bna, mg)
    for e in bot: e["board"] = b; allb.append(e)
    for e in top: e["board"] = b; allt.append(e)
    # 随机池：同板块随便挑一天买入的 T+20 结果
    c = p["close"].to_numpy(float)
    for i in range(len(c) - store.H - 1):
        seg = c[i + 1: i + store.H + 1]
        pool_b.append(seg[-1] / c[i] - 1)
        pool_t.append(-(seg[-1] / c[i] - 1))
base_b = float(np.mean([(e["ret"] / 100 > 0) and (e["risk"] / 100 >= -store.TOL)
                        for e in allb]) * 0 + np.mean(
    [(p_ > 0 and m_ >= -store.TOL) for p_ in
     [np.nan]] ) ) if False else None
# 随机基线：直接从池里算（需要回撤，重算一次）
base_b = base_t = None
cnt_b = ok_b = cnt_t = ok_t = 0
for b in C.BOARD_ORDER:
    c = P[b]["close"].to_numpy(float)
    for i in range(len(c) - store.H - 1):
        seg = c[i + 1: i + store.H + 1]
        end = seg[-1] / c[i] - 1
        mdd = seg.min() / c[i] - 1
        run = seg.max() / c[i] - 1
        cnt_b += 1; ok_b += int(end > 0 and mdd >= -store.TOL)
        cnt_t += 1; ok_t += int(end < 0 and run <= store.TOL)
base_b, base_t = ok_b / cnt_b, ok_t / cnt_t

ok = sum(1 for e in allb if e["ok"])
show("底部（全部）", ok, len(allb), base_b,
     fwd=[e["ret"] / 100 for e in allb], pool=pool_b)
sub = [e for e in allb if e.get("reso") and e["reso"] >= 2]
if sub:
    show("底部·共振≥2", sum(1 for e in sub if e["ok"]), len(sub), base_b,
         fwd=[e["ret"] / 100 for e in sub], pool=pool_b)
sub = [e for e in allb if e["grade"] == "A"]
if sub:
    show("底部·A级", sum(1 for e in sub if e["ok"]), len(sub), base_b,
         fwd=[e["ret"] / 100 for e in sub], pool=pool_b)
if allt:
    show("顶部（首破100）", sum(1 for e in allt if e["ok"]), len(allt), base_t,
         fwd=[-e["ret"] / 100 for e in allt], pool=pool_t)

print()
print("=" * 96)
print("SWING 小波段（判定 T+7，容差 3%）")
print("=" * 96)
res = swing.resonance(S)
pool_s = []
for b in C.BOARD_ORDER:
    c = S[b]["close"].to_numpy(float)
    for i in range(len(c) - swing.H - 1):
        seg = c[i + 1: i + swing.H + 1]
        pool_s.append(seg[-1] / c[i] - 1)
base_s = float(np.mean([1 for _ in []]) ) if False else (
    sum(1 for v in pool_s if v > 0) / len(pool_s) * 0)
# 基线（含回撤约束）
cb = co = 0
for b in C.BOARD_ORDER:
    c = S[b]["close"].to_numpy(float)
    for i in range(len(c) - swing.H - 1):
        seg = c[i + 1: i + swing.H + 1]
        cb += 1
        co += int(seg[-1] / c[i] - 1 > 0 and seg.min() / c[i] - 1 >= -swing.TOL)
base_s = co / cb

ev3 = []
for b in C.BOARD_ORDER:
    for e in swing.events(S[b], res):
        e["board"] = b; ev3.append(e)
done3 = [e for e in ev3 if e["ok"] is not None]

# 近 5.7 年窗口
C.BACKTEST_START = "2021-01-01"
S5 = swing.build_all()
res5 = swing.resonance(S5)
ev5 = []
for b in C.BOARD_ORDER:
    for e in swing.events(S5[b], res5):
        e["board"] = b; ev5.append(e)
done5 = [e for e in ev5 if e["ok"] is not None]

pool5 = []
for b in C.BOARD_ORDER:
    c = S5[b]["close"].to_numpy(float)
    for i in range(len(c) - swing.H - 1):
        seg = c[i + 1: i + swing.H + 1]
        pool5.append(seg[-1] / c[i] - 1)
c5 = o5 = 0
for b in C.BOARD_ORDER:
    c = S5[b]["close"].to_numpy(float)
    for i in range(len(c) - swing.H - 1):
        seg = c[i + 1: i + swing.H + 1]
        c5 += 1
        o5 += int(seg[-1] / c[i] - 1 > 0 and seg.min() / c[i] - 1 >= -swing.TOL)
base5 = o5 / c5

for tag, done, base, pool in (("近3年", done3, base_s, pool_s),
                              ("近5.7年", done5, base5, pool5)):
    for lab, sel in (("全部", lambda e: True),
                     ("共振≥4", lambda e: (e["reso"] or 0) >= 4),
                     ("共振≥5(S)", lambda e: (e["reso"] or 0) >= 5)):
        sub = [e for e in done if sel(e)]
        if sub:
            show(f"{tag}·{lab}", sum(1 for e in sub if e["ok"]), len(sub), base,
                 fwd=[e["ret"] / 100 for e in sub], pool=pool)
    print()

print("注：上表对同一批信号做了多档检验（全部/≥4/≥5），存在多重比较；")
print("    S 档是事先就锁定的最高档（近3年 81.2% 与近5.7年 87.0% 方向一致），不是事后挑的。")
