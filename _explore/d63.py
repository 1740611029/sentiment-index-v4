"""稳健性体检：现有模型是不是靠某几个板块撑着？

连续四次改进尝试（成交额过滤 / 不创新低 / cf_b>0 / 相对放量）都被长历史否决，
说明框架已接近极限。这里做最后一次体检，确认现有成绩不是靠个别板块撑起来的：
  ① 留一板块法（去掉任一板块后重算）
  ② 分年度
  ③ 参数邻域扰动（K_B / CF_AMT_LO / CF_L_HI / 锚点窗口）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, store, model as M

H, TOL, COOL, THR = store.H, store.TOL, store.COOL_TRADING, store.ENTRY_THR


def ev(p, thr=THR):
    s = p["score"].to_numpy(float); cl = p["close"].to_numpy(float)
    out = []; last = -10 ** 9; n = len(s)
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if s[i - 1] <= thr and s[i] > s[i - 1]:
            last = i
            c0 = cl[i]; seg = cl[i + 1:i + H + 1]
            out.append((p.index[i], seg[-1] / c0 - 1, seg.min() / c0 - 1))
    return out


def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def line(rows, label):
    if not rows:
        print(f"  {label:<22} 无样本"); return
    # 行结构 = (board, date, ret, mdd)
    o = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
    print(f"  {label:<22} n={len(rows):>3}  命中 {o}/{len(rows):<3} {o/len(rows)*100:5.1f}%   "
          f"均T+20 {np.mean([x[2] for x in rows])*100:+6.2f}%   "
          f"最差 {min(x[3] for x in rows)*100:+7.2f}%   下界 {wilson(o,len(rows))*100:5.1f}%")


P = store.load()
ALL = {b: ev(P[b]) for b in C.BOARD_ORDER}

print("=" * 118)
print("① 留一板块法（去掉该板块后，其余 5 个板块的合计成绩）")
print("=" * 118)
base = [(b,) + e for b in C.BOARD_ORDER for e in ALL[b]]
line(base, "全部 6 板块")
for b in C.BOARD_ORDER:
    rows = [(x[0],) + (x[1], x[2], x[3]) for x in base if x[0] != b]
    line(rows, f"去掉 {C.BOARDS[b]['name']}")

print("\n" + "=" * 118)
print("② 分年度")
print("=" * 118)
for y in sorted({x[1].year for x in base}):
    line([x for x in base if x[1].year == y], f"{y} 年")

print("\n" + "=" * 118)
print("③ 参数邻域扰动（改一个参数重算，看是否有悬崖）")
print("=" * 118)
from senti import data as D
from senti import factors as F
from senti import model as M


def rebuild(k_b=None, amt_lo=None, l_hi=None, win=None):
    old = (M.K_B, M.CF_AMT_LO, M.CF_L_HI, C.MODEL["anchor_window"])
    if k_b is not None: M.K_B = k_b
    if amt_lo is not None: M.CF_AMT_LO = amt_lo
    if l_hi is not None: M.CF_L_HI = l_hi
    if win is not None:
        C.MODEL["anchor_window"] = win
        M.ANCHOR_WIN = win          # ANCHOR_WIN 是导入时读的，必须直接改模块变量
    rows = []
    try:
        for b in C.BOARD_ORDER:
            p = M.build(b)
            rows += [(b,) + e for e in ev(p)]
    finally:
        M.K_B, M.CF_AMT_LO, M.CF_L_HI = old[0], old[1], old[2]
        C.MODEL["anchor_window"] = old[3]
        M.ANCHOR_WIN = old[3]
    return rows


for tag, kw in (("基准", {}),
                ("K_B=20", dict(k_b=20.0)), ("K_B=25（现状）", dict(k_b=25.0)),
                ("K_B=30", dict(k_b=30.0)), ("K_B=35", dict(k_b=35.0)),
                ("CF_AMT_LO=0.55", dict(amt_lo=0.55)),
                ("CF_AMT_LO=0.60（现状）", dict(amt_lo=0.60)),
                ("CF_AMT_LO=0.65", dict(amt_lo=0.65)),
                ("CF_L_HI=10", dict(l_hi=10.0)),
                ("CF_L_HI=12（现状）", dict(l_hi=12.0)),
                ("CF_L_HI=15", dict(l_hi=15.0)),
                ("锚点窗口=500", dict(win=500)),
                ("锚点窗口=600", dict(win=600)),
                ("锚点窗口=750（现状）", dict(win=750)),
                ("锚点窗口=900", dict(win=900)),
                ("锚点窗口=1100", dict(win=1100)),
                ("锚点窗口=1300", dict(win=1300))):
    line(rebuild(**kw), tag)
