"""冷却期合并规则修正验证。

现状：同一簇（间隔 ≤ 20 自然日）内保留「最早」的极值点。
问题：中证1000 2024-01-22(-2.2) 与 2024-02-05(-25.6) 相隔 14 天被合并，
      保留了更早更浅的 01-22，丢掉了真底 02-05（+28.03%、最深 +6.97%）。

修正：簇内保留「最极端」的点（底部取最小分值，顶部取最大分值）。
这不是调参，是修正一个明显的取舍错误。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, model

MAIN_H, TOL, GAP = 20, 0.03, 20

print("building panels ...", flush=True)
panels = model.build_all()
print("done\n", flush=True)


def fwd(close, d, h=MAIN_H):
    i = close.index.get_loc(d)
    if i + h >= len(close):
        return None
    c0 = close.iloc[i]; seg = close.iloc[i + 1:i + h + 1]
    return seg.iloc[-1] / c0 - 1, seg.min() / c0 - 1, seg.max() / c0 - 1


def runs(s, thr, hi):
    m = (s > thr) if hi else (s < thr)
    if not m.any():
        return []
    grp = (m != m.shift()).cumsum()
    return sorted({(seg.idxmax() if hi else seg.idxmin()) for _, seg in s[m].groupby(grp[m])})


def cluster(picks, gap=GAP):
    """把间隔 ≤ gap 的极值点聚成簇"""
    out = []
    for d in picks:
        if out and (d - out[-1][-1]).days <= gap:
            out[-1].append(d)
        else:
            out.append([d])
    return out


def pick_first(picks, s, hi):
    """现状规则：保留簇内最早"""
    return [c[0] for c in cluster(picks)]


def pick_extreme(picks, s, hi):
    """修正规则：保留簇内最极端"""
    out = []
    for c in cluster(picks):
        out.append(max(c, key=lambda d: s.loc[d]) if hi else min(c, key=lambda d: s.loc[d]))
    return out


def evaluate(rule):
    bd, tp = [], []
    for b in C.BOARD_ORDER:
        p = panels[b]; s = p["score"].dropna(); cl = p["close"]
        for d in rule(runs(s, 0.0, False), s, False):
            f = fwd(cl, d)
            if f: bd.append((b, d, f[0], f[1], s.loc[d]))
        for d in rule(runs(s, 100.0, True), s, True):
            f = fwd(cl, d)
            if f: tp.append((b, d, f[0], f[2], s.loc[d]))
    return bd, tp


def report(tag, bd, tp, verbose=False):
    bok = sum(1 for x in bd if x[2] > 0 and x[3] >= -TOL)
    bnt = sum(1 for x in bd if x[3] >= -TOL)
    tok = sum(1 for x in tp if x[2] < 0 and x[3] <= TOL)
    mr = lambda a, i: (np.mean([x[i] for x in a]) * 100) if a else float("nan")
    sb = [x for x in bd if x[4] <= -3]
    print(f"  {tag:<14} 底 {bok}/{len(bd)} = {bok/len(bd)*100:3.0f}%   未被套 {bnt}/{len(bd)} = "
          f"{bnt/len(bd)*100:3.0f}%   均收益 {mr(bd,2):+6.2f}%  均最深 {mr(bd,3):+6.2f}%   |"
          f" 强确认档 {sum(1 for x in sb if x[2]>0 and x[3]>=-TOL)}/{len(sb)}"
          f" 未被套 {sum(1 for x in sb if x[3]>=-TOL)}/{len(sb)}   |  顶 {tok}/{len(tp)}")
    if verbose:
        for b, d, r, m, sc in sorted(bd, key=lambda x: (x[2] > 0 and x[3] >= -TOL, x[1])):
            ok = r > 0 and m >= -TOL
            print(f"      {'OK  ' if ok else 'FAIL'} {C.BOARDS[b]['name']:>8} {str(d.date())} "
                  f"分{sc:6.1f}  20日{r*100:+7.2f}%  最深{m*100:+7.2f}%")


print("=" * 118)
print("合并规则对比（近 3 年，T+20，容差 3%）")
print("=" * 118)
bd_a, tp_a = evaluate(pick_first)
bd_b, tp_b = evaluate(pick_extreme)
report("现状·保留最早", bd_a, tp_a)
report("修正·保留最极端", bd_b, tp_b)

print("\n" + "=" * 118)
print("修正后的完整底部明细")
print("=" * 118)
report("修正", bd_b, tp_b, verbose=True)

print("\n" + "=" * 118)
print("差异：哪些事件被换掉了")
print("=" * 118)
sa = {(x[0], x[1]) for x in bd_a}
sbb = {(x[0], x[1]) for x in bd_b}
for b, d in sorted(sa - sbb):
    print(f"  移除  {C.BOARDS[b]['name']:>8} {str(d.date())}  分{panels[b]['score'].loc[d]:6.1f}")
for b, d in sorted(sbb - sa):
    x = [y for y in bd_b if y[0] == b and y[1] == d][0]
    print(f"  新增  {C.BOARDS[b]['name']:>8} {str(d.date())}  分{x[4]:6.1f}  "
          f"20日{x[2]*100:+7.2f}%  最深{x[3]*100:+7.2f}%")
