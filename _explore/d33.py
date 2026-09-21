"""L6（昨 ≤ −6 且今日回升）稳健性检验：100% 是真信号还是撞上的？

1) 阈值敏感性：X 从 −3 扫到 −15，看是不是 plateau 还是尖峰
2) 回升力度变体：只要回升 / 回升≥1分 / ≥2分 / 且今日仍<0
3) L6 的留一板块 / 分时段 / 分板块
4) 对照：不加「回升」条件会怎样（证明回升条件真的有价值，不是白加的）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, model

H, TOL, COOL = 20, 0.03, 20

print("building panels ...", flush=True)
panels = model.build_all()
print("done\n", flush=True)


def simulate(fire):
    out = []
    for b in C.BOARD_ORDER:
        p = panels[b]
        s = p["score"].to_numpy(dtype=float)
        c = p["close"].to_numpy(dtype=float)
        dates = p.index
        n = len(s); last = -10 ** 9
        for i in range(1, n - H):
            if i - last < COOL: continue
            if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
            if not fire(s[i - 1], s[i]): continue
            last = i
            c0 = c[i]; seg = c[i + 1:i + H + 1]
            out.append((b, dates[i], seg[-1] / c0 - 1, seg.min() / c0 - 1, s[i]))
    return out


def stat(tag, rows):
    if not rows:
        print(f"  {tag:<38} 无事件"); return
    ok = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
    nt = sum(1 for x in rows if x[3] >= -TOL)
    mr = lambda i: np.mean([x[i] for x in rows]) * 100
    print(f"  {tag:<38} 命中 {ok}/{len(rows)} = {ok/len(rows)*100:3.0f}%   "
          f"未被套 {nt}/{len(rows)} = {nt/len(rows)*100:3.0f}%   "
          f"均收益 {mr(2):+6.2f}%   均最深 {mr(3):+6.2f}%")


print("=" * 126)
print("【1】阈值敏感性：昨 ≤ X 且今日回升")
print("=" * 126)
for x in [-2, -3, -4, -5, -6, -7, -8, -10, -12, -15]:
    stat(f"X = {x}", simulate(lambda p, c, x=x: p <= x and c > p))

print("\n" + "=" * 126)
print("【2】回升力度变体（X = −6）")
print("=" * 126)
V = {
    "只要回升":           lambda p, c: p <= -6 and c > p,
    "回升 ≥ 1 分":        lambda p, c: p <= -6 and c - p >= 1,
    "回升 ≥ 2 分":        lambda p, c: p <= -6 and c - p >= 2,
    "回升 ≥ 3 分":        lambda p, c: p <= -6 and c - p >= 3,
    "回升 且 今日仍 < 0":   lambda p, c: p <= -6 and c > p and c < 0,
    "回升 且 今日仍 ≤ -3": lambda p, c: p <= -6 and c > p and c <= -3,
}
for tag, f in V.items():
    stat(tag, simulate(f))

print("\n" + "=" * 126)
print("【3】对照：不加「回升」条件（证明回升条件有价值）")
print("=" * 126)
for x in [-6, -8, -10]:
    stat(f"昨 > {x} 且今日 ≤ {x}（破线即买）", simulate(lambda p, c, x=x: p > x and c <= x))
    stat(f"昨 ≤ {x} 且今日回升（等掉头）", simulate(lambda p, c, x=x: p <= x and c > p))

print("\n" + "=" * 126)
print("【4】L6（昨 ≤ −6 且今日回升）稳健性")
print("=" * 126)
r6 = simulate(lambda p, c: p <= -6 and c > p)
stat("全部 6 板块", r6)
print()
for b in C.BOARD_ORDER:
    stat(f"去掉 {C.BOARDS[b]['name']}", [x for x in r6 if x[0] != b])
print()
half = pd.Timestamp("2025-02-01")
stat("2023-09 ~ 2025-01", [x for x in r6 if x[1] < half])
stat("2025-02 ~ 2026-09", [x for x in r6 if x[1] >= half])
print()
for b in C.BOARD_ORDER:
    stat(f"{C.BOARDS[b]['name']}", [x for x in r6 if x[0] == b])

print("\n" + "=" * 126)
print("【5】L6 完整明细")
print("=" * 126)
for b, d, r, m, sc in sorted(r6, key=lambda x: x[1]):
    ok = r > 0 and m >= -TOL
    print(f"  {'OK  ' if ok else 'FAIL'} {C.BOARDS[b]['name']:>8} {str(d.date())}  当日分{sc:6.1f}  "
          f"20日{r*100:+7.2f}%  最深{m*100:+7.2f}%")

print("\n" + "=" * 126)
print("【6】对照：基线（任意一天买入）")
print("=" * 126)
allr = []
for b in C.BOARD_ORDER:
    p = panels[b]; c = p["close"].to_numpy(dtype=float)
    for i in range(len(c) - H):
        c0 = c[i]; seg = c[i + 1:i + H + 1]
        allr.append((seg[-1] / c0 - 1, seg.min() / c0 - 1))
ok = sum(1 for r, m in allr if r > 0 and m >= -TOL)
nt = sum(1 for r, m in allr if m >= -TOL)
print(f"  随机一天买入  命中 {ok}/{len(allr)} = {ok/len(allr)*100:.0f}%   "
      f"未被套 {nt}/{len(allr)} = {nt/len(allr)*100:.0f}%")
