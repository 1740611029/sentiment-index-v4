"""找「真实时可执行」的底部入场规则。

前面 d31 证明：取「每段最深那天」是事后视角（26/27=96%），实盘做不到；
首次跌破 0 / 首次跌破 -3 只有 78% / 81%。

这里模拟逐日走一遍，只用「到今天为止」的信息，
入场后 20 个交易日内不再入场（冷却），统一比较：

  L1  昨日 ≥ 0，今日 < 0                      → 今日收盘买（破线即买）
  L2  昨日 > -3，今日 ≤ -3                    → 今日收盘买
  L3  昨日 ≤ -3，今日 > 昨日                  → 今日收盘买（确认掉头）
  L4  昨日 ≤ 0， 今日 > 昨日                  → 今日收盘买
  L5  昨日 ≤ -3，今日 − 昨日 ≥ 2              → 今日收盘买（掉头且力度够）
  L6  昨日 ≤ -6，今日 > 昨日                  → 今日收盘买
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
    """fire(prev, cur) -> bool；返回逐条入场记录"""
    out = []
    for b in C.BOARD_ORDER:
        p = panels[b]
        s = p["score"].to_numpy(dtype=float)
        c = p["close"].to_numpy(dtype=float)
        dates = p.index
        n = len(s)
        last = -10 ** 9
        for i in range(1, n - H):
            if i - last < COOL:
                continue
            if np.isnan(s[i]) or np.isnan(s[i - 1]):
                continue
            if not fire(s[i - 1], s[i]):
                continue
            last = i
            c0 = c[i]; seg = c[i + 1:i + H + 1]
            ret = seg[-1] / c0 - 1
            mdd = seg.min() / c0 - 1
            out.append((b, dates[i], ret, mdd, s[i]))
    return out


def stat(tag, rows):
    if not rows:
        print(f"  {tag:<40} 无事件"); return
    ok = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
    nt = sum(1 for x in rows if x[3] >= -TOL)
    mr = lambda i: np.mean([x[i] for x in rows]) * 100
    print(f"  {tag:<40} 命中 {ok}/{len(rows)} = {ok/len(rows)*100:3.0f}%   "
          f"未被套 {nt}/{len(rows)} = {nt/len(rows)*100:3.0f}%   "
          f"均收益 {mr(2):+6.2f}%   均最深 {mr(3):+6.2f}%")


RULES = {
    "L1 破 0 即买":            lambda p, c: p >= 0 and c < 0,
    "L2 破 -3 即买":           lambda p, c: p > -3 and c <= -3,
    "L3 昨≤-3 且今日回升":       lambda p, c: p <= -3 and c > p,
    "L4 昨≤0  且今日回升":       lambda p, c: p <= 0 and c > p,
    "L5 昨≤-3 且回升≥2分":      lambda p, c: p <= -3 and c - p >= 2,
    "L6 昨≤-6 且今日回升":       lambda p, c: p <= -6 and c > p,
    "L7 昨≤-3 且今日回升且昨<今-1": lambda p, c: p <= -3 and c > p and c < 0,
}

print("=" * 128)
print("真实时规则逐日模拟（入场后 20 交易日冷却；只用当日及历史信息）")
print("=" * 128)
res = {}
for tag, f in RULES.items():
    r = simulate(f)
    res[tag] = r
    stat(tag, r)

BEST = "L3 昨≤-3 且今日回升"
print("\n" + "=" * 128)
print(f"最优规则 {BEST} 的完整明细")
print("=" * 128)
for b, d, r, m, sc in sorted(res[BEST], key=lambda x: (x[2] > 0 and x[3] >= -TOL, x[1])):
    ok = r > 0 and m >= -TOL
    print(f"  {'OK  ' if ok else 'FAIL'} {C.BOARDS[b]['name']:>8} {str(d.date())}  分{sc:6.1f}  "
          f"20日{r*100:+7.2f}%  最深{m*100:+7.2f}%")

print("\n" + "=" * 128)
print(f"{BEST} 稳健性：留一板块 / 分时段 / 分板块")
print("=" * 128)
rows = res[BEST]
for b in C.BOARD_ORDER:
    stat(f"去掉 {C.BOARDS[b]['name']}", [x for x in rows if x[0] != b])
half = pd.Timestamp("2025-02-01")
stat("2023-09 ~ 2025-01", [x for x in rows if x[1] < half])
stat("2025-02 ~ 2026-09", [x for x in rows if x[1] >= half])
print()
for b in C.BOARD_ORDER:
    stat(f"{C.BOARDS[b]['name']}", [x for x in rows if x[0] == b])
