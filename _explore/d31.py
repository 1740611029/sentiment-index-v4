"""三种「实盘可执行」的入场规则对比（全部无未来函数）。

把每次溢出按 20 自然日传递式聚成一个 episode，然后在 episode 内分别取：
  R1  首次跌破 0      —— 现在的做法
  R2  首次跌破 -3     —— 等确认再动手
  R3  最深那一天      —— 事后才知道，仅供参照（不作为可执行规则）
顶部同理：R1t 首次突破 100。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, model

MAIN_H, TOL, GAP = 20, 0.03, 20
THR_STRONG = -3.0

print("building panels ...", flush=True)
panels = model.build_all()
print("done\n", flush=True)


def fwd(close, d, h=MAIN_H):
    i = close.index.get_loc(d)
    if i + h >= len(close):
        return None
    c0 = close.iloc[i]; seg = close.iloc[i + 1:i + h + 1]
    return seg.iloc[-1] / c0 - 1, seg.min() / c0 - 1, seg.max() / c0 - 1


def episodes(s, thr, hi):
    """返回 episode 列表，每个是若干连续溢出段的 [起, 止] 日期列表。"""
    m = (s > thr) if hi else (s < thr)
    if not m.any():
        return []
    grp = (m != m.shift()).cumsum()
    segs = [(g.index[0], g.index[-1]) for _, g in m[m].groupby(grp[m])]
    segs.sort()
    eps = [[segs[0]]]
    for sg in segs[1:]:
        if (sg[0] - eps[-1][-1][1]).days <= GAP:
            eps[-1].append(sg)
        else:
            eps.append([sg])
    return eps


def first_at(s, ep, thr, hi):
    """episode 内第一次越过阈值的日期"""
    for a, b in ep:
        seg = s[a:b]
        hit = seg[seg > thr] if hi else seg[seg < thr]
        if len(hit):
            return hit.index[0]
    return None


def extreme_at(s, ep, hi):
    best, bd = None, None
    for a, b in ep:
        seg = s[a:b]
        d = seg.idxmax() if hi else seg.idxmin()
        if best is None or (s.loc[d] > best if hi else s.loc[d] < best):
            best, bd = s.loc[d], d
    return bd


def collect(mode, hi=False, thr=0.0, strong=THR_STRONG):
    out = []
    for b in C.BOARD_ORDER:
        p = panels[b]; s = p["score"].dropna(); cl = p["close"]
        for ep in episodes(s, thr, hi):
            if mode == "R1":
                d = first_at(s, ep, thr, hi)
            elif mode == "R2":
                d = first_at(s, ep, strong, False)
            else:
                d = extreme_at(s, ep, hi)
            if d is None:
                continue
            f = fwd(cl, d)
            if f:
                out.append((b, d, f[0], f[1] if not hi else f[2], s.loc[d]))
    return out


def stat(tag, rows, hi=False):
    if not rows:
        print(f"  {tag:<34} 无事件"); return
    if hi:
        ok = sum(1 for x in rows if x[2] < 0 and x[3] <= TOL)
    else:
        ok = sum(1 for x in rows if x[2] > 0 and x[3] >= -TOL)
    ntrap = sum(1 for x in rows if x[3] >= -TOL) if not hi else 0
    mr = lambda i: np.mean([x[i] for x in rows]) * 100
    extra = f"  未被套 {ntrap}/{len(rows)} = {ntrap/len(rows)*100:3.0f}%" if not hi else ""
    print(f"  {tag:<34} 命中 {ok}/{len(rows)} = {ok/len(rows)*100:3.0f}%{extra}"
          f"   均收益 {mr(2):+6.2f}%   均风险 {mr(3):+6.2f}%")


print("=" * 120)
print("底部：三种入场规则（episode 内聚，20 自然日传递式合并）")
print("=" * 120)
r1 = collect("R1"); r2 = collect("R2"); r3 = collect("R3")
stat("R1 首次跌破 0 就买（现状）", r1)
stat("R2 首次跌破 -3 才买", r2)
stat("R3 最深那天买（事后，仅参照）", r3)

print("\n" + "=" * 120)
print("R2（等到 -3 才动手）完整明细")
print("=" * 120)
for b, d, r, m, sc in sorted(r2, key=lambda x: x[1]):
    ok = r > 0 and m >= -TOL
    print(f"  {'OK  ' if ok else 'FAIL'} {C.BOARDS[b]['name']:>8} {str(d.date())}  分{sc:6.1f}  "
          f"20日{r*100:+7.2f}%  最深{m*100:+7.2f}%")

print("\n" + "=" * 120)
print("R2 留一板块 / 分时段")
print("=" * 120)
for b in C.BOARD_ORDER:
    sub = [x for x in r2 if x[0] != b]
    stat(f"去掉 {C.BOARDS[b]['name']}", sub)
half = pd.Timestamp("2025-02-01")
stat("2023-09 ~ 2025-01", [x for x in r2 if x[1] < half])
stat("2025-02 ~ 2026-09", [x for x in r2 if x[1] >= half])

print("\n" + "=" * 120)
print("R2 分板块")
print("=" * 120)
for b in C.BOARD_ORDER:
    stat(f"{C.BOARDS[b]['name']}", [x for x in r2 if x[0] == b])

print("\n" + "=" * 120)
print("顶部：R1t 首次突破 100（同一 episode 口径）")
print("=" * 120)
rt = collect("R1", hi=True, thr=100.0)
stat("顶部首次突破 100", rt, hi=True)
for b, d, r, m, sc in sorted(rt, key=lambda x: x[1]):
    ok = r < 0 and m <= TOL
    print(f"  {'OK  ' if ok else 'FAIL'} {C.BOARDS[b]['name']:>8} {str(d.date())}  分{sc:6.1f}  "
          f"20日{r*100:+7.2f}%  最高{m*100:+7.2f}%")
