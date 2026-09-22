"""s2 —— 定冷却期与触发方向。

s1 的问题：07-28 触发（T+7 只有 +0.46%，且中间回撤超 −3% → 失败），
把真正更好的 08-03 / 08-04 吃掉了。
"下跌中途第一次碰到阈值"往往还没跌完；末端那次才是对的。

这里比较：冷却 3/4/5/7 × 触发方向（回升 / 无要求 / 仍在下探），
看命中率与 2026-08-03 能否被标记。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C, swing

H = swing.H
TOL = swing.TOL
W0 = pd.Timestamp(C.BACKTEST_START)


def sim(panels, thr, cool, direction):
    """direction: 'up' 回升 / 'any' 无要求 / 'down' 仍在下探"""
    reso = swing.resonance(panels)
    n = ok = pend = 0
    star = []
    for b in C.BOARD_ORDER:
        p = panels[b]
        s = p["swing"].to_numpy(dtype=float)
        c = p["close"].to_numpy(dtype=float)
        dates = p.index
        last = -10 ** 9
        for i in range(1, len(s)):
            if np.isnan(s[i]) or s[i] > thr or i - last < cool:
                continue
            if direction == "up" and not s[i] > s[i - 1]:
                continue
            if direction == "down" and not s[i] < s[i - 1]:
                continue
            last = i
            if b == "STAR" and dates[i] >= pd.Timestamp("2026-07-01"):
                star.append(str(dates[i].date()))
            if i + H > len(c) - 1:
                pend += 1
                continue
            seg = c[i + 1:i + H + 1]
            n += 1
            ok += int(seg[-1] / c[i] - 1 > 0 and seg.min() / c[i] - 1 >= -TOL)
    return n, ok, pend, star


if __name__ == "__main__":
    panels = swing.load()
    print(f"{'阈值':<5}{'冷却':<5}{'方向':<6}{'命中':<18}{'待验':<6}{'科创板2026-07后'}")
    for thr in (18, 20, 22, 25):
        for cool in (3, 4, 5, 7):
            for d in ("any", "up", "down"):
                n, ok, pend, star = sim(panels, thr, cool, d)
                if n < 25:
                    continue
                print(f"{thr:<5}{cool:<5}{d:<6}{ok}/{n}({ok/n*100:.1f}%){'':<6}{pend:<6}{star}")
