"""d99 —— 最终参数确定。

d98 暴露的冷却期问题：阈值 30 太松，下跌途中连着触发，
08-03 被 07-28 的冷却吃掉、09-14 被 09-04 的冷却吃掉。
收紧阈值能让 09-14 单独触发（09-04 分值 25.6 落在外侧）。

08-03 与 07-28/07-30 相隔只有 2~4 个交易日，属于同一波，
任何冷却 ≥5 的规则都只会标其中一个 —— 这是规则本身决定的，不是 bug。

本脚本扫阈值/冷却，看命中率与目标覆盖怎么权衡，定最终参数。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C
from d80 import build_full
from d86 import build as build_breadth
from d92 import prepare, hits_for

W0 = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2026-09-18")
H = 7
COLS = ["pos", "s1_T", "s1_L"]


def main():
    panels = build_full()
    wide = build_breadth()
    D = prepare(panels, wide)
    names = D[C.BOARD_ORDER[0]]["names"]
    for h in (5, 7):
        D = hits_for(D, h)
    ix = [names.index(c) for c in COLS]
    mx = {b: np.nanmax(D[b]["cols"].to_numpy(dtype=float)[:, ix], axis=1)
          for b in C.BOARD_ORDER}

    print(f"{'阈值':<5}{'冷却':<5}{'命中':<18}{'每板/年':<8}{'科创板2026信号'}")
    for thr in (18, 20, 22, 25, 30):
        for cool in (5, 7):
            n = ok = 0
            star = []
            for b in C.BOARD_ORDER:
                hit = D[b][f"hit{H}"]
                last = -10 ** 9
                for i in np.flatnonzero((mx[b] <= thr) & ~np.isnan(mx[b])):
                    if i < 1 or i >= len(hit) - H or i - last < cool:
                        continue
                    last = i
                    d = D[b]["index"][i]
                    if not (W0 <= d <= END):
                        continue
                    n += 1; ok += int(hit[i])
                    if b == "STAR" and d.year >= 2026:
                        star.append(str(d.date()))
            years = (END - W0).days / 365.25
            print(f"{thr:<5}{cool:<5}{ok}/{n}({ok/n*100:.1f}%){'':<8}"
                  f"{n/(6*years):<8.1f}{star}")

    # 目标日的分值与"是否会被冷却吃掉"
    print("\n科创板 2026-07~09 每日分值:")
    s = pd.Series(mx["STAR"], index=D["STAR"]["index"]).loc["2026-07-15":]
    print(s.round(1).to_string())


if __name__ == "__main__":
    main()
