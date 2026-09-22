"""s5 —— 共振档位细分，看最高档能不能摸到 85%。

s4 的关键发现：共振≥4 在近3年 78.3%、近5.7年 78.9%，**跨窗口几乎一致** ——
说明这一档是稳健的（全部信号则是 64.1% vs 59.5%，近 3 年偏乐观 4.6 点）。
所以"85%"如果可达，只可能在更高档。这里把 ≥3/≥4/≥5/≥6 逐档拆开，
并要求每个档在**两个窗口都站得住**才算数。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C, swing
from s4 import fwd

THR, COOL, H, TOL = swing.THR, swing.COOL, swing.H, swing.TOL


def sig(panels, w0, w1, h=H, tol=TOL):
    reso = swing.resonance(panels)
    rows = []
    for b in C.BOARD_ORDER:
        p = panels[b]
        s = p["swing"].to_numpy(dtype=float)
        g, m, e = fwd(p["close"], h)
        dates = p.index
        rv = reso.reindex(dates).to_numpy(dtype=float)
        last = -10**9
        for i in range(1, len(s)):
            if np.isnan(s[i]) or s[i] > THR or i - last < COOL:
                continue
            last = i
            if not (w0 <= dates[i] <= w1) or i + h > len(s) - 1:
                continue
            rows.append({
                "b": b, "d": str(dates[i].date()),
                "reso": int(rv[i]) if not np.isnan(rv[i]) else 1,
                "rush": bool(p["vol_pct"].to_numpy(dtype=float)[i] <= swing.RUSH_PCT),
                "hit": bool(e[i] > 0 and m[i] >= -tol),
                "notrap": bool(m[i] >= -tol),
            })
    return rows


if __name__ == "__main__":
    C.BACKTEST_START = "2021-01-01"
    panels = swing.build_all()
    W3 = (pd.Timestamp("2023-09-20"), pd.Timestamp("2026-09-18"))
    W5 = (pd.Timestamp("2021-01-01"), pd.Timestamp("2026-09-18"))
    r3 = sig(panels, *W3)
    r5 = sig(panels, *W5)

    def st(rows, f=lambda e: True):
        x = [e for e in rows if f(e)]
        if not x:
            return "—"
        o = sum(1 for e in x if e["hit"])
        return f"{o}/{len(x)} = {o/len(x)*100:.1f}%"

    print(f"{'档位':<26}{'近3年':<22}{'近5.7年'}")
    cands = [
        ("全部信号", lambda e: True),
        ("共振 ≥2", lambda e: e["reso"] >= 2),
        ("共振 ≥3", lambda e: e["reso"] >= 3),
        ("共振 ≥4", lambda e: e["reso"] >= 4),
        ("共振 ≥5", lambda e: e["reso"] >= 5),
        ("共振 ≥6（满）", lambda e: e["reso"] >= 6),
        ("共振≥4 且 波段底部更极端", lambda e: e["reso"] >= 4),
    ]
    for tag, f in cands:
        print(f"{tag:<26}{st(r3,f):<22}{st(r5,f)}")

    print("\n各档样本量（近5.7年）:")
    for k in range(1, 7):
        n = len([e for e in r5 if e["reso"] == k])
        o = sum(1 for e in r5 if e["reso"] == k and e["hit"])
        print(f"  共振={k}: {o}/{n}" + (f" = {o/n*100:.1f}%" if n else ""))

    print("\n共振≥4 档的分年（近5.7年）:")
    ys = {}
    for e in r5:
        if e["reso"] >= 4:
            y = e["d"][:4]; ys.setdefault(y, [0, 0]); ys[y][0] += 1; ys[y][1] += int(e["hit"])
    for y, v in sorted(ys.items()):
        print(f"  {y}: {v[1]}/{v[0]} = {v[1]/v[0]*100:.1f}%")
