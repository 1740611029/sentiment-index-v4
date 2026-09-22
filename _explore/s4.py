"""s4 —— 持有期 h 与回撤容差 tol 的敏感性。

用户要的是"4~10 天波段"，我前面固定用 h=7（中位数）。h=5 同样在需求范围内，
而且窗口越短、被"中途回撤"打掉的概率越低。这里把 h / tol 扫一遍，
看 64.1% 里有多少是"判定口径太严"造成的，有多少是模型本身的。

同时用**两个窗口**对比：
  近 3 年（2023-09 起）       —— 页面展示窗口，但只有 ~64 个信号
  近 5.7 年（2021-01 起）     —— 样本约 2.5 倍，统计更可靠（含 2021-22 熊市）
如果两者差很多，说明近 3 年的 64% 有牛市偏乐观的成分，必须如实说。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C, swing

THR, COOL = swing.THR, swing.COOL


def fwd(close: pd.Series, h: int):
    c = close.to_numpy(dtype=float)
    n = len(c)
    g = np.full(n, np.nan); m = np.full(n, np.nan); e = np.full(n, np.nan)
    for i in range(n - h):
        seg = c[i + 1:i + h + 1]
        g[i] = seg.max()/c[i]-1; m[i] = seg.min()/c[i]-1; e[i] = seg[-1]/c[i]-1
    return g, m, e


def run(panels, h, tol, w0, w1):
    reso = swing.resonance(panels)
    n = ok = pend = 0
    r4n = r4ok = 0
    per = {}
    for b in C.BOARD_ORDER:
        p = panels[b]
        s = p["swing"].to_numpy(dtype=float)
        g, m, e = fwd(p["close"], h)
        dates = p.index
        rv = reso.reindex(dates).to_numpy(dtype=float)
        cnt = 0; cok = 0
        last = -10**9
        for i in range(1, len(s)):
            if np.isnan(s[i]) or s[i] > THR or i - last < COOL:
                continue
            last = i
            if not (w0 <= dates[i] <= w1):
                continue
            if i + h > len(s) - 1:
                pend += 1; continue
            n += 1; cnt += 1
            hit = (e[i] > 0) and (m[i] >= -tol)
            ok += int(hit); cok += int(hit)
            if (rv[i] if not np.isnan(rv[i]) else 1) >= 4:
                r4n += 1; r4ok += int(hit)
        per[b] = (cok, cnt)
    # 基线
    bn = bok = 0
    for b in C.BOARD_ORDER:
        p = panels[b]
        g, m, e = fwd(p["close"], h)
        for i in range(len(e) - h):
            if not np.isnan(e[i]):
                bn += 1; bok += int(e[i] > 0 and m[i] >= -tol)
    return {"n": n, "ok": ok, "rate": ok/n*100 if n else np.nan,
            "r4": (r4ok/r4n*100 if r4n else np.nan), "r4n": r4n,
            "pend": pend, "base": bok/bn*100 if bn else np.nan, "per": per}


if __name__ == "__main__":
    C.BACKTEST_START = "2021-01-01"          # 一次 build，锚点仍在全历史上算
    panels = swing.build_all()
    W3 = (pd.Timestamp("2023-09-20"), pd.Timestamp("2026-09-18"))
    W5 = (pd.Timestamp("2021-01-01"), pd.Timestamp("2026-09-18"))

    for tag, W in (("近3年（页面窗口）", W3), ("近5.7年（含熊市）", W5)):
        print(f"\n########## {tag} ##########")
        print(f"{'h':<4}{'容差':<6}{'命中':<20}{'共振≥4':<16}{'待验':<6}{'基线':<8}{'科创板'}")
        for h in (5, 7, 10):
            for tol in (0.03, 0.04, 0.05):
                r = run(panels, h, tol, W[0], W[1])
                st = r["per"].get("STAR", (0, 0))
                print(f"{h:<4}{tol:<6.0%}{r['ok']}/{r['n']}({r['rate']:.1f}%){'':<8}"
                      f"{r['r4']:.1f}%({r['r4n']}){'':<8}{r['pend']:<6}"
                      f"{r['base']:.1f}%{'':<4}{st[0]}/{st[1]}")
