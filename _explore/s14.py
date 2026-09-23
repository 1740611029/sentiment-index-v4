"""s14 —— 第七轮：SWING-3 定稿前的分档 / 分年 / 分板块 / 逐笔复核。

模型定义（候选定稿）
------------------
    SWING-3 = _pct_map( MA20 / MA20.shift(5) − 1 )        0~100，越低 = 均线跌得越急
    信号    分值 <= 10 且 分值 > 前一日（回升确认）且 冷却 4 交易日
    判定    T+7 期末 > 0 且期间最深收盘回撤 >= −3%（与 SWING / SWING-2 完全一致）

本脚本输出
--------
  A. 6 板块 × 分年 / 分板块（展示窗口 + 近5.7年 + 长历史）
  B. 共振分档（含自己在内同日触发的板块数 ≥3/≥4/≥5）
  C. 展示窗口逐笔（2023-09-20 起），看它到底抓到了什么
  D. 三模型并集的最终数字
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s8
import s9                                       # noqa: F401
import s12
from senti import config as C                    # noqa: E402
from s8 import wilson, W_ALL, W_57, W_3          # noqa: E402

MAW, LB, THR, COOL, H, TOL = 20, 5, 10.0, 4, 7, 0.03


def ms_score(E):
    fn = (lambda df: s8.pm(df["close"].rolling(MAW).mean()
                           / df["close"].rolling(MAW).mean().shift(LB) - 1))
    s8.FACTORS["__ms"] = (fn, 1)
    S = E.score("__ms")
    s8.FACTORS.pop("__ms", None)
    return S


def up1(df, i, s):
    return s[i] > s[i - 1]


def events3(E, S, w0, w1, reso=None, use_up=True):
    out = []
    for b in E.BOARDS:
        s = S[b].to_numpy(float)
        c = E.CL[b]
        idx = c.index
        e, m = E.FW[b]
        df = E.D[b]
        last = -10 ** 9
        for i in range(1, len(s)):
            if np.isnan(s[i]) or s[i] > THR or i - last < COOL:
                continue
            if not (w0 <= idx[i] <= w1) or i + H > len(e) - 1:
                continue
            if use_up and not up1(df, i, s):
                continue
            last = i
            rec = {"b": b, "date": str(idx[i].date()), "score": round(float(s[i]), 1),
                   "ok": bool(e[i] > 0 and m[i] >= -TOL),
                   "ret": round(float(e[i]) * 100, 2),
                   "risk": round(float(m[i]) * 100, 2)}
            if reso is not None:
                rv = reso.reindex([idx[i]])
                rec["reso"] = int(rv.iloc[0]) if len(rv) and not np.isnan(rv.iloc[0]) else None
            out.append(rec)
    return out


def resonance6(S):
    D = pd.DataFrame({b: S[b] for b in C.BOARD_ORDER}).sort_index()
    return (D <= THR).sum(axis=1)


def st(ev):
    n = len(ev)
    k = sum(1 for r in ev if r["ok"])
    return n, k, (100 * k / n if n else float("nan")), wilson(k, n)


def main():
    E6 = s12.Ev6()
    S = ms_score(E6)
    reso = resonance6(S)

    print("=" * 96)
    print(f"A. SWING-3（MA{MAW} 斜率{THR:g} + 回升确认 + 冷却{COOL}）6 板块表现")
    print("=" * 96)
    for tag, (w0, w1) in (("展示窗口 2023-09~2026-09", W_3),
                          ("近 5.7 年 2021-01~2026-09", W_57),
                          ("全程 2013-01~2026-09", W_ALL)):
        ev = events3(E6, S, w0, w1, reso)
        n, k, r, wl = st(ev)
        print(f"  {tag:<28} n={n:<4} 命中 {k}/{n} = {r:.1f}%  下界 {wl:.1f}%"
              f"  基线 {E6.base(w0,w1):.1f}%")

    print("\n  分年（全程）")
    ys = {}
    for r in events3(E6, S, *W_ALL, reso):
        ys.setdefault(r["date"][:4], [0, 0])
        ys[r["date"][:4]][0] += 1
        ys[r["date"][:4]][1] += 1 if r["ok"] else 0
    for y in sorted(ys):
        nn, kk = ys[y]
        mark = "★" if y in ("2015", "2016") else " "
        print(f"   {mark} {y}  {kk:>3}/{nn:<3} = {100*kk/nn:5.1f}%")

    print("\n  分板块（全程）")
    bs = {}
    for r in events3(E6, S, *W_ALL, reso):
        bs.setdefault(r["b"], [0, 0])
        bs[r["b"]][0] += 1
        bs[r["b"]][1] += 1 if r["ok"] else 0
    for b in C.BOARD_ORDER:
        if b not in bs:
            continue
        nn, kk = bs[b]
        print(f"    {C.BOARDS[b]['name']:<10} {kk:>3}/{nn:<3} = {100*kk/nn:5.1f}%")

    print("\n" + "=" * 96)
    print("B. 共振分档（近 5.7 年 / 全程）")
    print("=" * 96)
    for lab, (w0, w1) in (("近5.7年", W_57), ("全程", W_ALL)):
        ev = events3(E6, S, w0, w1, reso)
        print(f"  [{lab}]")
        print(f"    全部      n={len(ev):<4} {100*sum(r['ok'] for r in ev)/max(len(ev),1):.1f}%")
        for lo in (3, 4, 5):
            sub = [r for r in ev if (r.get("reso") or 0) >= lo]
            if not sub:
                continue
            k = sum(1 for r in sub if r["ok"])
            print(f"    共振≥{lo}    n={len(sub):<4} {100*k/len(sub):.1f}%"
                  f"  下界 {wilson(k,len(sub)):.1f}%")

    print("\n" + "=" * 96)
    print("C. 展示窗口逐笔（2023-09-20 起，6 板块）")
    print("=" * 96)
    ev = events3(E6, S, *W_3, reso)
    ev.sort(key=lambda r: r["date"])
    print(f"  {'日期':<12}{'板块':<10}{'分值':>7}{'共振':>5}{'T+7收益':>9}{'最深回撤':>9}  判定")
    for r in ev:
        print(f"  {r['date']:<12}{C.BOARDS[r['b']]['name']:<10}{r['score']:>7.1f}"
              f"{str(r.get('reso')):>5}{r['ret']:>9.2f}{r['risk']:>9.2f}"
              f"  {'✔' if r['ok'] else '✘'}")

    print("\n" + "=" * 96)
    print("D. 三模型并集（6 板块）")
    print("=" * 96)
    for lab, (w0, w1) in (("展示窗口", W_3), ("近5.7年", W_57), ("全程", W_ALL)):
        sets = {}
        for tag, nm, thr, cond in (("SWING", "pos", 22, None),
                                   ("SWING-2", "macd", 12, None),
                                   ("SWING-3", "__ms", THR, True)):
            if nm == "__ms":
                s8.FACTORS["__ms"] = (lambda df: s8.pm(
                    df["close"].rolling(MAW).mean()
                    / df["close"].rolling(MAW).mean().shift(LB) - 1), 1)
            SS = E6.score(nm)
            ev = events3(E6, SS, w0, w1, reso if nm == "__ms" else None, use_up=(cond is not None))
            sets[tag] = {(r["b"], r["date"]): r["ok"] for r in ev}
            s8.FACTORS.pop("__ms", None)
        line = []
        for tag in ("SWING", "SWING-2", "SWING-3"):
            d = sets[tag]
            line.append(f"{tag} {len(d)}/{100*sum(d.values())/max(len(d),1):.1f}%")
        U = {}
        for tag in ("SWING", "SWING-2", "SWING-3"):
            U.update(sets[tag])
        U2 = dict(sets["SWING"]); U2.update(sets["SWING-2"])
        print(f"  [{lab}] " + "　".join(line))
        print(f"     并集(前两) {len(U2)}/{100*sum(U2.values())/len(U2):.1f}%"
              f"  →  并集(三个) {len(U)}/{100*sum(U.values())/len(U):.1f}%"
              f"  (下界 {wilson(sum(U.values()),len(U)):.1f}%)")


if __name__ == "__main__":
    main()
