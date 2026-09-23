"""s12 —— 第五轮：候选模型系统比较（9 宽基长历史 + 6 板块页面口径）。

前四轮的净结论
------------
  否决：长期位置（hi250/500/750，43~54%）、波动率族（atr/bbw/vol20，≈基线）、
        量能族单独用（amt_ratio/shr/amt_pct，≈基线）、分布族（skew/kurt/dnvol）。
  保留：
    (1) **ma20_slope**（MA20 的 5 日斜率分位）—— 与现 SWING/SWING-2 正交
        （J(pos)=0.05、J(macd)=0.03），近5.7年 71.3%(n=157)，并集 +2.7pp
    (2) **vconv 波动收敛确认**（近5日波动 < 近20日波动）—— 把 2015 年
        macd 的 35% 抬到 51%、pos 的 42% 抬到 54%。这是**唯一能改善
        2015 那一关**的维度，因为它过滤的正是「加速下跌」。

本脚本做三件事
------------
  A. 6 板块口径评估器（与页面完全一致，含中证2000）
  B. ma20_slope 阈值 × 确认层 的完整网格
  C. ma20_slope 的结构变体（MA 周期 / 回看期）找平台
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s8
import s9                                       # noqa: F401
from senti import config as C, data              # noqa: E402
from s8 import wilson, W_ALL, W_57, W_3          # noqa: E402


class Ev6:
    """6 板块口径（页面口径）：大盘/科创板/创业板/中证1000/中证2000/沪深300。"""
    BOARDS = C.BOARD_ORDER

    def __init__(self, h=7, tol=0.03):
        self.h, self.tol = h, tol
        self.D, self.CL, self.FW = {}, {}, {}
        for b in self.BOARDS:
            d = data.load_index(b).set_index("date")
            self.D[b] = d
            self.CL[b] = d["close"].astype(float)
            self.FW[b] = s8.fwd(self.CL[b], h)

    def score(self, name):
        fn, d = s8.FACTORS[name]
        return {b: (100.0 - fn(self.D[b])) if d < 0 else fn(self.D[b]) for b in self.BOARDS}

    def run(self, S, thr, cool, w0, w1, cond=None):
        out = []
        for b in self.BOARDS:
            s = S[b].to_numpy(float)
            c = self.CL[b]
            idx = c.index
            e, m = self.FW[b]
            df = self.D[b]
            last = -10 ** 9
            for i in range(1, len(s)):
                if np.isnan(s[i]) or s[i] > thr or i - last < cool:
                    continue
                if not (w0 <= idx[i] <= w1) or i + self.h > len(e) - 1:
                    continue
                if cond is not None and not cond(df, i, s):
                    continue
                last = i
                out.append((b, str(idx[i].date()),
                            bool(e[i] > 0 and m[i] >= -self.tol), float(e[i])))
        return out

    def base(self, w0, w1):
        o = []
        for b in self.BOARDS:
            e, m = self.FW[b]
            idx = self.CL[b].index
            for i in range(len(idx)):
                if not (w0 <= idx[i] <= w1) or i + self.h > len(e) - 1:
                    continue
                o.append(e[i] > 0 and m[i] >= -self.tol)
        return 100 * float(np.mean(o)) if o else float("nan")


def st(ev):
    n = len(ev)
    k = sum(1 for r in ev if r[2])
    return n, k, (100 * k / n if n else float("nan")), wilson(k, n)


def yr(ev, y):
    sub = [r for r in ev if r[1][:4] == y]
    if not sub:
        return "—"
    k = sum(1 for r in sub if r[2])
    return f"{k}/{len(sub)}={100*k/len(sub):.0f}%"


# ---------------------------------------------------------------- 确认条件
def mk_vconv(k_short=5, k_long=20):
    def f(df, i, s):
        c = df["close"].to_numpy(float)
        if i < k_long + 1:
            return False
        r = np.diff(c[i - k_long:i + 1]) / c[i - k_long:i]
        return r[-k_short:].std() < r.std()
    return f


def mk_decel(k_short=5, k_long=20):
    def f(df, i, s):
        c = df["close"].to_numpy(float)
        if i < k_long:
            return False
        return (c[i] / c[i - k_short] - 1) > (c[i] / c[i - k_long] - 1)
    return f


def C_up1(df, i, s):
    return s[i] > s[i - 1]


CONDS = {"无": None, "vconv": mk_vconv(), "decel": mk_decel(), "up1": C_up1}


def eval_all(E, S, thr, cool, cond, tag=""):
    r = {}
    for k, (w0, w1) in (("ALL", W_ALL), ("57", W_57), ("3", W_3)):
        ev = E.run(S, thr, cool, w0, w1, cond=cond)
        r[k] = st(ev)
        r[k + "_ev"] = ev
    r["base_ALL"] = E.base(*W_ALL)
    return r


if __name__ == "__main__":
    E9 = s8.Evaluator()
    E6 = Ev6()
    print(f"6 板块口径基线：近5.7年 {E6.base(*W_57):.1f}%　近3年 {E6.base(*W_3):.1f}%")
    print(f"9 宽基基线：全程 {E9.base(*W_ALL):.1f}%　近5.7年 {E9.base(*W_57):.1f}%"
          f"　近3年 {E9.base(*W_3):.1f}%\n")

    print("=" * 118)
    print("A. 现模型对照（两个口径）")
    print("=" * 118)
    print(f"  {'模型':<22}{'口径':<8}{'n':>6}{'命中':>8}{'下界':>7}"
          f"{'2015':>13}{'2016':>11}{'近3年':>16}")
    for tag, nm, thr, cool in (("SWING pos≤22", "pos", 22, 4),
                               ("SWING-2 macd≤12", "macd", 12, 4)):
        for lab, E in (("9宽基", E9), ("6板块", E6)):
            S = E.score(nm)
            ev = E.run(S, thr, cool, *W_ALL)
            n, k, r, wl = st(ev)
            ev3 = E.run(S, thr, cool, *W_3)
            n3, k3, r3, _ = st(ev3)
            print(f"  {tag:<22}{lab:<8}{n:>6}{r:>7.1f}%{wl:>7.1f}"
                  f"{yr(ev,'2015'):>13}{yr(ev,'2016'):>11}{f'{r3:.1f}%({n3})':>16}")

    print("\n" + "=" * 118)
    print("B. ma20_slope 阈值 × 确认层（9 宽基，全程 / 近5.7年 / 近3年）")
    print("=" * 118)
    S_ms = E9.score("ma20_slope")
    print(f"  {'阈值':>5}{'确认':<8}{'n':>6}{'命中':>8}{'下界':>7}{'2015':>13}{'2016':>11}"
          f"{'近5.7年':>16}{'近3年':>16}")
    for thr in (3, 5, 6, 8, 10, 12, 15):
        for cname, cond in CONDS.items():
            ev = E9.run(S_ms, thr, 4, *W_ALL, cond=cond)
            n, k, r, wl = st(ev)
            if n < 40:
                continue
            ev57 = E9.run(S_ms, thr, 4, *W_57, cond=cond)
            ev3 = E9.run(S_ms, thr, 4, *W_3, cond=cond)
            n57, k57, r57, _ = st(ev57)
            n3, k3, r3, _ = st(ev3)
            print(f"  {thr:>5g}{cname:<8}{n:>6}{r:>7.1f}%{wl:>7.1f}"
                  f"{yr(ev,'2015'):>13}{yr(ev,'2016'):>11}"
                  f"{f'{r57:.1f}%({n57})':>16}{f'{r3:.1f}%({n3})':>16}")

    print("\n" + "=" * 118)
    print("C. ma20_slope 结构变体（MA 周期 × 回看期），阈值固定为「让全程 n≈220」的水平")
    print("=" * 118)
    print(f"  {'MA窗':>6}{'回看':>5}{'阈值':>6}{'n':>6}{'命中':>8}{'下界':>7}{'2015':>13}"
          f"{'2016':>11}{'近5.7年':>16}")
    for maw in (10, 15, 20, 25, 30, 40, 60):
        for lb in (3, 5, 8, 10):
            fn = (lambda df, m=maw, l=lb: s8.pm(
                df["close"].rolling(m).mean() / df["close"].rolling(m).mean().shift(l) - 1))
            s8.FACTORS["__tmp"] = (fn, 1)
            S = E9.score("__tmp")
            for thr in (3, 5, 8, 10, 12, 15):
                ev = E9.run(S, thr, 4, *W_ALL)
                n, k, r, wl = st(ev)
                if n < 180:
                    continue
                ev57 = E9.run(S, thr, 4, *W_57)
                n57, k57, r57, _ = st(ev57)
                print(f"  {maw:>6}{lb:>5}{thr:>6g}{n:>6}{r:>7.1f}%{wl:>7.1f}"
                      f"{yr(ev,'2015'):>13}{yr(ev,'2016'):>11}{f'{r57:.1f}%({n57})':>16}")
                break
    s8.FACTORS.pop("__tmp", None)
