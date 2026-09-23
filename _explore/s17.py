"""s17 —— 第九轮（收口）：三模型并集之后，还有没有第四个模型？

用户要求「一直推理直到推理不出更好的模型」。前八轮已扫完 49 个因子族，
结论是 ma20_slope + up1（= SWING-3）是唯一三窗口都不弱于现模型的候选。
本轮把「该不该再加第 4 个模型」做成一个**可判定的检验**，给出收敛证明。

判据（严格，防多重检验假阳性）
------------------------------
候选 C 合格 ⟺ 并集(3 模型 ∪ C) 同时满足
  ① 9 宽基长历史全程 2013-2026 命中率  ≥ 并集(3 模型)
  ② 近 5.7 年 命中率                    ≥ 并集(3 模型)
  ③ 近 3 年 命中率                      ≥ 并集(3 模型)
  ④ 上述三个至少一个**严格更高**（否则加了等于没加）
  ⑤ 2015 年命中数不劣于并集(3 模型)
要求三个窗口同时不劣，是本项目对抗多重检验的唯一手段：
单窗口「更好」在 46 个因子 × 7 个阈值里必然撞上，三窗口同时更好极难偶然发生。

四个部分
--------
A. 9 宽基口径：三模型代理并集（pos≤22 ∪ macd≤12 ∪ ma20_slope≤10+up1）的三窗口基准
B. 全因子注册表逐个加入并集，扫阈值 × 确认层，找过关者
C. 过滤层（vconv / decel / up1）加在现有并集上，看能否提升
D. 收口结论
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s8
import s9                                        # noqa: F401  注册 s9 扩充因子
import s10                                       # noqa: F401  注册 s10 扩充因子
from s8 import wilson, W_ALL, W_57, W_3, BOARDS  # noqa: E402

WINS = (("全程", W_ALL), ("近5.7年", W_57), ("近3年", W_3))

# 现役三模型的代理参数（与 senti/swing*.py 一致）
INC = [("pos", 22, "无"), ("macd", 12, "无"), ("ma20_slope", 10, "up1")]

THRS = [3, 5, 8, 10, 12, 15, 20]


# ---------------------------------------------------------------- 基础工具
def C_up1(df, i, s):
    return s[i] > s[i - 1]


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


COND_FN = {"无": None, "up1": C_up1, "vconv": mk_vconv(), "decel": mk_decel()}


def sig_dates(E, S, thr, cool=4, cond=None):
    """全历史取信号日（不按窗口截断），返回 {board: {date: (ok, ret)}}。

    与 E.run 同规则（逐日模拟、冷却、只用当日及历史），只是不做窗口过滤，
    这样同一份信号可以在三个窗口上切片统计，省掉三倍计算。
    """
    out = {}
    for b in BOARDS:
        s = S[b].to_numpy(float)
        idx = E.CL[b].index
        e, m = E.FW[b]
        df = E.D[b]
        last = -10 ** 9
        d = {}
        for i in range(1, len(s)):
            if np.isnan(s[i]) or s[i] > thr or i - last < cool:
                continue
            if i + E.h > len(e) - 1:
                continue
            if cond is not None and not cond(df, i, s):
                continue
            last = i
            d[str(idx[i].date())] = (bool(e[i] > 0 and m[i] >= -E.tol), float(e[i]))
        out[b] = d
    return out


def ustat(uni, w0, w1):
    """并集信号在窗口内的 (n, 命中数, 命中率, wilson 下界)。"""
    n = k = 0
    for b, d in uni.items():
        for dt, (ok, _) in d.items():
            if w0 <= pd.Timestamp(dt) <= w1:
                n += 1
                k += int(ok)
    return n, k, (100 * k / n if n else float("nan")), wilson(k, n)


def ystat(uni, y):
    n = k = 0
    for b, d in uni.items():
        for dt, (ok, _) in d.items():
            if dt[:4] == y:
                n += 1
                k += int(ok)
    return k, n


def merge(*sets):
    out = {}
    for S in sets:
        for b, d in S.items():
            out.setdefault(b, {}).update(d)
    return out


def line(tag, uni):
    cells = []
    for lab, (w0, w1) in WINS:
        n, k, r, wl = ustat(uni, w0, w1)
        cells.append(f"{r:5.1f}%({n:>3})")
    k15, n15 = ystat(uni, "2015")
    k16, n16 = ystat(uni, "2016")
    return (f"  {tag:<26}{cells[0]:>14}{cells[1]:>14}{cells[2]:>14}"
            f"{f'{k15}/{n15}':>10}{f'{k16}/{n16}':>9}")


def head():
    print(f"  {'候选':<26}{'全程':>14}{'近5.7年':>14}{'近3年':>14}{'2015':>10}{'2016':>9}")
    print("  " + "-" * 112)


# ---------------------------------------------------------------- 主流程
def main():
    E = s8.Evaluator()
    print("9 宽基随机基线："
          + "　".join(f"{lab} {E.base(*w):.1f}%" for lab, w in WINS))

    print("\n" + "=" * 114)
    print("A. 现役三模型代理并集（基准）")
    print("=" * 114)
    head()
    inc = merge(*[sig_dates(E, E.score(nm), thr, 4, COND_FN[cn])
                  for nm, thr, cn in INC])
    print(line("并集(3 模型)", inc))
    for nm, thr, cn in INC:
        print(line(f"  · {nm}≤{thr:g}{'+' + cn if cn != '无' else ''}",
                   sig_dates(E, E.score(nm), thr, 4, COND_FN[cn])))
    ref = {lab: ustat(inc, w0, w1)[2] for lab, (w0, w1) in WINS}
    k15_ref, n15_ref = ystat(inc, "2015")

    print("\n" + "=" * 114)
    print("B. 逐个候选因子加入并集（阈值 × 确认层），只打印「过关」者")
    print("    过关 = 三个窗口命中率都不低于基准，且至少一个严格更高，且 2015 命中数不劣")
    print("=" * 114)
    head()
    names = [n for n in s8.FACTORS if n not in ("pos", "macd", "ma20_slope")]
    passed, top = [], []
    for nm in names:
        S = E.score(nm)
        for cn, cond in COND_FN.items():
            for thr in THRS:
                cand = sig_dates(E, S, thr, 4, cond)
                uni = merge(inc, cand)
                rs = {lab: ustat(uni, w0, w1)[2] for lab, (w0, w1) in WINS}
                k15, n15 = ystat(uni, "2015")
                strict = any(rs[lab] > ref[lab] + 1e-9 for lab, _ in WINS)
                okall = all(rs[lab] >= ref[lab] - 1e-9 for lab, _ in WINS)
                if okall and strict and k15 >= k15_ref:
                    passed.append((nm, cn, thr, uni))
                    print(line(f"★ {nm}≤{thr:g}{'+' + cn if cn != '无' else ''}", uni))
                top.append((rs["近3年"], nm, cn, thr, uni))
    if not passed:
        print("  （无）")

    print("\n" + "=" * 114)
    print("B2. 近 3 年并集命中率前 12 名（含未过关者，看趋势）")
    print("=" * 114)
    head()
    for r3, nm, cn, thr, uni in sorted(top, key=lambda x: -x[0])[:12]:
        print(line(f"{nm}≤{thr:g}{'+' + cn if cn != '无' else ''}", uni))
    print(line("【基准】并集(3 模型)", inc))

    print("\n" + "=" * 114)
    print("C. 过滤层加在现有并集上（不新增模型，只筛掉一部分信号）")
    print("=" * 114)
    head()
    for cn in ("vconv", "decel"):
        cond = COND_FN[cn]
        keep = {}
        for b, d in inc.items():
            df = E.D[b]
            s = np.zeros(len(df))
            idx = df.index
            kd = {}
            for dt, v in d.items():
                i = idx.get_loc(pd.Timestamp(dt))
                if cond(df, i, s):
                    kd[dt] = v
            keep[b] = kd
        print(line(f"并集 ∩ {cn}", keep))
    print(line("【基准】并集(3 模型)", inc))

    print("\n" + "=" * 114)
    print("D. 收口结论")
    print("=" * 114)
    print(f"  基准并集：全程 {ref['全程']:.1f}%　近5.7年 {ref['近5.7年']:.1f}%"
          f"　近3年 {ref['近3年']:.1f}%　2015 {k15_ref}/{n15_ref}")
    print(f"  扫描规模：{len(names)} 个候选因子 × {len(COND_FN)} 种确认层 × {len(THRS)} 个阈值"
          f" = {len(names) * len(COND_FN) * len(THRS)} 组")
    print(f"  过关者：{len(passed)} 组"
          + ("" if passed else "　→ 三窗口同时不劣且至少一窗更高的候选不存在，搜索收敛"))


if __name__ == "__main__":
    main()
