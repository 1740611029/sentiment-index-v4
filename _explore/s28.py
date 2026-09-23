"""s28 —— 第二十轮：低波动过滤的「提升」是不是机械假象？

s27 的 B 段
----------
「v20 低半（波动小）」过滤后：6 板块 70.9→72.4%、9 宽基 58.7→68.2%（+9.5pp）
「dd60 ≥ −15%」：6 板块 70.9→74.3%、9 宽基 58.7→63.8%

看着是两个样本同时提升。**但先别高兴 —— 判定口径里藏着一个机械耦合。**

判定 = 「T+7 期末 > 0」**且**「期间最深收盘回撤 ≥ −3%」。
在**低波动**环境里，7 天内跌 3% 的概率本来就低得多 ——
也就是说「低波动」这个过滤条件，直接放松了**判定标准的后半句**。
这不是预测能力变强，是**判卷标准跟着考生的波动率一起变松了**。

（这与本 skill 的第一条纪律是同一件事：判定标准必须固定、不能跟着信号变。）

必须做的三件事
------------
A. 把判定拆成两个分量：方向（T+7>0）与不套（回撤≥−3%）。
   若低波动过滤只提升「不套」而方向没提升 → 机械假象。
B. **条件基线**：低波动时期「随便买一天」的基线本身就更高。
   必须与**同波动环境**的基线比，而不是与全样本 48.6% 比。
C. 阈值扫描 + 2015/2016 关。若 A/B 任一不过，判定为机械假象，不收编。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s8
import s9                                        # noqa: F401
import s10                                       # noqa: F401
import s12
import s17
import s18
import s24
import s27
from s8 import wilson, W_ALL, W_57, W_3, BOARDS  # noqa: E402

WINS = (("全程", W_ALL), ("近5.7年", W_57), ("近3年", W_3))


def cond_baseline(E, S, thr, w0, w1, lo=None, hi=None, h=7):
    """同波动环境基线：在该环境下「随便挑一天买入」满足判定的比例。

    环境由 close 的 20 日已实现波动率界定（与信号用的 v20 同一个量）。
    返回 (方向命中率, 不套率, 联合命中率, n)
    """
    dirs, traps, both, n = 0, 0, 0, 0
    for b in BOARDS:
        c = E.CL[b]
        a = c.to_numpy(float)
        idx = c.index
        e, m = E.FW[b]
        v = pd.Series(a).pct_change().rolling(20).std().to_numpy()
        for i in range(20, len(a) - h):
            if not (w0 <= idx[i] <= w1) or i + h > len(e) - 1:
                continue
            if np.isnan(v[i]):
                continue
            if lo is not None and not (lo <= v[i] < hi):
                continue
            if hi is not None and lo is None and not (v[i] < hi):
                continue
            n += 1
            d = e[i] > 0
            t = m[i] >= -E.tol
            dirs += int(d); traps += int(t); both += int(d and t)
    if not n:
        return None
    return (100 * dirs / n, 100 * traps / n, 100 * both / n, n)


def sig_split(evs):
    """把信号结果拆成方向 / 不套 / 联合三个分量。"""
    done = [e for e in evs if e.get("ok") is not None]
    n = len(done)
    if not n:
        return None
    d = sum(1 for e in done if e["ret"] > 0)
    t = sum(1 for e in evs if e.get("notrap"))
    return (100 * d / n, 100 * t / n, 100 * sum(1 for e in done if e["ok"]) / n, n)


def main():
    from senti import config as C
    from senti import store, swing, swing2, swing3

    sp, s2p, s3p = store.swing_panels(), store.swing2_panels(), store.swing3_panels()
    sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
    panels = store.load()
    E = s8.Evaluator()
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])
    lev = [(b, {"date": dt, "ok": ok, "ret": rt, "notrap": ok or True})
           for b, d in inc.items() for dt, (ok, rt) in d.items()]
    # notrap 未记录在 sig_dates 里，重算：用 E.FW
    lev = []
    for b, d in inc.items():
        idx = E.CL[b].index
        e, m = E.FW[b]
        for dt, (ok, rt) in d.items():
            i = idx.get_loc(pd.Timestamp(dt))
            lev.append((b, {"date": dt, "ok": ok, "ret": rt,
                            "notrap": bool(m[i] >= -E.tol)}))

    u_ev = [(b, e) for b in C.BOARD_ORDER
            for e in store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r)]

    print("=" * 104)
    print("A. 判定拆分量：低波动过滤提升的是「方向」还是「不套」？")
    print("=" * 104)
    print(f"  {'口径':<34}{'方向(T+7>0)':>13}{'不套(≥−3%)':>13}{'联合命中':>10}{'件数':>7}")
    for tag, evs, thr_v in (("并集 · 无过滤", u_ev, None),
                            ("并集 · v20<0.018", u_ev, 0.018),
                            ("并集 · v20<0.015", u_ev, 0.015)):
        use = []
        for b, e in evs:
            if thr_v is None:
                use.append((b, e)); continue
            p = panels[b]
            i = p.index.get_loc(pd.Timestamp(e["date"]))
            f = s27.feats(p, i, None)
            if f and f["v20"] < thr_v:
                use.append((b, e))
        s_ = sig_split([e for _, e in use])
        if s_:
            print(f"  {tag:<34}{s_[0]:>12.1f}%{s_[1]:>12.1f}%{s_[2]:>9.1f}%{s_[3]:>7}")
    print()
    for tag, evs, thr_v in (("9宽基并集 · 无过滤", lev, None),
                            ("9宽基并集 · v20<0.018", lev, 0.018),
                            ("9宽基并集 · v20<0.015", lev, 0.015)):
        use = []
        for b, e in evs:
            if thr_v is None:
                use.append((b, e)); continue
            df = E.D[b]
            i = df.index.get_loc(pd.Timestamp(e["date"]))
            f = s27.feats(df, i, None)
            if f and f["v20"] < thr_v:
                use.append((b, e))
        s_ = sig_split([e for _, e in use])
        if s_:
            print(f"  {tag:<34}{s_[0]:>12.1f}%{s_[1]:>12.1f}%{s_[2]:>9.1f}%{s_[3]:>7}")

    print("\n" + "=" * 104)
    print("B. 条件基线：低波动时期「随便买一天」本来就有多好？（9 宽基 · 全程）")
    print("=" * 104)
    print(f"  {'波动环境(v20)':<22}{'方向':>10}{'不套':>10}{'联合':>10}{'样本天':>10}")
    for lo, hi, lab in ((0, 0.012, "< 1.2%"), (0.012, 0.018, "1.2~1.8%"),
                        (0.018, 0.025, "1.8~2.5%"), (0.025, 9, "≥ 2.5%"),
                        (0, 9, "全样本")):
        r = cond_baseline(E, None, None, *W_ALL, lo=lo, hi=hi)
        if r:
            print(f"  {lab:<22}{r[0]:>9.1f}%{r[1]:>9.1f}%{r[2]:>9.1f}%{r[3]:>10}")

    print("\n" + "=" * 104)
    print("C. 信号 vs 同环境基线（净优势）—— 这才是低波动过滤的真实价值")
    print("=" * 104)
    print(f"  {'口径':<24}{'信号命中':>10}{'同环境基线':>12}{'净优势':>10}{'件数':>8}")
    for lab, evs, getf in (("9宽基 · 无过滤", lev, None),
                           ("9宽基 · v20<0.018", lev, 0.018),
                           ("9宽基 · v20<0.015", lev, 0.015),
                           ("9宽基 · v20<0.012", lev, 0.012)):
        use = []
        for b, e in evs:
            if getf is None:
                use.append((b, e)); continue
            df = E.D[b]
            i = df.index.get_loc(pd.Timestamp(e["date"]))
            f = s27.feats(df, i, None)
            if f and f["v20"] < getf:
                use.append((b, e))
        s_ = sig_split([e for _, e in use])
        if not s_:
            continue
        lo, hi = (0, getf) if getf else (0, 9)
        r = cond_baseline(E, None, None, *W_ALL, lo=lo, hi=hi)
        print(f"  {lab:<24}{s_[2]:>9.1f}%{r[2]:>11.1f}%{s_[2]-r[2]:>+9.1f}pp{s_[3]:>8}")

    print("\n" + "=" * 104)
    print("D. 阈值平台 + 2015/2016 关（9 宽基，全程）")
    print("=" * 104)
    print(f"  {'v20 上限':>10}{'件数':>8}{'联合命中':>10}{'同环境基线':>12}{'净优势':>10}"
          f"{'2015':>9}{'2016':>9}")
    for cap in (0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.025):
        use = []
        for b, e in lev:
            df = E.D[b]
            i = df.index.get_loc(pd.Timestamp(e["date"]))
            f = s27.feats(df, i, None)
            if f and f["v20"] < cap:
                use.append((b, e))
        s_ = sig_split([e for _, e in use])
        if not s_ or s_[3] < 20:
            continue
        r = cond_baseline(E, None, None, *W_ALL, lo=0, hi=cap)
        y15 = [e for _, e in use if e["date"][:4] == "2015"]
        y16 = [e for _, e in use if e["date"][:4] == "2016"]
        f15 = f"{sum(1 for e in y15 if e['ok'])}/{len(y15)}" if y15 else "—"
        f16 = f"{sum(1 for e in y16 if e['ok'])}/{len(y16)}" if y16 else "—"
        print(f"  {cap:>10.3f}{s_[3]:>8}{s_[2]:>9.1f}%{r[2]:>11.1f}%"
              f"{s_[2]-r[2]:>+9.1f}pp{f15:>9}{f16:>9}")


if __name__ == "__main__":
    main()
