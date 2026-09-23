"""s19 —— 第十一轮：两个必须落地的结论。

s18 查出两件事
------------
① s17 的「过关者」绝大多数是重复计数。±3 交易日邻域合并后 35 组只剩 15 组，
   分半检验后 12 组，而且其中 9 组的「增益」只是往 916 个信号里加 1~4 个信号
   （58.7% → 58.8%），属于噪声不是模型。
   **唯一有真实量级的是 rsi_slope≤3 + vconv**（全程 58.7→60.3、近5.7年 67.9→68.9、
   近3年 71.1→71.8，且两半都为正）。
② 交付口径的「189 次」里 61% 是同一波下跌的相邻日重复。这跟 SENTI-1
   「17 次信号只对应 5 次事件」是同一类问题，页面口径必须补上**事件级**统计。

本脚本
-----
A. 邻域合并窗口敏感性（±2/3/5/8/10 交易日）：rsi_slope+vconv 是真新事件还是慢 3 天的重复？
B. 交付口径事件级统计：同板块 ±4 自然日聚成一件事，只算每件事的**第一枪**
C. rsi_slope≤3+vconv 在 6 板块页面口径的完整评估（够不够当 SWING-4）
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
import s12                                       # noqa: F401  提供 Ev6 / mk_vconv
import s17
import s18
from s8 import wilson, W_ALL, W_57, W_3, BOARDS  # noqa: E402

WINS = (("全程", W_ALL), ("近5.7年", W_57), ("近3年", W_3))


def rate(uni, w0, w1):
    n = k = 0
    for b, d in uni.items():
        for dt, (ok, _) in d.items():
            if w0 <= pd.Timestamp(dt) <= w1:
                n += 1
                k += int(ok)
    return n, k, (100 * k / n if n else float("nan"))


def show(tag, uni):
    cells = []
    for lab, (w0, w1) in WINS:
        n, k, r = rate(uni, w0, w1)
        cells.append(f"{r:5.1f}%({n:>3})")
    print(f"  {tag:<30}{cells[0]:>14}{cells[1]:>14}{cells[2]:>14}")


# ---------------------------------------------------------------- 事件聚类
def cluster(events, gap_days=4):
    """同一板块内，相隔 ≤gap_days 自然日的信号聚成一件事，返回每件事的第一枪。

    第一枪才是实盘能买到的那一枪；后面几枪是「已经在场内」的重复记录。
    events 为 [(board, event_dict), ...]
    """
    by_b = {}
    for b, e in events:
        by_b.setdefault(b, []).append(e)
    firsts = []
    for b, ev in by_b.items():
        ev = sorted(ev, key=lambda x: x["date"])
        last = None
        for e in ev:
            d = pd.Timestamp(e["date"])
            if last is None or (d - last).days > gap_days:
                firsts.append(e)
                last = d
    return firsts


def cstat(evs):
    done = [e for e in evs if e.get("ok") is not None]
    if not done:
        return 0, 0, float("nan"), float("nan")
    k = sum(1 for e in done if e["ok"])
    return len(done), k, 100 * k / len(done), wilson(k, len(done))


def main():
    E = s8.Evaluator()
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])

    print("=" * 114)
    print("A. 邻域合并窗口敏感性：rsi_slope≤3+vconv 新增的信号离现模型有多远")
    print("=" * 114)
    print(f"  {'合并窗口':<30}{'全程':>14}{'近5.7年':>14}{'近3年':>14}")
    show("【基准】并集(3 模型)", inc)
    S_rs = E.score("rsi_slope")
    cand = s17.sig_dates(E, S_rs, 3, 4, s12.mk_vconv())
    nraw = sum(len(v) for v in cand.values())
    print(f"  rsi_slope≤3+vconv 原始信号 {nraw} 个")
    for m in (2, 3, 5, 8, 10):
        add = s18.dedup_near(E, inc, cand, merge=m)
        nadd = sum(len(v) for v in add.values())
        show(f"  ±{m} 日合并（新增 {nadd}）", s18.merge_all(inc, add))

    print("\n" + "=" * 114)
    print("B. 交付口径（6 板块 · 展示窗口）事件级统计：只算每件事的第一枪")
    print("=" * 114)
    from senti import config as C
    from senti import store, swing, swing2, swing3
    sp, s2p, s3p = store.swing_panels(), store.swing2_panels(), store.swing3_panels()
    sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
    D = {}
    for b in C.BOARD_ORDER:
        D[b] = {
            "SWING": swing.events(sp[b], sr),
            "SWING-2": swing2.events(s2p[b], s2r),
            "SWING-3": swing3.events(s3p[b], s3r),
        }
        u = store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r)
        D[b]["并集(三)"] = u
        two = [e for e in u if e["src"] in ("swing", "macd", "both")]
        D[b]["并集(两)"] = two
    # 基线（每板块随机一天）
    bases = [swing2.baseline(s2p[b]) for b in C.BOARD_ORDER]
    base = float(np.mean(bases))
    print(f"  随机基线（同口径）{base:.1f}%\n")
    print(f"  {'口径':<14}{'信号数':>8}{'事件数':>8}{'重复率':>8}"
          f"{'信号命中':>10}{'事件命中':>10}{'事件下界':>10}")
    print("  " + "-" * 84)
    for key in ("SWING", "SWING-2", "SWING-3", "并集(两)", "并集(三)"):
        evs = [(b, e) for b in C.BOARD_ORDER for e in D[b][key]]
        sig_n, sig_k, sig_r, _ = cstat([e for _, e in evs])
        ev = cluster(evs)
        ev_n, ev_k, ev_r, ev_wl = cstat(ev)
        dup = 100 * (1 - ev_n / sig_n) if sig_n else 0
        print(f"  {key:<14}{sig_n:>8}{ev_n:>8}{dup:>7.0f}%"
              f"{sig_r:>9.1f}%{ev_r:>9.1f}%{ev_wl:>9.1f}%")
    allu = [(b, e) for b in C.BOARD_ORDER for e in D[b]["并集(三)"]]
    ev3 = cluster(allu)
    print(f"\n  → 「{len(allu)} 次」是信号次数；按事件算只有 {len(ev3)} 件事"
          f"（命中 {cstat(ev3)[1]}/{cstat(ev3)[0]} = {cstat(ev3)[2]:.1f}%）。")

    print("\n" + "=" * 114)
    print("C. rsi_slope≤3+vconv 在 6 板块页面口径（够不够当第 4 个模型）")
    print("=" * 114)
    E6 = s12.Ev6()
    S6 = E6.score("rsi_slope")
    print(f"  6 板块基线：近5.7年 {E6.base(*W_57):.1f}%　近3年 {E6.base(*W_3):.1f}%")
    print(f"\n  {'口径':<26}{'信号数':>8}{'事件数':>8}{'信号命中':>10}{'事件命中':>10}{'下界':>9}")
    print("  " + "-" * 74)

    def conv(ev):
        return [(b, {"date": d, "ok": ok}) for b, d, ok, _ in ev]

    def line(tag, ev):
        tup = [(b, {"date": d, "ok": ok}) for b, d, ok, _ in ev]
        sn, sk, sr_, _ = cstat([e for _, e in tup])
        cl = cluster(tup)
        cn, ck, cr, cwl = cstat(cl)
        print(f"  {tag:<26}{sn:>8}{cn:>8}{sr_:>9.1f}%{cr:>9.1f}%{cwl:>8.1f}%")

    for thr in (1, 2, 3, 5, 8):
        line(f"rsi_slope≤{thr:g}+vconv(近3年)",
             E6.run(S6, thr, 4, *W_3, cond=s12.mk_vconv()))
    line("rsi_slope≤3+vconv(近5.7年)",
         E6.run(S6, 3, 4, *W_57, cond=s12.mk_vconv()))
    print("\n  对照：现役三模型在 6 板块页面口径（近3年）")
    for key in ("SWING", "SWING-2", "SWING-3", "并集(三)"):
        evs = [(b, e) for b in C.BOARD_ORDER for e in D[b][key]]
        sn, sk, sr_, _ = cstat([e for _, e in evs])
        cl = cluster(evs)
        cn, ck, cr, cwl = cstat(cl)
        print(f"  {key:<26}{sn:>8}{cn:>8}{sr_:>9.1f}%{cr:>9.1f}%{cwl:>8.1f}%")


if __name__ == "__main__":
    main()
