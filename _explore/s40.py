"""s40 —— 仓位规则第二轮校准：「当日信号 + 指标数值」对胜率 / 盈亏比 / 期望收益的区分度

背景：s39 按「事件等级 + 持有天数」分桶冻结了 position.py 映射，用户反馈要的是
「根据当前的信号和信号指标数值制定胜率高、盈亏比好的仓位」——即层数应由
当日可观测的连续数值（几个模型同时跌破、跌破多深、SENTI-1 分值读数）驱动，
而不是只由离散状态驱动。本脚本检验这些数值维度**有没有单调区分度**，
有 → 作为分档依据；没有 → 如实回报「数值维度无增量，维持状态桶」。

口径纪律：
  - 全部特征 ≤t 可知（当日收盘分值），无前视；只读面板，不 refresh（铁律八）。
  - 每桶同时报：n / ok 胜率 / Wilson 下界 / 均盈 / 均亏 / 盈亏比 PF / 期望 ret / 中位 ret。
    （ok = T+H 涨且期间回撤 ≥ -3%；盈亏比 PF = 均盈 / 均亏，用纯 T+H 收益算。）
  - 显著性闸门沿用 s39：事件日单日无自相关用信号级；连续状态用连续段级（每段首日）。
  - 单调性才配做分档依据：只看单峰/单调，不看孤立的最高桶（防参数刀刃，AGENTS §4-17）。

A. 小机会·当日模型状态维度（全部可判定日，T+7）
   n_mod   = 当日三模型中几个 ≤ 各自触发线（0/1/2/3）
   rel     = 最深相对跌破深度 max_m max(0, THR_m - v_m)/THR_m（各模型刻度不可横比，
             深度只在模型内部归一后取 max，不跨模型相加）
   交叉：n_mod × 是否近 7 日内有并集事件（事件/持有 vs cold）
B. 小机会·事件日细分（事件日当天的 n_mod / rel 是否比等级更有区分度）
C. 大机会·SENTI-1 分值读数桶（全部可判定日，T+20 ok / T+60 win）
   score 桶：≤-20 / (-20,0] / (0,10] / (10,20] / (20,40] / >40
   交叉：是否近 20 日内有底部事件（事件/持有 vs 无事件）
D. 汇总：哪些维度单调、哪些平/乱 → 建议规则形态
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C, position, store, swing, swing2, swing3  # noqa: E402

H7, TOL7 = swing.H, swing.TOL
H20, H60 = store.H, store.H60


def fwd(close, i, h):
    """T+h 前瞻标签：(ret, mdd)；未来不足 h 天返回 None。"""
    if i + h > len(close) - 1:
        return None
    seg = close[i + 1: i + h + 1]
    return float(seg[-1] / close[i] - 1), float(seg.min() / close[i] - 1)


def pl(rows: list[tuple[bool, float]]) -> dict:
    """rows = [(ok, ret)] → n/胜率/Wilson/均盈/均亏/PF/期望/中位。"""
    n = len(rows)
    if not n:
        return {"n": 0}
    oks = [r[0] for r in rows]
    rets = np.array([r[1] for r in rows])
    wins = rets[rets > 0]
    loss = rets[rets <= 0]
    aw = float(wins.mean()) if len(wins) else 0.0
    al = float(-loss.mean()) if len(loss) else 0.0
    return {"n": n, "k": sum(oks),
            "rate": 100.0 * sum(oks) / n,
            "lb": store._wilson(sum(oks), n),
            "avg_w": 100 * aw, "avg_l": 100 * al,
            "pf": (aw / al if al > 0 else float("inf")),
            "exp": 100 * float(rets.mean()),
            "med": 100 * float(np.median(rets))}


def fmt(st: dict, base_rate: float | None = None) -> str:
    if not st["n"]:
        return "n=0"
    s = (f"n={st['n']:<4d} ok {st['rate']:5.1f}% (LB {st['lb']:5.1f}) "
         f"| 均盈 {st['avg_w']:5.2f} 均亏 {st['avg_l']:5.2f} PF {st['pf']:4.2f} "
         f"| 期望 {st['exp']:+5.2f}% 中位 {st['med']:+5.2f}%")
    if base_rate is not None:
        s += f" | vs 基线 {st['rate'] - base_rate:+5.1f}pp"
    return s


def main():
    panels = store.load()
    sp, s2p, s3p = swing.load(), swing2.load(), swing3.load()
    reso, bna = store.resonance(panels), store.bna_series()
    sreso, s2reso, s3reso = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
    uev = {b: store.union_events(b, sp[b], s2p[b], s3p[b], sreso, s2reso, s3reso)
           for b in C.BOARD_ORDER}
    ures = store.union_resonance(uev)
    for b in C.BOARD_ORDER:
        for e in uev[b]:
            e["ures"] = ures.get(e["date"], 1)

    d0 = min(p.index[0] for p in panels.values())
    d1 = max(p.index[-1] for p in panels.values())
    print(f"窗口: {d0.date()} ~ {d1.date()}（生产面板，近 3 年）")

    # ============ A/B. 小机会：当日 n_mod × rel × 是否有近期事件 ============
    print("\n== A. 小机会·当日模型状态（全部可判定日，T+7；ok=涨且回撤≥-3%）==")
    bkt: dict[str, list] = {}        # 日级
    streak: dict[str, list] = {}     # 连续段级（每段首日）
    evday: dict[str, list] = {}      # B 段：仅事件日
    holdx: dict[str, list] = {}      # B 段：SA 源持有期 × 当日 n_mod
    base: list = []
    REL_BANDS = [(0.0, 0.25, "浅破0~25%"), (0.25, 0.5, "中破25~50%"), (0.5, 9.9, "深破>50%")]
    for b in C.BOARD_ORDER:
        p = sp[b]
        c = p["close"].to_numpy(dtype=float)
        v1 = p["swing"].to_numpy(dtype=float)
        v2 = dict(zip([str(d.date()) for d in s2p[b].index],
                      s2p[b]["swing2"].to_numpy(dtype=float)))
        v3 = dict(zip([str(d.date()) for d in s3p[b].index],
                      s3p[b]["swing3"].to_numpy(dtype=float)))
        ev_idx = {}
        for e in uev[b]:
            # 事件日期 → 面板下标
            matches = [i for i, d in enumerate(p.index) if str(d.date()) == e["date"]]
            if matches:
                ev_idx[matches[0]] = position.swing_grade(e.get("ures"))
        last_ev = None
        last_g = None
        prev_bk = None
        for i in range(len(p.index)):
            f = fwd(c, i, H7)
            if f is None:
                if i in ev_idx:
                    last_ev = i
                    last_g = ev_idx[i]
                prev_bk = None
                continue
            ret, mdd = f
            ok = ret > 0 and mdd >= -TOL7
            base.append((ok, ret))
            d = str(p.index[i].date())
            x2, x3 = v2.get(d, np.nan), v3.get(d, np.nan)
            n_mod = int((not np.isnan(v1[i])) and v1[i] <= swing.THR) \
                  + int((not np.isnan(x2)) and x2 <= swing2.THR) \
                  + int((not np.isnan(x3)) and x3 <= swing3.THR)
            rel = max(
                (swing.THR - v1[i]) / swing.THR if not np.isnan(v1[i]) else 0.0,
                (swing2.THR - x2) / swing2.THR if not np.isnan(x2) else 0.0,
                (swing3.THR - x3) / swing3.THR if not np.isnan(x3) else 0.0,
                0.0)
            rel_lab = next((lab for lo, hi, lab in REL_BANDS if lo <= rel < hi), "?")
            if i in ev_idx:
                last_ev = i
                last_g = ev_idx[i]
                ctx = "事件日"
                evday.setdefault(f"nmod{n_mod}", []).append((ok, ret))
                evday.setdefault(f"rel_{rel_lab}", []).append((ok, ret))
                evday.setdefault(f"grade_{ev_idx[i]}", []).append((ok, ret))
                evday.setdefault("ALL", []).append((ok, ret))
                gg = "SA" if ev_idx[i] in ("S", "A") else "BC"
                evday.setdefault(f"{gg}|{'深破' if rel >= 0.5 else '浅中破'}", []).append((ok, ret))
            elif last_ev is not None and 1 <= i - last_ev <= H7:
                ctx = "持有期"
                if last_g in ("S", "A"):
                    seg = "d1_3" if i - last_ev <= 3 else "d4_7"
                    holdx.setdefault(f"{seg}|当日nmod{'0' if n_mod == 0 else '≥1'}",
                                     []).append((ok, ret))
            else:
                ctx = "无事件"
            bk = f"{ctx}|nmod{n_mod}"
            bkt.setdefault(bk, []).append((ok, ret))
            if n_mod >= 1:
                bkt.setdefault(f"{ctx}|nmod{n_mod}|{rel_lab}", []).append((ok, ret))
            if bk != prev_bk:
                streak.setdefault(bk, []).append((ok, ret))
            prev_bk = bk

    bs = pl(base)
    print(f"随机基线: {fmt(bs)}")

    def show(title, table, keys, base_rate):
        print(f"\n  -- {title} --")
        for k in keys:
            st = pl(table.get(k, []))
            print(f"  {k:<28s} {fmt(st, base_rate)}")

    show("n_mod 单维（合并事件/持有/无事件）", bkt,
         [f"{ctx}|nmod{m}" for ctx in ("事件日", "持有期", "无事件") for m in (1, 2, 3)],
         bs["rate"])
    show("n_mod × 深度（事件日+持有期合并为「有事件」另行看 B；此处全合并）", bkt,
         [f"{ctx}|nmod{m}|{lab}" for ctx in ("事件日", "持有期", "无事件")
          for m in (1, 2, 3) for lo, hi, lab in REL_BANDS],
         bs["rate"])
    print("\n  段级显著性（连续段首日，供闸门参考）:")
    for k in sorted(streak):
        st = pl(streak[k])
        print(f"  段 {k:<28s} {fmt(st, bs['rate'])}")

    print("\n== B. 小机会·事件日细分（信号级；n_mod / rel / 等级 三把刀对照）==")
    show("事件日 ALL / n_mod / rel / 等级", evday,
         ["ALL", "nmod1", "nmod2", "nmod3",
          "rel_浅破0~25%", "rel_中破25~50%", "rel_深破>50%",
          "grade_S", "grade_A", "grade_B", "grade_C"], bs["rate"])
    show("事件日 等级组 × 深度（深破=最深相对跌破 ≥50%）", evday,
         ["SA|浅中破", "SA|深破", "BC|浅中破", "BC|深破"], bs["rate"])
    show("SA 源持有期 × 当日 n_mod（0=三模型都已回到触发线上）", holdx,
         ["d1_3|当日nmod0", "d1_3|当日nmod≥1", "d4_7|当日nmod0", "d4_7|当日nmod≥1"],
         bs["rate"])

    # ============ C. 大机会：SENTI-1 分值读数桶 ============
    print("\n== C. 大机会·SENTI-1 分值读数（T+20 ok / T+60 win）==")
    SC_BANDS = [(-1e9, -20, "≤-20"), (-20, 0, "(-20,0]"), (0, 10, "(0,10]"),
                (10, 20, "(10,20]"), (20, 40, "(20,40]"), (40, 1e9, ">40")]
    se_bkt: dict[str, list] = {}
    se_win: dict[str, list] = {}
    base20: list = []
    base60: list = []
    for b in C.BOARD_ORDER:
        p = panels[b]
        c = p["close"].to_numpy(dtype=float)
        sc = p["score"].to_numpy(dtype=float)
        fires = set(position.scan_fires(sc, "senti"))
        last_f = None
        for i in range(len(c)):
            f20 = fwd(c, i, H20)
            r60 = (c[i + H60] / c[i] - 1) if i + H60 <= len(c) - 1 else None
            if f20 is not None:
                base20.append((f20[0] > 0 and f20[1] >= -store.TOL, f20[0]))
            if r60 is not None:
                base60.append((r60 > 0, r60))
            if i in fires:
                last_f = i
            ctx = "事件/持有" if (last_f is not None and i - last_f <= H20) else "无事件"
            if np.isnan(sc[i]):
                continue
            lab = next((lab for lo, hi, lab in SC_BANDS if lo < sc[i] <= hi), "?")
            if f20 is not None:
                se_bkt.setdefault(f"{ctx}|{lab}", []).append((f20[0] > 0 and f20[1] >= -store.TOL, f20[0]))
            if r60 is not None:
                se_win.setdefault(f"{ctx}|{lab}", []).append((r60 > 0, r60))

    b20 = pl(base20)
    b60 = pl(base60)
    print(f"随机基线 T+20: {fmt(b20)}")
    print(f"随机基线 T+60: {fmt(b60)}")
    keys = [f"{ctx}|{lab}" for ctx in ("事件/持有", "无事件") for lo, hi, lab in SC_BANDS]
    print("\n  -- T+20 ok（含回撤判定）--")
    for k in keys:
        print(f"  {k:<22s} {fmt(pl(se_bkt.get(k, [])), b20['rate'])}")
    print("\n  -- T+60 纯收益方向 --")
    for k in keys:
        print(f"  {k:<22s} {fmt(pl(se_win.get(k, [])), b60['rate'])}")

    # ============ D. 汇总判读 ============
    print("\n== D. 判读要点（单调性手检）==")
    print("小机会事件日 n_mod 1→2→3 的 ok 率:",
          " / ".join(f"{pl(evday.get(f'nmod{m}', [])).get('rate', 0):.1f}%"
                     f"(n={pl(evday.get(f'nmod{m}', []))['n']})" for m in (1, 2, 3)))
    print("小机会事件日 rel 浅→中→深 的 ok 率:",
          " / ".join(f"{pl(evday.get(f'rel_{lab}', [])).get('rate', 0):.1f}%"
                     f"(n={pl(evday.get(f'rel_{lab}', []))['n']})"
                     for lo, hi, lab in REL_BANDS))
    seq = []
    for lo, hi, lab in SC_BANDS:
        st = pl(se_bkt.get(f"无事件|{lab}", []))
        seq.append(f"{lab}:{st.get('rate', 0):.1f}%(n={st['n']})")
    print("大机会无事件 score 低→高 的 T+20 ok 率:", " / ".join(seq))


if __name__ == "__main__":
    main()
