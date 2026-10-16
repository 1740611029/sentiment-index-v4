"""s39 —— 仓位建议映射校准：「当日状态 → T+H 命中率」分桶统计

目的：为板块卡「建议持有几层仓」（senti/position.py）提供胜率证据，
     输出冻结映射表。只读面板 parquet，不联网、不 refresh（铁律八）。

桶定义（全部 ≤t 可知，无前视；状态构造函数直接 import 自 senti/position.py，
保证校准口径 = 生产口径）：
  小机会（T+7，ok = 涨且期间回撤 ≥ -3%）：
    evS/A/B/C      并集事件当日，按 ures≥5/4/2 分级（信号级 + cluster_events 事件级）
    hold_d1_3 / hold_d4_7   事件后第 1~3 / 4~7 个交易日；按源事件等级再分 SA / BC 两组
    cold           无近期事件、但任一模型分值 ≤ 其触发线（「已跌破」未确认）
    none           其余
  大机会（T+20 ok + T+60 win）：
    evA/B/C        底部事件当日，按 grade（capit=破净≥60 且 resonant=共振≥2）
    hold_d1_10 / hold_d11_20；按源事件等级再分 A / B 两组
    panic          score ≤ 0 且无近期事件
    edge           0 < score ≤ 20 且无近期事件
    none           其余

口径纪律：
  - ev 桶：分档用**信号级**点估计（事件日是单日，无自相关），
    显著性用 **cluster_events 事件级** Wilson 下界（铁律五）。
  - hold/cold/none 等连续状态桶：分档用**日级**点估计
    （用户每天看卡片，「状态总体中的随机一天」才是正确估计总体）；
    显著性用**连续段级**（每段取首日，消除自相关）的 Wilson 下界。
  - 分档规则：点估计落档（≥85→8 / 75~85→7 / 65~75→5 / 55~65→3 / 其余→0），
    显著性闸门不过则降一档；子桶 n<8 → 退父桶，父桶 n<8 → 0；封顶 8。
  - senti A/C 级样本稀少时允许引用 DELIVERY.md §5/§6 已登记的长历史证据人工定档
    （脚本只出数据，引用与理由写在 §21）。

A. 小机会桶（信号级 + 事件级 + 连续段级 + 源事件等级分组）
B. 大机会桶（ok20 + win60 + 连续段级 + 源事件等级分组）
C. 分档建议（打印可直接粘贴进 position.py 的 dict 字面量）
D. parity 自检（position.scan_fires vs 各 events() / union_events）
E. 关键桶逐板拆分
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C, position, store, swing, swing2, swing3  # noqa: E402

H7, TOL7 = swing.H, swing.TOL            # 7, 0.03（swing2/swing3 同）
H20, H60 = store.H, store.H60            # 20, 60
LADDER = [0, 3, 5, 7, 8]


def fwd(close: np.ndarray, i: int, h: int, tol: float):
    """T+h 前瞻标签：(ok, ret, mdd)；未来不足 h 天返回 None。"""
    if i + h > len(close) - 1:
        return None
    seg = close[i + 1: i + h + 1]
    ret = float(seg[-1] / close[i] - 1)
    mdd = float(seg.min() / close[i] - 1)
    return (ret > 0 and mdd >= -tol), ret, mdd


def stat(oks: list) -> dict:
    n = len(oks)
    k = sum(oks)
    return {"n": n, "k": k,
            "rate": (100.0 * k / n if n else None),
            "lb": (store._wilson(k, n) if n else None)}


def band(rate: float | None) -> int:
    if rate is None:
        return 0
    if rate >= 85:
        return 8
    if rate >= 75:
        return 7
    if rate >= 65:
        return 5
    if rate >= 55:
        return 3
    return 0


def decide(day_oks: list, gate_oks: list, base_rate: float):
    """分档：点估计（day_oks）落档 + 显著性闸门（gate_oks 的 Wilson 下界 > 基线，
    否则降一档）；闸门样本 n<8 返回 None（退父桶）。封顶 8。"""
    if len(gate_oks) < 8 or not day_oks:
        return None
    r = band(100.0 * sum(day_oks) / len(day_oks))
    if r > 0 and store._wilson(sum(gate_oks), len(gate_oks)) <= base_rate:
        r = LADDER[max(0, LADDER.index(r) - 1)]
    return r


def fmt(st: dict, base: float | None = None) -> str:
    if not st["n"]:
        return "n=0"
    s = f"n={st['n']:<4d} 命中 {st['rate']:5.1f}%  (Wilson下界 {st['lb']:5.1f}%)"
    if base is not None:
        s += f"  vs 基线 {st['rate'] - base:+5.1f}pp"
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

    # ================= A. 小机会（T+7） =================
    print("\n== A. 小机会（T+7，ok=涨且回撤≥-3%）==")
    sw_day: dict[str, list] = {}              # 桶 -> 日级 ok
    sw_streak: dict[str, list] = {}           # 桶 -> 连续段级 ok（每段取首日）
    base7: list = []
    for b in C.BOARD_ORDER:
        p = sp[b]
        c = p["close"].to_numpy(dtype=float)
        s1 = p["swing"].to_numpy(dtype=float)
        s2m = dict(zip([str(d.date()) for d in s2p[b].index],
                       s2p[b]["swing2"].to_numpy(dtype=float)))
        s3m = dict(zip([str(d.date()) for d in s3p[b].index],
                       s3p[b]["swing3"].to_numpy(dtype=float)))
        ev_by_date = {e["date"]: position.swing_grade(e.get("ures")) for e in uev[b]}
        last_ev = None
        last_g = None
        prev_bk = None
        for i in range(len(p.index)):
            d = str(p.index[i].date())
            fk = fwd(c, i, H7, TOL7)
            if fk is None:
                if d in ev_by_date:
                    last_ev, last_g = i, ev_by_date[d]
                prev_bk = None
                continue
            base7.append(fk[0])
            if d in ev_by_date:
                last_ev, last_g = i, ev_by_date[d]
                bk = "ev" + last_g
            else:
                st = position.swing_st(None if last_ev is None else i - last_ev, None)
                if st["state"] == "hold":
                    grp = "SA" if last_g in ("S", "A") else "BC"
                    bk = f"hold_{st['seg']}_{grp}"
                elif ((not np.isnan(s1[i]) and s1[i] <= swing.THR)
                      or s2m.get(d, 99) <= swing2.THR
                      or s3m.get(d, 99) <= swing3.THR):
                    bk = "cold"
                else:
                    bk = "none"
            sw_day.setdefault(bk, []).append(fk[0])
            if bk != prev_bk:                  # 连续段级：每段只取首日
                sw_streak.setdefault(bk, []).append(fk[0])
            prev_bk = bk

    base7r = 100.0 * sum(base7) / len(base7)
    print(f"随机基线（全部可判定日 T+7 ok）: {base7r:.1f}%  (n={len(base7)})")
    firsts = store.cluster_events({b: uev[b] for b in C.BOARD_ORDER})
    ev_lvl: dict[str, list] = {}
    for e in firsts:
        if e.get("ok") is not None:
            ev_lvl.setdefault("ev" + position.swing_grade(e.get("ures")), []).append(e["ok"])
    n_sig = sum(len(sw_day.get("ev" + g, [])) for g in "SABC")
    print(f"并集信号总数 {n_sig} / 事件级第一枪 {sum(len(v) for v in ev_lvl.values())}")
    for bk in ("evS", "evA", "evB", "evC"):
        print(f"  {bk:<16s} 信号级 {fmt(stat(sw_day.get(bk, [])), base7r)}")
        print(f"  {'':<16s} 事件级 {fmt(stat(ev_lvl.get(bk, [])), base7r)}")
    for bk in ("hold_d1_3_SA", "hold_d1_3_BC", "hold_d4_7_SA", "hold_d4_7_BC"):
        print(f"  {bk:<16s} 日级   {fmt(stat(sw_day.get(bk, [])), base7r)}")
        print(f"  {'':<16s} 段级   {fmt(stat(sw_streak.get(bk, [])), base7r)}")
    for bk in ("cold", "none"):
        print(f"  {bk:<16s} 日级   {fmt(stat(sw_day.get(bk, [])), base7r)}"
              f"   段级 n={len(sw_streak.get(bk, []))}")

    # ================= B. 大机会（T+20 / T+60） =================
    print("\n== B. 大机会（T+20 ok / T+60 win）==")
    se_day: dict[str, list] = {}
    se_streak: dict[str, list] = {}
    se_win: dict[str, list] = {}
    base20: list = []
    base60: list = []
    for b in C.BOARD_ORDER:
        p = panels[b]
        c = p["close"].to_numpy(dtype=float)
        sc = p["score"].to_numpy(dtype=float)
        fires = position.scan_fires(sc, "senti")
        fire_at = {i: (position._senti_grade_at(p, i, reso, bna) or "C") for i in fires}
        last_f = None
        last_g = None
        prev_bk = None
        for i in range(len(c)):
            fk20 = fwd(c, i, H20, store.TOL)
            r60 = (c[i + H60] / c[i] - 1 > 0) if i + H60 <= len(c) - 1 else None
            if fk20 is not None:
                base20.append(fk20[0])
            if r60 is not None:
                base60.append(r60)
            if i in fire_at:
                last_f, last_g = i, fire_at[i]
                bk = "ev" + last_g
            else:
                st = position.senti_st(None if last_f is None else i - last_f, None)
                if st["state"] == "hold":
                    bk = f"hold_{st['seg']}_{last_g}"
                elif not np.isnan(sc[i]) and sc[i] <= 0:
                    bk = "panic"
                elif not np.isnan(sc[i]) and sc[i] <= 20:
                    bk = "edge"
                else:
                    bk = "none"
            if fk20 is not None:
                se_day.setdefault(bk, []).append(fk20[0])
                if bk != prev_bk:
                    se_streak.setdefault(bk, []).append(fk20[0])
            if r60 is not None:
                se_win.setdefault(bk, []).append(r60)
            prev_bk = bk if fk20 is not None else None

    base20r = 100.0 * sum(base20) / len(base20)
    base60r = 100.0 * sum(base60) / len(base60)
    print(f"随机基线: T+20 ok {base20r:.1f}% (n={len(base20)})   "
          f"T+60 win {base60r:.1f}% (n={len(base60)})")
    for bk in ("evA", "evB", "evC",
               "hold_d1_10_A", "hold_d1_10_B", "hold_d11_20_A", "hold_d11_20_B",
               "panic", "edge", "none"):
        print(f"  {bk:<16s} T+20 {fmt(stat(se_day.get(bk, [])), base20r)}"
              f"   段级 n={len(se_streak.get(bk, []))}")
        if se_win.get(bk):
            stw = stat(se_win[bk])
            print(f"  {'':<16s} T+60 win {stw['rate']:5.1f}% (n={stw['n']}, 基线 {base60r:.1f}%)")

    # ================= C. 分档建议 =================
    print("\n== C. 分档建议（点估计落档 + 事件级/段级 Wilson 闸门；n<8 退父桶；封顶 8）==")

    def grab(table: dict, key: str) -> list:
        return table.get(key, [])

    def grab_pooled(table: dict, prefix: str, groups=("SA", "BC")) -> list:
        out: list = []
        for g in groups:
            out += table.get(prefix + g, [])
        return out

    sw_map = {}
    ev_any_sig = [x for g in "SABC" for x in sw_day.get("ev" + g, [])]
    ev_any_lvl = [x for g in "SABC" for x in ev_lvl.get("ev" + g, [])]
    for g in "SABC":
        r = decide(grab(sw_day, "ev" + g), grab(ev_lvl, "ev" + g), base7r)
        if r is None:
            r = decide(ev_any_sig, ev_any_lvl, base7r)
        sw_map[("event", g)] = r if r is not None else 0
    for seg in ("d1_3", "d4_7"):
        r = decide(grab_pooled(sw_day, f"hold_{seg}_"), grab_pooled(sw_streak, f"hold_{seg}_"),
                   base7r)
        if r is None:
            r = decide(grab_pooled(sw_day, "hold_"), grab_pooled(sw_streak, "hold_"), base7r)
        sw_map[("hold", seg)] = r if r is not None else 0
    print("SWING_MAP 建议（hold 为 SA/BC 合并口径；分组数据见 A 段，用于判断要不要按源等级收缩）:")
    for k, v in sw_map.items():
        print(f"    {k}: {v},")

    se_map = {}
    se_ev_any = [x for g in "ABC" for x in se_day.get("ev" + g, [])]
    se_ev_any_st = [x for g in "ABC" for x in se_streak.get("ev" + g, [])]
    for g in "ABC":
        r = decide(grab(se_day, "ev" + g), grab(se_streak, "ev" + g), base20r)
        if r is None:
            r = decide(se_ev_any, se_ev_any_st, base20r)
        se_map[("event", g)] = r if r is not None else 0
    for seg in ("d1_10", "d11_20"):
        r = decide(grab_pooled(se_day, f"hold_{seg}_", ("A", "B", "C")),
                   grab_pooled(se_streak, f"hold_{seg}_", ("A", "B", "C")), base20r)
        if r is None:
            r = decide(grab_pooled(se_day, "hold_", ("A", "B", "C")),
                       grab_pooled(se_streak, "hold_", ("A", "B", "C")), base20r)
        se_map[("hold", seg)] = r if r is not None else 0
    print("SENTI_MAP 建议（ev 桶 n 稀少，允许引用 DELIVERY.md 已登记的长历史证据人工定档）:")
    for k, v in se_map.items():
        print(f"    {k}: {v},")

    # ================= D. parity 自检 =================
    print("\n== D. parity 自检（scan_fires vs 各 events()）==")
    bad = 0
    for b in C.BOARD_ORDER:
        for kind, pan, mod, col, sr in (("swing", sp, swing, "swing", sreso),
                                        ("swing2", s2p, swing2, "swing2", s2reso),
                                        ("swing3", s3p, swing3, "swing3", s3reso)):
            f = position.scan_fires(pan[b][col].to_numpy(dtype=float), kind)
            got = {str(pan[b].index[i].date()) for i in f}
            want = {e["date"] for e in mod.events(pan[b], sr)}
            if got != want:
                bad += 1
                print(f"  X {b} {kind}: 多 {sorted(got - want)} 少 {sorted(want - got)}")
        p = panels[b]
        f = position.scan_fires(p["score"].to_numpy(dtype=float), "senti")
        got = {str(p.index[i].date()) for i in f if i < len(p) - store.H}   # events() 截断末尾 H 天
        want = {e["date"] for e in store.events(p, reso, bna)}
        if got != want:
            bad += 1
            print(f"  X {b} senti: 多 {sorted(got - want)} 少 {sorted(want - got)}")
        u = {e["date"] for e in uev[b]}
        fu = set()
        for kind, pan, col in (("swing", sp, "swing"), ("swing2", s2p, "swing2"),
                               ("swing3", s3p, "swing3")):
            fu |= {str(pan[b].index[i].date())
                   for i in position.scan_fires(pan[b][col].to_numpy(dtype=float), kind)}
        if fu != u:
            bad += 1
            print(f"  X {b} union: 多 {sorted(fu - u)} 少 {sorted(u - fu)}")
    print("parity:", "OK 全部一致" if bad == 0 else f"X {bad} 处不一致")

    # ================= E. 关键桶逐板拆分 =================
    print("\n== E. 关键桶逐板拆分（事件日 T+H ok）==")
    for label, pan, kind, h, tol in (("小机会·事件日", sp, None, H7, TOL7),
                                     ("大机会·事件日", panels, "senti", H20, store.TOL)):
        cells = []
        for b in C.BOARD_ORDER:
            p = pan[b]
            c = p["close"].to_numpy(dtype=float)
            if kind == "senti":
                idxs = position.scan_fires(p["score"].to_numpy(dtype=float), "senti")
            else:
                idxs = [i for i, d in enumerate(p.index) if str(d.date()) in
                        {e["date"] for e in uev[b]}]
            oks = [fk[0] for i in idxs if (fk := fwd(c, i, h, tol)) is not None]
            cells.append(f"{C.BOARDS[b]['name']} {sum(oks)}/{len(oks)}")
        print(f"  {label}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
