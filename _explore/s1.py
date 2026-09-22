"""s1 —— 验证 senti/swing.py：命中率、基线、目标日期覆盖。"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C, swing

if __name__ == "__main__":
    t0 = time.time()
    panels = swing.build_and_cache(force=True)
    print(f"build {time.time()-t0:.0f}s\n")
    s = swing.summary(panels)
    t = s["total"]
    print(f"阈值 ≤{s['thr']}　冷却 {s['cool']}　判定 T+{s['hold']} 期末>0 且回撤≥−{s['tol']}%")
    print(f"{'板块':<10}{'信号':>6}{'命中':>7}{'命中率':>9}{'未被套':>9}{'待验':>6}{'基线':>9}")
    for b in C.BOARD_ORDER:
        v = s["per"][b]
        print(f"{C.BOARDS[b]['name']:<10}{v['n']:>6}{v['ok']:>7}"
              f"{(str(v['rate'])+'%') if v['rate'] is not None else '—':>9}"
              f"{v['nt']:>9}{v['pend']:>6}{v['base']:>8}%")
    print(f"\n合计  命中 {t['ok']}/{t['n']} = {t['rate']}%　"
          f"未被套 {t['nt']}/{t['n']}　待验证 {t['pend']}　基线 {t['base']}%")
    print(f"  急跌档(rush) {t['rush_ok']}/{t['rush_n']}"
          f"　共振档(reso≥{s['reso_min']}) {t['res_ok']}/{t['res_n']}"
          f"　两者同时 {t['ab_ok']}/{t['ab_n']}")

    # 目标日期
    print("\n科创板近段 SWING 分值:")
    p = panels["STAR"]
    print(p["swing"].loc["2026-07-24":].round(1).to_string())
    ev = swing.events(p, swing.resonance(panels))
    print("\n科创板信号（近 3 个月）:")
    for e in ev:
        if e["date"] >= "2026-07-01":
            print(f"  {e['date']}  分值 {e['score']:>5}  急跌={e['rush']}"
                  f"  共振={e['reso']}  T+7 {e['ret']}%  {'✔' if e['ok'] else ('待验证' if e['ok'] is None else '✘')}")
