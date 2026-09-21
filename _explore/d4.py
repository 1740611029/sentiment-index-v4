"""诊断 4：改用绝对刻度。查看各原始因子在真实拐点日的读数 vs 全窗口分布。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, model, factors

big = data.build_stock_indicators()
raws = {b: factors.build_board_raw(b, big) for b in C.BOARD_ORDER}

FACTORS = ["b20", "b60", "r5", "nh", "lim", "amt_pct", "rsi", "bias", "ret20", "vol"]

for b in C.BOARD_ORDER:
    p = raws[b]
    p = p[p.index >= pd.Timestamp(C.BACKTEST_START)]
    bot_d = p["close"].idxmin(); top_d = p["close"].idxmax()
    print("="*112)
    print(f"{C.BOARDS[b]['name']}   真底 {bot_d.date()}   真顶 {top_d.date()}")
    print("="*112)
    print(f"{'因子':>8} {'真底读数':>10} {'真顶读数':>10} {'窗口min':>9} {'p1':>8} {'p50':>8} {'p99':>8} {'窗口max':>9}  {'底分位':>7} {'顶分位':>7}")
    for k in FACTORS:
        s = p[k].dropna()
        if s.empty: continue
        bv = s.asof(bot_d); tv = s.asof(top_d)
        bp = (s <= bv).mean() * 100; tp = (s <= tv).mean() * 100
        print(f"{k:>8} {bv:>10.4f} {tv:>10.4f} {s.min():>9.4f} {s.quantile(.01):>8.4f} "
              f"{s.median():>8.4f} {s.quantile(.99):>8.4f} {s.max():>9.4f}  {bp:>6.1f}% {tp:>6.1f}%")
    print()
