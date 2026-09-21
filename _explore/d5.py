"""诊断 5：绝对刻度模型。每因子按本板块自身分布的 p10→0 / p90→100 映射，再加权合成。
检验：合成分的极值日是否命中真实拐点（每板块 2 底 + 1 顶）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors

big = data.build_stock_indicators()
raws = {}
for b in C.BOARD_ORDER:
    r = factors.build_board_raw(b, big)
    raws[b] = r[r.index >= pd.Timestamp(C.BACKTEST_START)]

FACTORS = ["b20", "b60", "r5", "nh", "lim", "amt_pct", "rsi", "bias", "ret20", "vol"]
INV = {"vol": True}

def make_panel(raw, factors_list, weights, lo_q=10, hi_q=90):
    s = pd.DataFrame(index=raw.index)
    for k in factors_list:
        v = raw[k].astype(float)
        lo = np.nanpercentile(v.values, lo_q)
        hi = np.nanpercentile(v.values, hi_q)
        m = 100.0 * (v - lo) / (hi - lo)
        if INV.get(k): m = 100.0 - m
        s[k] = m
    tw = sum(weights[k] for k in factors_list)
    return sum(s[k] * weights[k] for k in factors_list) / tw, s

def report(title, factors_list, weights, lo_q=10, hi_q=90, thr=0.0):
    print("\n" + "=" * 108)
    print(f"{title}   (p{lo_q}→0, p{hi_q}→100)")
    print("=" * 108)
    tot_b = tot_bok = tot_t = tot_tok = 0
    for b in C.BOARD_ORDER:
        raw = raws[b]
        comp, _ = make_panel(raw, factors_list, weights, lo_q, hi_q)
        comp = comp.dropna()
        bot_d = raw["close"].idxmin(); top_d = raw["close"].idxmax()
        low_ev = comp[comp < thr]; high_ev = comp[comp > (100 - thr)]
        # 合并连续段
        def pick(mask, how):
            if not mask.any(): return []
            grp = (mask != mask.shift()).cumsum()
            return [ (seg.idxmin() if how=='min' else seg.idxmax())
                     for _, seg in comp[mask].groupby(grp[mask]) ]
        bl = sorted(set(pick(comp < thr, 'min')))
        tl = sorted(set(pick(comp > 100 - thr, 'max')))
        # 正确性：底部事件未来20日收益>0；顶部事件未来20日收益<0
        close = raw["close"]; pos = {d: i for i, d in enumerate(close.index)}
        def ok(ds, sign):
            n = o = 0
            for d in ds:
                i = pos.get(d)
                if i is None or i + 20 >= len(close): continue
                r = close.iloc[i+20] / close.iloc[i] - 1
                n += 1; o += int(r * sign > 0)
            return o, n
        bo, bn = ok(bl, 1); to, tn = ok(tl, -1)
        tot_b += bn; tot_bok += bo; tot_t += tn; tot_tok += to
        print(f"\n{C.BOARDS[b]['name']:>8}  真底 {bot_d.date()}  真顶 {top_d.date()}")
        print(f"   分值最低3天: " + "  ".join(f"{d.date()}({v:.1f})" for d, v in comp.nsmallest(3).items()))
        print(f"   分值最高3天: " + "  ".join(f"{d.date()}({v:.1f})" for d, v in comp.nlargest(3).items()))
        print(f"   真底日分值={comp.asof(bot_d):.1f}  真顶日分值={comp.asof(top_d):.1f}")
        print(f"   溢出<0: {len(bl)}次 {[str(d.date()) for d in bl]}  正确 {bo}/{bn}")
        print(f"   溢出>100:{len(tl)}次 {[str(d.date()) for d in tl]}  正确 {to}/{tn}")
    print(f"\n  >>> 合计  底部 {tot_bok}/{tot_b}   顶部 {tot_tok}/{tot_t}")
    return tot_bok, tot_b, tot_tok, tot_t

W = {"b20": .16, "b60": .14, "r5": .10, "nh": .12, "lim": .08,
     "amt_pct": .07, "rsi": .12, "bias": .11, "ret20": .10, "vol": .00}

report("方案A：全因子（含成交额，vol权重0）", FACTORS, W)
report("方案B：剔除 amt_pct 与 vol", ["b20","b60","r5","nh","lim","rsi","bias","ret20"],
       {"b20": .18, "b60": .15, "r5": .11, "nh": .13, "lim": .09, "rsi": .14, "bias": .12, "ret20": .08})
report("方案C：方案B + p15/p85 收窄刻度", ["b20","b60","r5","nh","lim","rsi","bias","ret20"],
       {"b20": .18, "b60": .15, "r5": .11, "nh": .13, "lim": .09, "rsi": .14, "bias": .12, "ret20": .08},
       lo_q=15, hi_q=85)
report("方案D：方案B + p20/p80 更窄", ["b20","b60","r5","nh","lim","rsi","bias","ret20"],
       {"b20": .18, "b60": .15, "r5": .11, "nh": .13, "lim": .09, "rsi": .14, "bias": .12, "ret20": .08},
       lo_q=20, hi_q=80)
