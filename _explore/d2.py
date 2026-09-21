"""诊断 2：寻找能真正预测「顶部」的连续信号，并验证底部侧。
重点：A股贪婪极值后往往继续涨，必须测试「过热 + 破位确认」路线。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, model, backtest

T = 20
big = data.build_stock_indicators()
panels = model.build_all(big)

def fwd(p, h=T):
    """返回 date -> (ret_h, mdd_h, run_h)"""
    close = p["close"].values; idx = p.index
    pos = {d: i for i, d in enumerate(idx)}
    out = {}
    for d in idx:
        i = pos[d]
        if i + h >= len(close): continue
        c0 = close[i]; seg = close[i+1:i+h+1]
        out[d] = (seg[-1]/c0-1, seg.min()/c0-1, seg.max()/c0-1)
    return out

print("="*110)
print("【0】基线：回测窗口内随便挑一天的未来 20 日表现")
print("="*110)
FW = {}
for b in C.BOARD_ORDER:
    p = panels[b]; p = p[p.index >= pd.Timestamp(C.BACKTEST_START)]
    FW[b] = fwd(p)
    v = list(FW[b].values())
    up = np.mean([x[0] > 0 for x in v]); mdd = np.mean([x[1] for x in v]); run = np.mean([x[2] for x in v])
    print(f"  {C.BOARDS[b]['name']:>8}  n={len(v)}  上涨概率={up*100:5.1f}%  平均收益={np.mean([x[0] for x in v])*100:+5.2f}%  "
          f"平均最深回撤={mdd*100:+5.2f}%  平均最大上涨={run*100:+5.2f}%")

# ---- 破位强度 d 的几种定义 ----
def add_break(p):
    close = p["close"]; ma5 = p["ma5"]; ma20 = p["ma20"]
    p = p.copy()
    p["d_ma20"] = ((ma20 - close)/ma20/0.05).clip(0, 1)              # 跌破MA20幅度(5%记满)
    p["d_cross"] = ((ma20 - ma5)/ma20/0.03).clip(0, 1)               # MA5在MA20下方幅度
    p["d_brk"] = p[["d_ma20","d_cross"]].max(axis=1)                 # 综合破位强度
    p["ret1"] = close.pct_change()
    p["yin3"] = ((close.pct_change() < 0).rolling(3).sum() >= 2).astype(float)  # 近3日≥2阴
    p["bigdn"] = (close.pct_change() <= -0.02).astype(float)
    return p

P = {b: add_break(panels[b]) for b in C.BOARD_ORDER}

def ev_stats(b, mask, kind):
    """mask: 与 panel 同索引的布尔序列。返回 (事件数, 正确数, 平均收益, 平均风险)"""
    p = P[b]
    m = mask.reindex(p.index).fillna(False)
    m = m & (p.index >= pd.Timestamp(C.BACKTEST_START))
    ds = p.index[m]
    if len(ds) == 0: return (0, 0, np.nan, np.nan)
    # 合并连续段，取 H 最极端的一天
    h = p["H"]
    grp = (m != m.shift()).cumsum()
    picks = []
    for _, seg in h[m].groupby(grp[m]):
        picks.append(seg.idxmin() if kind == "bottom" else seg.idxmax())
    rows = []
    for d in sorted(set(picks)):
        if d in FW[b]: rows.append(FW[b][d])
    if not rows: return (0, 0, np.nan, np.nan)
    ret = np.array([r[0] for r in rows])
    risk = np.array([r[1] for r in rows]) if kind == "bottom" else np.array([r[2] for r in rows])
    ok = (ret > 0) & (risk >= -0.03) if kind == "bottom" else (ret < 0) & (risk <= 0.03)
    return (len(rows), int(ok.sum()), ret.mean(), risk.mean())

print("\n" + "="*110)
print("【1】底部侧：H 取不同低阈值时的表现（严格口径：20日收益>0 且 最深回撤≥-3%）")
print("="*110)
print(f"{'板块':>8} " + " ".join(f"|H<={-t:<5}" for t in (1.8, 2.0, 2.2, 2.4, 2.6)))
for b in C.BOARD_ORDER:
    line = f"{C.BOARDS[b]['name']:>8} "
    for t in (1.8, 2.0, 2.2, 2.4, 2.6):
        n, ok, r, rk = ev_stats(b, P[b]["H"] <= -t, "bottom")
        line += f" {ok}/{n}({ok/n*100 if n else 0:.0f}%)" if n else "      -     "
    print(line)

print("\n" + "="*110)
print("【2】顶部侧候选（严格口径：20日收益<0 且 最大踏空≤+3%）")
print("="*110)
cands = {
    "A 纯过热 H>=2.0":            lambda p: p["H"] >= 2.0,
    "B 纯过热 H>=2.4":            lambda p: p["H"] >= 2.4,
    "C 过热2.0+破位>=0.5":        lambda p: (p["H"] >= 2.0) & (p["d_brk"] >= 0.5),
    "D 过热1.6+破位>=0.5":        lambda p: (p["H"] >= 1.6) & (p["d_brk"] >= 0.5),
    "E 过热1.6+破位>=0.5+连阴":   lambda p: (p["H"] >= 1.6) & (p["d_brk"] >= 0.5) & (p["yin3"] > 0),
    "F 过热1.6+破位>=0.5+大阴":   lambda p: (p["H"] >= 1.6) & (p["d_brk"] >= 0.5) & (p["bigdn"] > 0),
    "G 过热1.3+破位>=0.8+连阴":   lambda p: (p["H"] >= 1.3) & (p["d_brk"] >= 0.8) & (p["yin3"] > 0),
    "H 过热1.3+破位>=0.8+大阴":   lambda p: (p["H"] >= 1.3) & (p["d_brk"] >= 0.8) & (p["bigdn"] > 0),
    "I 仅破位>=0.8+连阴":         lambda p: (p["d_brk"] >= 0.8) & (p["yin3"] > 0),
    "J 过热1.0+破位>=1.0":        lambda p: (p["H"] >= 1.0) & (p["d_brk"] >= 1.0),
}
hdr = f"{'候选':<24}" + "".join(f"{C.BOARDS[b]['name']:>10}" for b in C.BOARD_ORDER)
print(hdr)
for name, fn in cands.items():
    line = f"{name:<24}"
    tot_n = tot_ok = 0
    for b in C.BOARD_ORDER:
        n, ok, r, rk = ev_stats(b, fn(P[b]), "top")
        tot_n += n; tot_ok += ok
        line += f"{f'{ok}/{n}':>10}" if n else f"{'-':>10}"
    print(line + f"   合计 {tot_ok}/{tot_n}" + (f" ({tot_ok/tot_n*100:.0f}%)" if tot_n else ""))
