"""诊断 14：最终标定 —— 看清真顶/真底在合成值 U 上的实际读数，确定最终刻度映射。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors

big = data.build_stock_indicators()
RAW = {}
for b in C.BOARD_ORDER:
    r = factors.build_board_raw(b, big)
    r = r[r.index >= pd.Timestamp(C.BACKTEST_START)].copy()
    d = data.load_index(b).set_index("date")
    for col in ("open","high","low"): r[col] = d[col].reindex(r.index)
    RAW[b] = r

LEVEL_F=["b20","b60","r5","nh","lim","rsi","bias","ret20"]
W={"b20":.18,"b60":.15,"r5":.11,"nh":.13,"lim":.09,"rsi":.14,"bias":.12,"ret20":.08}
def pmap(v,lq=2,hq=98):
    lo=np.nanpercentile(v.values,lq); hi=np.nanpercentile(v.values,hq)
    return (100.0*(v-lo)/(hi-lo)).clip(-25,125)

def build(b, wl=0.5, lq=2, hq=98):
    r=RAW[b]
    s=pd.Series(0.0,index=r.index); tw=0.0
    for k in LEVEL_F:
        s=s+pmap(r[k].astype(float),lq,hq)*W[k]; tw+=W[k]
    L=s/tw
    T=(pmap(r["bias"],lq,hq)*.30+pmap(r["ret20"],lq,hq)*.30
       +pmap(r["rsi"],lq,hq)*.22+r["amt_pct"]*100*.18)
    return L, T, wl*L+(1-wl)*T

print("="*120)
print("【1】各板块在真实拐点日的 L / T / U 读数（wl=0.5, p2/p98）")
print("="*120)
print(f"{'板块':>8} {'真底日':<12}{'L':>7}{'T':>7}{'U':>7} | {'真顶日':<12}{'L':>7}{'T':>7}{'U':>7} | {'U分布 p1/p50/p99':>22}")
Us={}
for b in C.BOARD_ORDER:
    L,T,U = build(b); Us[b]=(L,T,U)
    r=RAW[b]; bd=r["close"].idxmin(); td=r["close"].idxmax()
    print(f"{C.BOARDS[b]['name']:>8} {str(bd.date()):<12}{L.loc[bd]:7.1f}{T.loc[bd]:7.1f}{U.loc[bd]:7.1f} | "
          f"{str(td.date()):<12}{L.loc[td]:7.1f}{T.loc[td]:7.1f}{U.loc[td]:7.1f} | "
          f"{U.quantile(.01):7.1f}{U.quantile(.5):7.1f}{U.quantile(.99):7.1f}")

print("\n" + "="*120)
print("【2】L/T 权重对真顶覆盖的影响（U = wl·L + (1-wl)·T）")
print("="*120)
print(f"{'wl':>5} | " + " | ".join(f"{C.BOARDS[b]['name']}顶" for b in C.BOARD_ORDER))
for wl in (0.7, 0.5, 0.4, 0.3, 0.2, 0.0):
    cells=[]
    for b in C.BOARD_ORDER:
        L,T,U = build(b, wl=wl)
        td=RAW[b]["close"].idxmax()
        cells.append(f"{U.loc[td]:6.1f}")
    print(f"{wl:>5} | " + " | ".join(cells))

print("\n" + "="*120)
print("【3】最终刻度：把 U 的分位映射到 0~100（q 越小，溢出越罕见）")
print("="*120)
for lq in (2, 5, 10, 15):
    print(f"\n--- 因子锚点 p{lq}/p{100-lq} ---")
    print(f"{'板块':>8} {'真底U':>8}{'真顶U':>8} | {'最低3天':<46} | {'最高3天':<46}")
    for b in C.BOARD_ORDER:
        L,T,U = build(b, wl=0.5, lq=lq, hq=100-lq)
        lo=U.nsmallest(3); hi=U.nlargest(3)
        bd=RAW[b]["close"].idxmin(); td=RAW[b]["close"].idxmax()
        print(f"{C.BOARDS[b]['name']:>8} {U.loc[bd]:8.1f}{U.loc[td]:8.1f} | "
              + " ".join(f"{str(d.date())[5:]}({v:.0f})" for d,v in lo.items()).ljust(46) + " | "
              + " ".join(f"{str(d.date())[5:]}({v:.0f})" for d,v in hi.items()))
