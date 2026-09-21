"""用生产代码 senti.model 做最终验证：事件清单 + 命中率 + 与随机基线对比。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, model
from senti.backtest import HORIZONS, MAIN_H

panels = model.build_all()

print("="*118)
print("SENTI-1 最终验证   近3年 " + C.BACKTEST_START + " ~ 末日")
print("="*118)

def fwd(close, d, h):
    i = close.index.get_loc(d)
    if i + h >= len(close): return None
    c0 = close.iloc[i]; seg = close.iloc[i+1:i+h+1]
    return seg.iloc[-1]/c0-1, seg.min()/c0-1, seg.max()/c0-1

def events(p, lo=0.0, hi=100.0, gap=20):
    s = p["score"].dropna(); out=[]
    for thr, how in ((lo,"min"), (hi,"max")):
        m = (s<thr) if how=="min" else (s>thr)
        if not m.any(): out.append([]); continue
        grp=(m!=m.shift()).cumsum()
        picks=[seg.idxmin() if how=="min" else seg.idxmax() for _,seg in s[m].groupby(grp[m])]
        picks=sorted(set(picks)); res=[]
        for d in picks:
            if res and (d-res[-1]).days<=gap: continue
            res.append(d)
        out.append(res)
    return out

def baseline(p, kind, h=MAIN_H, tol=0.03):
    close=p["close"]; ok=[]
    for i in range(len(close)-h):
        c0=close.iloc[i]; seg=close.iloc[i+1:i+h+1]
        r=seg.iloc[-1]/c0-1
        if kind=="bottom": ok.append(r>0 and seg.min()/c0-1>=-tol)
        else: ok.append(r<0 and seg.max()/c0-1<=tol)
    return float(np.mean(ok))*100

tb=ob=tt=ot=0; RB=[]; RM=[]; RT=[]; RU=[]
DETAIL=[]
for b in C.BOARD_ORDER:
    p=panels[b]; bl,tl=events(p)
    r=p["close"]
    bt=[]; tp=[]
    for d in bl:
        f=fwd(r,d,MAIN_H)
        if f: bt.append((d,f))
    for d in tl:
        f=fwd(r,d,MAIN_H)
        if f: tp.append((d,f))
    bok=sum(1 for _,f in bt if f[0]>0 and f[1]>=-0.03)
    tok=sum(1 for _,f in tp if f[0]<0 and f[2]<=0.03)
    tb+=len(bt); ob+=bok; tt+=len(tp); ot+=tok
    if bt: RB.append(np.mean([f[0] for _,f in bt])); RM.append(np.mean([f[1] for _,f in bt]))
    if tp: RT.append(np.mean([f[0] for _,f in tp])); RU.append(np.mean([f[2] for _,f in tp]))
    DETAIL.append((b, bt, tp, p))

for b, bt, tp, p in DETAIL:
    bd=p["close"].idxmin(); td=p["close"].idxmax()
    print(f"\n■ {C.BOARDS[b]['name']}  真底 {bd.date()}  真顶 {td.date()}   "
          f"分值区间 {p['score'].min():.0f} ~ {p['score'].max():.0f}  最新 {p['score'].iloc[-1]:.1f}")
    print(f"   真底日分值 {p['score'].loc[bd]:6.1f}   真顶日分值 {p['score'].loc[td]:6.1f}")
    print("   【底部溢出 <0】")
    for d,f in bt:
        ok = f[0]>0 and f[1]>=-0.03
        print(f"     {'OK ' if ok else 'FAIL'} {d.date()}  分{p['score'].loc[d]:6.1f}  "
              f"20日{f[0]*100:+7.2f}%  最深{f[1]*100:+7.2f}%")
    print("   【顶部溢出 >100】")
    for d,f in tp:
        ok = f[0]<0 and f[2]<=0.03
        print(f"     {'OK ' if ok else 'FAIL'} {d.date()}  分{p['score'].loc[d]:6.1f}  "
              f"20日{f[0]*100:+7.2f}%  最高{f[2]*100:+7.2f}%")

print("\n" + "="*118)
print("汇总（严格口径：底部=20日收益>0 且 最深回撤≥-3%；顶部=20日收益<0 且 踏空≤+3%）")
print("="*118)
print(f"  底部  {ob}/{tb} = {ob/tb*100:.0f}%    平均20日收益 {np.mean(RB)*100:+.2f}%   平均最深回撤 {np.mean(RM)*100:+.2f}%")
print(f"  顶部  {ot}/{tt} = {ot/tt*100:.0f}%    平均20日收益 {np.mean(RT)*100:+.2f}%   平均最大踏空 {np.mean(RU)*100:+.2f}%")
print(f"  合计  {ob+ot}/{tb+tt} = {(ob+ot)/(tb+tt)*100:.0f}%")
print("\n  随机基线（随便挑一天满足同一口径的比例）:")
for b in C.BOARD_ORDER:
    p=panels[b]
    print(f"    {C.BOARDS[b]['name']:>8}  底部基线 {baseline(p,'bottom'):5.1f}%   顶部基线 {baseline(p,'top'):5.1f}%")
