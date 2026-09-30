# -*- coding: utf-8 -*-
"""t5: 因子「结合」——多振荡器共振 / max(AND) / min(OR) / 加权平均，看能否造出
比现役并集更优（事件数不降 且 命中率升）的第 4 路信号。全部 分位+回升 语义。"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"d:\情绪指标4")
import numpy as np
import pandas as pd
from senti import config as C, store, swing, swing2, swing3, data
from senti.model import _pct_map
LO,HI=C.MODEL["anchor_lo"],C.MODEL["anchor_hi"]; LC,HC=C.MODEL["map_clip_lo"],C.MODEL["map_clip_hi"]
H,TOL,COOL=7,0.03,4
DISP=pd.Timestamp(C.BACKTEST_START); L57=pd.Timestamp("2021-01-01"); MID=pd.Timestamp("2025-06-01")
def pm(v): return _pct_map(v,LO,HI,LC,HC)
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(c,n=14):
    d=c.diff();up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))
def cci(c,n=20):
    ma=c.rolling(n).mean();md=(c-ma).abs().rolling(n).mean().replace(0,np.nan);return (c-ma)/(0.015*md)
def pctb(c,n=20):
    ma=c.rolling(n).mean();sd=c.rolling(n).std();return (c-(ma-2*sd))/(4*sd).replace(0,np.nan)
def wpr(c,n=14):
    hh=c.rolling(n).max();ll=c.rolling(n).min();return -(hh-c)/(hh-ll).replace(0,np.nan)*100

FAC={"rsi":rsi,"cci":cci,"pctb":pctb,"ret5":lambda c:c.pct_change(5),"wpr":wpr}
CLOSE={b:data.load_index(b).set_index("date").sort_index()["close"].astype(float) for b in C.BOARD_ORDER}
sp,s2p,s3p=swing.load(),swing2.load(),swing3.load()
sr,s2r,s3r=swing.resonance(sp),swing2.resonance(s2p),swing3.resonance(s3p)
UEX={b:store.union_events(b,sp[b],s2p[b],s3p[b],sr,s2r,s3r) for b in C.BOARD_ORDER}

# 每板块的振荡器分位表
SC={b:pd.DataFrame({k:pm(f(CLOSE[b])) for k,f in FAC.items()}) for b in C.BOARD_ORDER}

def cluster(rows):
    evs=sorted(rows,key=lambda x:x[0]);last=None;out=[]
    for d,ok in evs:
        dt=pd.Timestamp(d)
        if last is None or (dt-last).days>4: out.append((d,ok));last=dt
    return out
def sim(c,trig):
    idx=c.index;n=len(idx);cv=c.to_numpy(float);tv=trig.reindex(idx).fillna(False).to_numpy(bool)
    out=[];last=-10**9
    for i in range(n):
        if not tv[i] or i-last<COOL or i+H>n-1: continue
        last=i;seg=cv[i+1:i+H+1]
        out.append((str(idx[i].date()),bool(seg[-1]/cv[i]-1>0 and seg.min()/cv[i]-1>=-TOL)))
    return out
def ustats(extra,window,mask=None):
    ev=[]
    for b in C.BOARD_ORDER:
        rows=[(e["date"],e["ok"]) for e in UEX[b] if pd.Timestamp(e["date"])>=window]
        rows+=[(d,ok) for d,ok in extra.get(b,[]) if pd.Timestamp(d)>=window and (mask is None or mask(pd.Timestamp(d)))]
        ev+=cluster(rows)
    n=len(ev);k=sum(1 for _,ok in ev if ok);return n,(round(k/n*100,1) if n else None)
def sstats(extra,window,mask=None):
    ev=[]
    for b in C.BOARD_ORDER:
        rows=[(d,ok) for d,ok in extra.get(b,[]) if pd.Timestamp(d)>=window and (mask is None or mask(pd.Timestamp(d)))]
        ev+=cluster(rows)
    n=len(ev);k=sum(1 for _,ok in ev if ok);return n,(round(k/n*100,1) if n else None)

def make_trig(mode,thr=10,k=2,rec=True):
    """返回 {b: [(date,ok)]}。mode: and_all / consensus_k / avg / or_any"""
    pb={}
    for b in C.BOARD_ORDER:
        S=SC[b]
        if mode=="and_all": comp=S.max(axis=1)            # 全部超卖→最大者<=thr
        elif mode=="or_any": comp=S.min(axis=1)           # 任一超卖
        elif mode=="avg":    comp=S.mean(axis=1)
        elif mode.startswith("cons"):
            cnt=(S<=thr).sum(axis=1); base=(cnt>=k)
            # 回升：低分位成员数较昨日增加
            prev=(S.shift(1)<=thr).sum(axis=1)
            trig=base&(cnt>prev) if rec else base
            pb[b]=sim(CLOSE[b],trig); continue
        trig=(comp<=thr)&(comp>comp.shift(1)) if rec else (comp<=thr)
        pb[b]=sim(CLOSE[b],trig)
    return pb

print("=== 组合式第4路（thr=10, 回升）并入现有并集：Δ件/Δ率 ===")
bn,bk=ustats({},DISP); b5n,b5k=ustats({},L57)
print(f"基线 展示 {bn}/{bk}%   5.7y {b5n}/{b5k}%\n")
for mode,kk in [("and_all",None),("avg",None),("or_any",None),("cons2",2),("cons3",3),("cons4",4)]:
    pb=make_trig("cons",10,kk) if mode.startswith("cons") else make_trig(mode,10)
    n,k=ustats(pb,DISP); sn,sk=sstats(pb,DISP)
    n5,k5=ustats(pb,L57)
    print(f"  {mode:<10} 自身{sn:>3}件/{sk}%  并入{n}件/{k}%(Δ{n-bn:+d},{round((k or 0)-bk,1)})  5.7y并入{n5}/{k5}%(Δ{k5-b5k if k5 else None})")

print("\n=== 最有量的 cons2/cons3 做阈值邻域 + 分半 ===")
for kk in (2,3):
    print(f"--- consensus>={kk} ---")
    for thr in (10,15,20,25):
        pb=make_trig("cons",thr,kk); n,k=ustats(pb,DISP)
        pf,bf=ustats(pb,DISP,lambda d:d<MID); nb,bk2=ustats({},DISP,lambda d:d<MID)
        ph,bh=ustats(pb,DISP,lambda d:d>=MID); hb,hbk=ustats({},DISP,lambda d:d>=MID)
        print(f"  thr{thr}: 并入{n}/{k}%(Δ{n-bn:+d})  前半Δ{pf-nb:+d},{round((pf or 0)-bk2,1)}  后半Δ{ph-hb:+d},{round((ph or 0)-hbk,1)}")
