# -*- coding: utf-8 -*-
"""t4: 对 t3 冒头的 rsi14 / cci20 / ret5 / bb_pctb20 做全套铁律体检。
关注量 = **并入现有并集后的边际**（Δ事件数、Δ事件命中率），两窗口 + 分半 + 逐年 + 阈值邻域。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"d:\情绪指标4")
import numpy as np
import pandas as pd
from senti import config as C, store, swing, swing2, swing3, data
from senti.model import _pct_map

LO, HI = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
H, TOL, COOL = 7, 0.03, 4
DISP = pd.Timestamp(C.BACKTEST_START); L57 = pd.Timestamp("2021-01-01")
def pm(v): return _pct_map(v, LO, HI, LC, HC)
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn.replace(0, np.nan))
def cci(c, n=20):
    ma = c.rolling(n).mean(); md = (c-ma).abs().rolling(n).mean().replace(0, np.nan)
    return (c-ma)/(0.015*md)
def pctb(c, n=20):
    ma = c.rolling(n).mean(); sd = c.rolling(n).std()
    return (c-(ma-2*sd))/(4*sd).replace(0, np.nan)

CAND = {"rsi14": rsi, "cci20": cci, "ret5": lambda c: c.pct_change(5),
        "bb_pctb20": pctb}

CLOSE = {b: data.load_index(b).set_index("date").sort_index()["close"].astype(float)
         for b in C.BOARD_ORDER}
sp, s2p, s3p = swing.load(), swing2.load(), swing3.load()
sr, s2r, s3r = swing.resonance(sp), swing2.resonance(s2p), swing3.resonance(s3p)
UEX = {b: store.union_events(b, sp[b], s2p[b], s3p[b], sr, s2r, s3r) for b in C.BOARD_ORDER}

def cluster(rows):
    evs = sorted(rows, key=lambda x: x[0]); last=None; out=[]
    for d, ok in evs:
        dt = pd.Timestamp(d)
        if last is None or (dt-last).days > 4:
            out.append((d, ok)); last = dt
    return out

def sim(c, trig):
    idx = c.index; n=len(idx); cv=c.to_numpy(float)
    tv = trig.reindex(idx).fillna(False).to_numpy(bool)
    out=[]; last=-10**9
    for i in range(n):
        if not tv[i] or i-last<COOL or i+H>n-1: continue
        last=i; seg=cv[i+1:i+H+1]
        out.append((str(idx[i].date()), bool(seg[-1]/cv[i]-1>0 and seg.min()/cv[i]-1>=-TOL)))
    return out

def per_board(fn, thr, rec):
    pb={}
    for b in C.BOARD_ORDER:
        s=pm(fn(CLOSE[b]))
        trig=(s<=thr)&(s>s.shift(1)) if rec else (s<=thr)
        pb[b]=sim(CLOSE[b], trig)
    return pb

def ustats(extra, window, mask=None):
    """extra: {b:[(date,ok)]} 并入现有并集，算事件级 n/率。mask: 日期过滤额外作用。"""
    ev=[]
    for b in C.BOARD_ORDER:
        rows=[(e["date"], e["ok"]) for e in UEX[b] if pd.Timestamp(e["date"])>=window]
        rows+= [(d,ok) for d,ok in extra.get(b,[]) if pd.Timestamp(d)>=window
                and (mask is None or mask(pd.Timestamp(d)))]
        ev+=cluster(rows)
    n=len(ev); k=sum(1 for _,ok in ev if ok)
    return n, (round(k/n*100,1) if n else None)

def sstats(extra, window, mask=None):
    """只算候选自身（不并现有）事件级，看它自己带来信号的命中。"""
    ev=[]
    for b in C.BOARD_ORDER:
        rows=[(d,ok) for d,ok in extra.get(b,[]) if pd.Timestamp(d)>=window
              and (mask is None or mask(pd.Timestamp(d)))]
        ev+=cluster(rows)
    n=len(ev); k=sum(1 for _,ok in ev if ok)
    return n,(round(k/n*100,1) if n else None)

print("=== 阈值邻域（recovery=True, 展示窗）：并入后 Δ件/Δ率 ===")
for name, fn in CAND.items():
    line=f"{name:<10}"
    for thr in (8,10,12,15):
        pb=per_board(fn, thr, True)
        bn,bk=ustats({}, DISP); n,k=ustats(pb, DISP)
        line+=f"  T{thr}:{n}件/{k}%(Δ{n-bn:+d},{round(k-bk,1) if k and bk else None})"
    print(line)

print("\n=== recovery 开关（thr=10, 展示窗）===")
for name, fn in CAND.items():
    for rec in (True, False):
        pb=per_board(fn, 10, rec); bn,bk=ustats({}, DISP); n,k=ustats(pb, DISP)
        sn,sk=sstats(pb, DISP)
        print(f"  {name:<10} rec={str(rec):<5} 自身{sn}件/{sk}%  并入{n}件/{k}%(Δ{n-bn:+d},{round(k-bk,1)})")

print("\n=== 分半（2025-06-01）：thr=10 recovery, 并入后各自窗口 ===")
mid=pd.Timestamp("2025-06-01")
for name, fn in CAND.items():
    pb=per_board(fn, 10, True)
    for tag,mk in (("前半",lambda d:d<mid),("后半",lambda d:d>=mid)):
        bn,bk=ustats({}, DISP, mk); n,k=ustats(pb, DISP, mk)
        sn,sk=sstats(pb, DISP, mk)
        print(f"  {name:<10} {tag}: 基线{bn}/{bk}% → 并入{n}/{k}%(Δ{n-bn:+d},{round((k or 0)-(bk or 0),1)})  自身新增{sn}/{sk}%")

print("\n=== 5.7年窗口（thr=10 recovery）并入增益 ===")
for name, fn in CAND.items():
    pb=per_board(fn, 10, True)
    bn,bk=ustats({}, L57); n,k=ustats(pb, L57)
    print(f"  {name:<10} 5.7y: 基线{bn}/{bk}% → 并入{n}/{k}% (Δ{n-bn:+d}, Δ{round((k or 0)-(bk or 0),1)})")

print("\n=== 逐年 自身新增信号命中（展示窗内, thr=10 rec）===")
for name, fn in CAND.items():
    pb=per_board(fn, 10, True); allrows=[(b,d,ok) for b,rows in pb.items() for d,ok in rows]
    byyear={}
    for b,d,ok in allrows:
        y=pd.Timestamp(d).year; byyear.setdefault(y,[]).append(ok)
    s=f"  {name:<10}"
    for y in sorted(byyear):
        v=byyear[y]; s+=f" {y}:{sum(v)}/{len(v)}"
    print(s)

print("\n=== 候选彼此重叠（自身信号日期 Jaccard, thr=10 rec, 展示窗）===")
sets={}
for name, fn in CAND.items():
    pb=per_board(fn, 10, True)
    sets[name]={ (b,d) for b,rows in pb.items() for d,_ in rows if pd.Timestamp(d)>=DISP}
ks=list(sets)
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        a,bb=sets[ks[i]],sets[ks[j]]
        jac=len(a&bb)/len(a|bb) if a|bb else 0
        print(f"  {ks[i]:<10}-{ks[j]:<10} J={jac:.2f}")
