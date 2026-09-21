"""失败诊断：22 次底部信号里，5 次失败的和 17 次成功的，入场当天到底差在哪？

不猜参数，先把特征摊开看。若某个特征能把两类干净分开，再拿去长历史（尤其 2015-2016）验证。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, store

H, TOL, COOL = store.H, store.TOL, store.COOL_TRADING
THR = store.ENTRY_THR

PAN = store.load()
print("面板列：", sorted(PAN["HS300"].columns.tolist()), "\n")

FEATS = [
    ("score", "score"), ("L", "L"), ("T", "T"), ("U", "U"),
    ("cf_b", "cf_b"), ("cf_g", "cf_g"),
    ("b20", "b20"), ("b60", "b60"), ("r5", "r5"), ("nh", "nh"), ("lim", "lim"),
    ("disp", "disp"), ("amt_pct", "amt_pct"), ("dd250", "dd250"),
    ("bias", "bias"), ("ret20", "ret20"), ("rsi", "rsi"), ("vol", "vol"),
]


def row(p, i):
    out = {}
    for lab, col in FEATS:
        if col in p.columns:
            v = p[col].iloc[i]
            out[lab] = float(v) if pd.notna(v) else np.nan
    # 派生：价格相对均线的位置
    c = p["close"]
    for n in (5, 10, 20, 60):
        m = c.rolling(n).mean().iloc[i]
        out[f"c>ma{n}"] = float(c.iloc[i] > m) if pd.notna(m) else np.nan
    out["chg1"] = float(c.iloc[i] / c.iloc[i - 1] - 1)
    out["chg5"] = float(c.iloc[i] / c.iloc[i - 5] - 1) if i >= 5 else np.nan
    lo250 = c.rolling(250, min_periods=60).min().iloc[i]
    out["above_lo250"] = float(c.iloc[i] / lo250 - 1) if pd.notna(lo250) else np.nan
    hi250 = c.rolling(250, min_periods=60).max().iloc[i]
    out["below_hi250"] = float(c.iloc[i] / hi250 - 1) if pd.notna(hi250) else np.nan
    # 近 5 日是否还在创新低
    out["new_lo20_5d"] = float(c.iloc[i] <= c.iloc[max(0, i - 4):i + 1].min())
    return out


rows = []
for b in C.BOARD_ORDER:
    p = PAN[b]
    s = p["score"].to_numpy(float)
    cl = p["close"].to_numpy(float)
    n = len(s)
    last = -10 ** 9
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]): continue
        if i - last < COOL: continue
        if s[i - 1] <= THR and s[i] > s[i - 1]:
            last = i
            c0 = cl[i]; seg = cl[i + 1:i + H + 1]
            ret = float(seg[-1] / c0 - 1); mdd = float(seg.min() / c0 - 1)
            r = row(p, i)
            r.update(board=C.BOARDS[b]["name"], date=str(p.index[i].date()),
                     prev=float(s[i - 1]), ret=ret, mdd=mdd,
                     ok=bool(ret > 0 and mdd >= -TOL))
            rows.append(r)

df = pd.DataFrame(rows)
print("=" * 130)
print(f"共 {len(df)} 次信号：命中 {df['ok'].sum()}，失败 {(~df['ok']).sum()}")
print("=" * 130)

cols = ["board", "date", "ok", "ret", "mdd", "prev", "score", "L", "T", "cf_b", "cf_g",
        "b20", "b60", "lim", "disp", "amt_pct", "dd250", "bias", "ret20", "rsi",
        "chg1", "chg5", "above_lo250", "below_hi250", "c>ma5", "c>ma10", "c>ma20", "c>ma60", "new_lo20_5d"]
cols = [c for c in cols if c in df.columns]
pd.set_option("display.width", 250, "display.max_columns", 60)
show = df[cols].copy()
for c in show.columns:
    if show[c].dtype.kind == "f":
        show[c] = show[c].round(3)
print(show.sort_values(["ok", "mdd"]).to_string(index=False))

print("\n" + "=" * 130)
print("命中组 vs 失败组：各特征均值（看哪几列拉得开）")
print("=" * 130)
num = [c for c in df.columns if df[c].dtype.kind == "f" and c not in ("ret", "mdd")]
g = df.groupby("ok")[num].mean().T
g["差(命中-失败)"] = g.get(True, 0) - g.get(False, 0)
g["分离度"] = (g["差(命中-失败)"] / df[num].std()).abs()
print(g.round(3).sort_values("分离度", ascending=False).to_string())

df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d53_signals.csv"),
          index=False, encoding="utf-8-sig")
print("\n明细已存 d53_signals.csv")
