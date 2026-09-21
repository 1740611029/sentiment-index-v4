"""稳定性检验：新增数据会不会改写已发布的历史分值？

这是纯粹的工程正确性检验，不看命中率。
做法：分别用「截至 2026-03-31」和「截至 2026-09-18」的数据算一遍分值序列，
比较两段重叠区间的分值。

  全样本锚  → 历史会被大幅改写（今天新增的数据会重算 2023 年的分值）
  因果锚    → 重叠区间应当逐日完全一致
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data, factors
from senti import model as M
import longhist

LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
LO_Q, HI_Q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
KEYS = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20", "disp", "dd250"]
T1 = pd.Timestamp("2026-03-31")


def mkeys(raw):
    src = raw.copy()
    src["dd250"] = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    return src


def anchors_full(src):
    return {k: (float(np.nanpercentile(src[k].to_numpy(float), LO_Q)),
                float(np.nanpercentile(src[k].to_numpy(float), HI_Q))) for k in KEYS}


def anchors_trail(src, win):
    return {k: (src[k].rolling(win, min_periods=win).quantile(LO_Q / 100.0).shift(1),
                src[k].rolling(win, min_periods=win).quantile(HI_Q / 100.0).shift(1)) for k in KEYS}


def pmm(v, a, k):
    lo, hi = a[k]; d = hi - lo
    if isinstance(d, pd.Series):
        d = d.replace(0.0, np.nan)
    elif d == 0:
        d = np.nan
    return (100.0 * (v - lo) / d).clip(LC, HC)


def score(src, a):
    num = pd.Series(0.0, index=src.index); den = 0.0
    for k, w in M.L_WEIGHTS.items():
        num = num + pmm(src[k], a, k) * w; den += w
    L = num / den
    T = (pmm(src["bias"], a, "bias") * M.T_WEIGHTS["bias"]
         + pmm(src["ret20"], a, "ret20") * M.T_WEIGHTS["ret20"]
         + pmm(src["rsi"], a, "rsi") * M.T_WEIGHTS["rsi"]
         + src["amt_pct"] * 100.0 * M.T_WEIGHTS["amt"])
    U = M.W_L * L + (1 - M.W_L) * T
    cfb = (((src["amt_pct"].astype(float) - M.CF_AMT_LO) / (1.0 - M.CF_AMT_LO)).clip(0, 1).fillna(0.0)
           * ((M.CF_L_HI - L) / M.CF_L_HI).clip(0, 1).fillna(0.0)).fillna(0.0)
    f_disp = (100.0 - pmm(src["disp"], a, "disp")).fillna(0.0)
    f_dd = (100.0 - pmm(src["dd250"], a, "dd250")).fillna(0.0)
    cfg = (((f_dd - M.GR_DD_A) / M.GR_DD_W).clip(0, 1)
           * ((f_disp - M.GR_DP_A) / M.GR_DP_W).clip(0, 1)).fillna(0.0)
    return (U - M.K_B * cfb - M.K_G * cfg).clip(M.CLIP_LO, M.CLIP_HI)


PAN = longhist.build()
NAMES = ["HS300", "CHINEXT", "CSI1000", "CSI500"]
print("=" * 116)
print("历史改写检验：截至 2026-03-31 vs 截至 2026-09-18，重叠区间分值差异")
print("=" * 116)
print(f"  {'板块':<9}{'口径':<16}{'重叠天数':>8}{'完全相同':>9}{'均绝对差':>10}{'最大差':>10}{'相关系数':>10}")

CONFIGS = [("全样本锚（现状）", None)] + [(f"trailing {w}日", w) for w in (500, 750, 1000, 1250, 1500)]
for nm in NAMES:
    src_full = mkeys(PAN[nm])
    src_t1 = src_full[src_full.index <= T1]
    for tag, win in CONFIGS:
        fn = anchors_full if win is None else (lambda s, w=win: anchors_trail(s, w))
        s2 = score(src_full, fn(src_full))
        s1 = score(src_t1, fn(src_t1))
        j = pd.concat([s1.rename("a"), s2.rename("b")], axis=1).dropna()
        j = j[j.index <= T1]
        if j.empty:
            print(f"  {nm:<9}{tag:<16}{'0':>8}"); continue
        d = (j["a"] - j["b"]).abs()
        same = (d < 1e-9).mean()
        print(f"  {nm:<9}{tag:<16}{len(j):>8}{same*100:>8.1f}%{d.mean():>10.4f}"
              f"{d.max():>10.4f}{j['a'].corr(j['b']):>10.4f}")
    print()

print("=" * 116)
print("结论口径说明：'完全相同' 一列若不是 100%，说明新增数据会改写已发布的历史分值。")
print("=" * 116)
