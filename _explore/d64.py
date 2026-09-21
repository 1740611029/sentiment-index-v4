"""路线②：放弃「单点百分百判定」，改成「底部区域 + 分批建仓」，用区间换确定性。

背景：底部单点信号在生产窗口是 17/19 = 89.5%，但最差回撤 −23.61%，
      且仅剩的 2 次失败是同一天（2024-01-23 中小盘流动性危机）。
      继续在「单点」上加过滤已经被长历史否决了 4 次。

这次换产品形态：信号出现后不再一把梭，而是分批部署。检验的问题是——
  「分批能不能把 −23.61% 那种深套变成可承受的浮亏，同时不牺牲命中率？」

策略（全部以**同一个信号日**为起点，只是资金部署方式不同）：
  S1 单点     信号日收盘 100%
  S3 跌补     信号日 1/3，之后 40 个交易日内每创一次收盘新低补 1/3（最多 2 次），
              未触发的份额在第 40 日无条件补满
  S4 两段     信号日 1/2，第 10 个交易日 1/2
  S5 三段     信号日 1/3，第 10 日 1/3，第 20 日 1/3（纯时间分批，不看价格）

口径：
  组合净值收益率 = 当日总市值 / 当日累计投入 − 1   （正确反映分批的资金占用）
  最大浮亏      = 信号日 → 信号日+60 区间内，逐日 mark-to-market 的最低值
  期末收益      = 信号日+60 那天的组合净值收益率
  命中          = 期末收益 > 0

纪律：生产好看不算，必须过 2015-2016。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C
from senti import model as M
from senti import factors as F
from senti import data as D
import longhist

H, COOL, THR = 60, 20, 0.0
W2 = pd.Timestamp("2023-09-20")
LC, HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
LO_Q, HI_Q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
EXCLUDE = {"CSI2000"}
KEYS = ["b20", "b60", "r5", "nh", "lim", "rsi", "bias", "ret20", "disp", "dd250"]
WIN, MINP = C.MODEL["anchor_window"], C.MODEL["anchor_min"]
CLIP_LO, CLIP_HI = -30.0, 140.0


def make_anchors(raw, mode):
    src = raw.copy()
    src["dd250"] = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    if mode == "trail":
        return {k: (src[k].rolling(WIN, min_periods=MINP).quantile(LO_Q / 100.0).shift(1),
                    src[k].rolling(WIN, min_periods=MINP).quantile(HI_Q / 100.0).shift(1)) for k in KEYS}
    out = {}
    for k in KEYS:
        x = src[k].to_numpy(float); n = len(x)
        lo = np.full(n, np.nan); hi = np.full(n, np.nan); j = 500
        while j < n:
            seg = x[:j]; seg = seg[~np.isnan(seg)]
            if len(seg) >= 500:
                e = min(j + 20, n)
                lo[j:e] = np.nanpercentile(seg, LO_Q); hi[j:e] = np.nanpercentile(seg, HI_Q)
            j += 20
        out[k] = (pd.Series(lo, index=src.index).ffill(), pd.Series(hi, index=src.index).ffill())
    return out


def pmm(v, a, k):
    lo, hi = a[k]; d = (hi - lo).replace(0.0, np.nan)
    return (100.0 * (v - lo) / d).clip(LC, HC)


def score(raw, a):
    num = pd.Series(0.0, index=raw.index); den = 0.0
    for k, w in M.L_WEIGHTS.items():
        num = num + pmm(raw[k], a, k) * w; den += w
    L = num / den
    T = (pmm(raw["bias"], a, "bias") * M.T_WEIGHTS["bias"]
         + pmm(raw["ret20"], a, "ret20") * M.T_WEIGHTS["ret20"]
         + pmm(raw["rsi"], a, "rsi") * M.T_WEIGHTS["rsi"]
         + raw["amt_pct"] * 100.0 * M.T_WEIGHTS["amt"])
    U = M.W_L * L + (1 - M.W_L) * T
    cf_low = ((M.CF_L_HI - L) / M.CF_L_HI).clip(0, 1).fillna(0.0)
    cf_amt = ((raw["amt_pct"].astype(float) - M.CF_AMT_LO) / (1.0 - M.CF_AMT_LO)).clip(0, 1).fillna(0.0)
    cfb = (cf_amt * cf_low).fillna(0.0)
    return (U - M.K_B * cfb).clip(CLIP_LO, CLIP_HI)


def sig_idx(s):
    """信号日的整数下标（昨日 ≤ THR 且今日 > 昨日，冷却 COOL）。"""
    v = s.to_numpy(float); out = []; last = -10 ** 9; n = len(v)
    for i in range(1, n - H):
        if np.isnan(v[i]) or np.isnan(v[i - 1]): continue
        if i - last < COOL: continue
        if v[i - 1] <= THR and v[i] > v[i - 1]:
            last = i; out.append(i)
    return out


def deploy(cl, i, mode):
    """返回 [(买入日下标, 资金权重)]，权重合计 1.0。"""
    n = len(cl)
    if mode == "S1":
        return [(i, 1.0)]
    if mode == "S3":
        b = [(i, 1 / 3.0)]; run_min = cl[i]; end40 = min(i + 40, n - 1)
        for j in range(i + 1, end40 + 1):
            if cl[j] < run_min:
                run_min = cl[j]; b.append((j, 1 / 3.0))
                if len(b) == 3: break
        used = sum(w for _, w in b)
        if used < 0.999:
            b.append((end40, 1.0 - used))
        return b
    if mode == "S4":
        return [(i, 0.5), (min(i + 10, n - 1), 0.5)]
    if mode == "S5":
        return [(i, 1 / 3.0), (min(i + 10, n - 1), 1 / 3.0), (min(i + 20, n - 1), 1 / 3.0)]
    if mode.startswith("S3x"):          # 跌补，份数可变
        k = int(mode[3:])
        w = 1.0 / k
        b = [(i, w)]; run_min = cl[i]; end40 = min(i + 40, n - 1)
        for j in range(i + 1, end40 + 1):
            if cl[j] < run_min:
                run_min = cl[j]; b.append((j, w))
                if len(b) == k: break
        used = sum(x for _, x in b)
        if used < 0.999:
            b.append((end40, 1.0 - used))
        return b
    raise ValueError(mode)


def evaluate(cl, buys, H=60):
    """组合净值口径：市值/累计投入−1。返回 (期末收益, 最大浮亏) 或 None。"""
    n = len(cl)
    end = min(buys[0][0] + H, n - 1)
    if end <= buys[0][0]:
        return None
    bmap = {}
    for j, w in buys:
        j = min(j, end)
        bmap[j] = bmap.get(j, 0.0) + w
    shares = 0.0; inv = 0.0; mdd = 0.0
    for t in range(buys[0][0], end + 1):
        if t in bmap:
            shares += bmap[t] / cl[t]; inv += bmap[t]
        if inv > 1e-9:
            mdd = min(mdd, shares * cl[t] / inv - 1.0)
    return shares * cl[end] / inv - 1.0, mdd


def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


MODES = [("S1 单点", "S1"), ("S3 跌补1/3×3", "S3"), ("S3x2 跌补1/2×2", "S3x2"),
         ("S3x4 跌补1/4×4", "S3x4"), ("S4 两段", "S4"), ("S5 三段", "S5")]


def report(rows_by_mode, title, extra_note=""):
    print("\n" + "=" * 132)
    print(title + extra_note)
    print("=" * 132)
    print(f"  {'策略':<16}{'n':>4}{'期末盈利':>10}{'命中率':>9}{'均收益':>10}{'中位':>9}"
          f"{'最差浮亏':>10}{'浮亏>-8%':>10}{'浮亏>-15%':>11}{'最差期末':>10}")
    for lab, m in MODES:
        rows = rows_by_mode[m]
        if not rows:
            print(f"  {lab:<16} 无样本"); continue
        r = np.array([x[2] for x in rows]); d = np.array([x[3] for x in rows])
        win = int((r > 0).sum())
        print(f"  {lab:<16}{len(rows):>4}{win:>10}{win/len(rows)*100:>8.1f}%"
              f"{r.mean()*100:>+9.2f}%{np.median(r)*100:>+8.2f}%"
              f"{d.min()*100:>+9.2f}%{(d > -0.08).mean()*100:>9.1f}%"
              f"{(d > -0.15).mean()*100:>10.1f}%{r.min()*100:>+9.2f}%")


# ==================== ① 生产 6 板块 ====================
print("=" * 132)
print("① 生产 6 板块（因果锚，阈值 0，信号日 +60 交易日，组合净值口径）")
print("=" * 132)
RAW = {b: F.build_board_raw(b) for b in C.BOARD_ORDER}
prod = {m: [] for _, m in MODES}
for b in C.BOARD_ORDER:
    raw = RAW[b]
    sc = score(raw, make_anchors(raw, "trail"))
    sc = sc[sc.index >= W2]
    cl = raw["close"][raw.index >= W2].to_numpy(float)
    for i in sig_idx(sc):
        for _, m in MODES:
            r = evaluate(cl, deploy(cl, i, m))
            if r: prod[m].append((b, sc.index[i], r[0], r[1]))
report(prod, "① 生产 6 板块 · 全窗口")

print("\n  【两次失败事件逐笔对比】(2024-01-23 那两次)")
for lab, m in MODES:
    sub = [x for x in prod[m] if str(x[1])[:7] in ("2024-01", "2024-02")]
    for x in sub:
        print(f"    {lab:<16} {x[0]:<9} {x[1].date()}  期末 {x[2]*100:+6.2f}%   最大浮亏 {x[3]*100:+7.2f}%")

# ==================== ② 长历史 trailing-750 ====================
PAN = {k: v for k, v in longhist.build().items() if k not in EXCLUDE}
NAME2FILE = {k: v[1] for k, v in longhist.BOARDS.items()}
NAME2FILE.update({v[0]: v[1] for v in longhist.BOARDS.values()})

def run_long(mode_anchor):
    res = {m: [] for _, m in MODES}
    for nm, raw in PAN.items():
        sc = score(raw, make_anchors(raw, mode_anchor))
        cl = raw["close"].to_numpy(float)
        for i in sig_idx(sc):
            for _, m in MODES:
                r = evaluate(cl, deploy(cl, i, m))
                if r: res[m].append((nm, sc.index[i], r[0], r[1]))
    return res

res_trail = run_long("trail")
report(res_trail, "② 长历史 8 宽基 · trailing-750 锚")
report({m: [x for x in res_trail[m] if x[1] >= W2] for _, m in MODES},
       "②b 长历史 · trailing-750 锚", " · 仅交付窗口")

res_exp = run_long("expand")
report(res_exp, "③ 长历史 8 宽基 · 扩展因果锚（含 2015-2016）")

print("\n  【分年度 · 期末盈利】（扩展锚，命中/总数）")
yrs = sorted({x[1].year for _, m in MODES for x in res_exp[m]})
print(f"    {'策略':<16}" + "".join(f"{y:>9}" for y in yrs))
for lab, m in MODES:
    cells = ""
    for y in yrs:
        sub = [x for x in res_exp[m] if x[1].year == y]
        cells += (f"{int((np.array([x[2] for x in sub]) > 0).sum())}/{len(sub)}" if sub else "-").rjust(9)
    print(f"    {lab:<16}" + cells)

print("\n  【平均最大浮亏】")
for lab, m in MODES:
    for name, res in (("生产", prod), ("长历史-扩展锚", res_exp)):
        d = np.array([x[3] for x in res[m]])
        print(f"    {lab:<16} {name:<16} 均值 {d.mean()*100:+6.2f}%   中位数 {np.median(d)*100:+6.2f}%")
