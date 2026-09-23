"""s6 —— SWING-2（MACD 动量拐点）长历史验证 2013~2026。

为什么 SWING-2 能做长历史而 SWING 不能：
  SWING-2 只用**指数收盘价**，不需要成分股广度；广度长表只有 2019-01 起。
  所以这里能用 data/cache/index_hist（2013-01-04 起，11 个宽基）跑满 13 年。

目的只有一个：过 AGENTS.md 的「2015-2016 那关」。
  SENTI-1 的教训是「19 个样本上再漂亮，不过 2015-2016 就不能采」。
  这里看 MACD 在 2015 年股灾、2016 年熔断上会不会连续误判。

判定口径与 SWING 完全一致：T+7 期末 > 0 且期间最深收盘回撤 >= -3%。
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C
from senti.model import _pct_map

_LO, _HI = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
_LC, _HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]
pm = lambda v: _pct_map(v, _LO, _HI, _LC, _HC)

HIST_DIR = os.path.join(C.CACHE_DIR, "index_hist")
# 中证2000 剔除：新浪返回的是替代序列，与中证1000 价格相关 -0.32（见 AGENTS.md 第 6 节）
BOARDS = ["SH", "CHINEXT", "CSI1000", "HS300", "CSI500", "CSI800", "SH180", "SZ50", "CSI100"]
NAME = {"SH": "大盘", "CHINEXT": "创业板", "CSI1000": "中证1000", "HS300": "沪深300",
        "CSI500": "中证500", "CSI800": "中证800", "SH180": "上证180",
        "SZ50": "上证50", "CSI100": "中证100"}

THR = 12.0
COOL = 4
H = 7
TOL = 0.03


def load(b: str) -> pd.Series:
    df = pd.read_parquet(os.path.join(HIST_DIR, f"{b}.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    return df.set_index("date")["close"].astype(float)


def score(c: pd.Series) -> pd.Series:
    """SWING-2 分值：MACD 柱 / 收盘价，因果锚分位（0~100，越低越超卖）。"""
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    mh = e12 - e26
    hist = mh - mh.ewm(span=9, adjust=False).mean()
    return pm(hist / c)


def fwd(c: pd.Series, h: int = H):
    a = c.to_numpy(float)
    n = len(a)
    e = np.full(n, np.nan)
    m = np.full(n, np.nan)
    for i in range(n - h):
        seg = a[i + 1:i + h + 1]
        e[i] = seg[-1] / a[i] - 1
        m[i] = seg.min() / a[i] - 1
    return e, m


def main():
    print(f"SWING-2 长历史验证　阈值 <={THR}　冷却 {COOL}　判定 T+{H} 期末>0 且回撤>=-{TOL:.0%}")
    print("剔除中证2000（坏序列）；科创板仅 2020 起，锚点预热后样本极少，不纳入\n")

    S, CLOSE, FW = {}, {}, {}
    for b in BOARDS:
        c = load(b)
        CLOSE[b] = c
        S[b] = score(c)
        FW[b] = fwd(c)

    def base(w0, w1):
        o = []
        for b in BOARDS:
            e, m = FW[b]
            idx = CLOSE[b].index
            for i in range(len(idx)):
                if not (w0 <= idx[i] <= w1):
                    continue
                if i + H > len(e) - 1:
                    continue
                o.append(e[i] > 0 and m[i] >= -TOL)
        return 100 * np.mean(o) if o else float("nan")

    def run(w0, w1, thr=THR):
        out = []
        for b in BOARDS:
            s = S[b].to_numpy(float)
            e, m = FW[b]
            idx = CLOSE[b].index
            last = -10 ** 9
            for i in range(1, len(s)):
                if np.isnan(s[i]) or s[i] > thr or i - last < COOL:
                    continue
                if not (w0 <= idx[i] <= w1):
                    continue
                if i + H > len(e) - 1:
                    continue
                last = i
                out.append((b, str(idx[i].date()), bool(e[i] > 0 and m[i] >= -TOL)))
        return out

    def wilson(k, n, z=1.96):
        if n == 0:
            return float("nan")
        p = k / n
        den = 1 + z * z / n
        c_ = p + z * z / (2 * n)
        m_ = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return 100 * (c_ - m_) / den

    W = pd.Timestamp("2013-01-01"), pd.Timestamp("2026-09-18")
    BL = base(*W)
    allx = run(*W)
    k = sum(1 for r in allx if r[2])
    print(f"全程 2013-01 ~ 2026-09（{len(BOARDS)} 宽基）")
    print(f"  基线 {BL:.1f}%")
    print(f"  SWING-2  n={len(allx)}  命中 {k}/{len(allx)} = {100*k/len(allx):.1f}%"
          f"  Wilson 下界 {wilson(k, len(allx)):.1f}%")

    print("\n分年度（★ = 必须过关的 2015 / 2016）")
    yr = {}
    for r in allx:
        yr.setdefault(r[1][:4], [0, 0])
        yr[r[1][:4]][0] += 1
        yr[r[1][:4]][1] += 1 if r[2] else 0
    for y in sorted(yr):
        n_, k_ = yr[y]
        mark = "★" if y in ("2015", "2016") else " "
        print(f"  {mark} {y}  {k_:>3}/{n_:<3} = {100*k_/n_:5.1f}%")

    print("\n分板块")
    bd = {}
    for r in allx:
        bd.setdefault(r[0], [0, 0])
        bd[r[0]][0] += 1
        bd[r[0]][1] += 1 if r[2] else 0
    for b in BOARDS:
        if b not in bd:
            continue
        n_, k_ = bd[b]
        print(f"  {NAME[b]:<8} {k_:>3}/{n_:<3} = {100*k_/n_:5.1f}%")

    print("\n2015-06 ~ 2016-02 股灾+熔断区间逐笔（最容易连环误判的一段）")
    sub = run(pd.Timestamp("2015-06-01"), pd.Timestamp("2016-02-29"))
    for r in sub:
        print(f"  {r[1]}  {NAME[r[0]]:<8} {'✔' if r[2] else '✘'}")

    print("\n近 3 年窗口复验（与 SWING 同窗口，便于对照）")
    for tag, (w0, w1) in [("2021-01~2026-09", (pd.Timestamp("2021-01-01"), pd.Timestamp("2026-09-18"))),
                          ("2023-09~2026-09", (pd.Timestamp("2023-09-20"), pd.Timestamp("2026-09-18")))]:
        x = run(w0, w1)
        kk = sum(1 for r in x if r[2])
        print(f"  {tag}  n={len(x):<4} 命中 {100*kk/len(x):.1f}%  基线 {base(w0,w1):.1f}%")


if __name__ == "__main__":
    main()
