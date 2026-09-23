"""s8 —— 小波段候选因子批量扫描框架（第四个模型搜索 · 第一轮广扫）。

统一口径（与 SWING / SWING-2 完全一致，便于直接对照）
------------------------------------------------------
  数据    data/cache/index_hist 的 9 个宽基（剔除中证2000，新浪返回坏序列）
  分值    候选因子 → 因果锚分位（复用 SENTI-1 的 model._pct_map），0~100 越低越超卖
  信号    score <= thr，冷却 cool 个交易日，逐日模拟（无未来函数）
  判定    T+H 期末收益 > 0 且期间最深收盘回撤 >= -TOL
  必报    n / 命中率 / 随机基线 / Wilson 95% 下界 / 2015 / 2016（AGENTS.md 的关）

为什么用 9 宽基而不是 6 板块
----------------------------
  6 板块只有 2019 起（锚点预热后 2021 起），过不了「2015-2016 那关」。
  9 宽基从 2013-01 起，能同时给长短两个窗口，且样本量大 3 倍。

用法
----
  python _explore/s8.py            # 扫描 FACTORS 里全部因子
  python _explore/s8.py pos macd   # 只跑指定因子（对照用）
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
pm = lambda v: _pct_map(v, _LO, _HI, _LC, _HC)          # noqa: E731

HIST_DIR = os.path.join(C.CACHE_DIR, "index_hist")
BOARDS = ["SH", "CHINEXT", "CSI1000", "HS300", "CSI500", "CSI800",
          "SH180", "SZ50", "CSI100"]
NAME = {"SH": "大盘", "CHINEXT": "创业板", "CSI1000": "中证1000", "HS300": "沪深300",
        "CSI500": "中证500", "CSI800": "中证800", "SH180": "上证180",
        "SZ50": "上证50", "CSI100": "中证100"}

H = 7
TOL = 0.03

W_ALL = (pd.Timestamp("2013-01-01"), pd.Timestamp("2026-09-18"))
W_57 = (pd.Timestamp("2021-01-01"), pd.Timestamp("2026-09-18"))
W_3 = (pd.Timestamp("2023-09-20"), pd.Timestamp("2026-09-18"))


# ---------------------------------------------------------------- 数据
def load(b: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(HIST_DIR, f"{b}.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").set_index("date")


def fwd(c: pd.Series, h: int = H):
    """未来 h 日：期末收益、期间最深收盘回撤。只用未来，供判定用。"""
    a = c.to_numpy(float)
    n = len(a)
    e = np.full(n, np.nan)
    m = np.full(n, np.nan)
    for i in range(n - h):
        seg = a[i + 1:i + h + 1]
        e[i] = seg[-1] / a[i] - 1
        m[i] = seg.min() / a[i] - 1
    return e, m


def wilson(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return float("nan")
    p = k / n
    den = 1 + z * z / n
    c_ = p + z * z / (2 * n)
    m_ = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c_ - m_) / den


# ---------------------------------------------------------------- 因子
# 约定：返回 (raw_series, direction)
#   direction = +1 → 原始值越小越超卖（直接映射）
#   direction = -1 → 原始值越大越超卖（取负后映射）
def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


FACTORS: dict[str, tuple] = {}


def F(name, direction=1):
    def deco(fn):
        FACTORS[name] = (fn, direction)
        return fn
    return deco


# ---- 对照组：现 SWING 的 POS 与现 SWING-2 的 MACD ----
@F("pos")
def _pos(df):
    c, h = df["close"], df["high"]
    ma20 = c.rolling(20).mean()
    parts = [c / h.rolling(10).max() - 1, c / h.rolling(20).max() - 1,
             c / ma20 - 1, c / c.rolling(5).mean() - 1, c / c.rolling(10).mean() - 1]
    return sum(pm(v) for v in parts) / len(parts)


@F("macd")
def _macd(df):
    c = df["close"]
    mh = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    return pm((mh - mh.ewm(span=9, adjust=False).mean()) / c)


# ---- A 波动率族 ----
@F("atr14")
def _f_atr(df):
    return pm(_atr(df, 14) / df["close"])


@F("bbw")
def _f_bbw(df):
    c = df["close"]
    return pm(4 * c.rolling(20).std() / c.rolling(20).mean())


@F("vol20")
def _f_vol20(df):
    return pm(df["close"].pct_change().rolling(20).std())


@F("vol_ratio")
def _f_volr(df):
    """波动放大：近 5 日波动 / 近 20 日波动。"""
    r = df["close"].pct_change()
    return pm(r.rolling(5).std() / r.rolling(20).std().replace(0, np.nan))


@F("vol_chg")
def _f_volc(df):
    """波动率的 5 日变化率（负 = 波动在压缩）。"""
    r = df["close"].pct_change()
    v = r.rolling(10).std()
    return pm(v / v.shift(5) - 1)


@F("rng20")
def _f_rng(df):
    return pm(((df["high"] - df["low"]) / df["close"]).rolling(20).mean())


# ---- B 量能族 ----
@F("amt_ratio")
def _f_amtr(df):
    """缩量程度：成交额 / 其 20 日均值。"""
    a = df["amount"]
    return pm(a / a.rolling(20).mean())


@F("amt_pct")
def _f_amtp(df):
    """成交额 250 日因果分位。"""
    return pm(df["amount"].rolling(250, min_periods=120).rank(pct=True) * 100)


@F("vp_corr")
def _f_vpc(df):
    """20 日量价相关（负 = 量价背离，价跌量增）。"""
    r = df["close"].pct_change()
    v = df["amount"].pct_change()
    return pm(r.rolling(20).corr(v))


@F("shr")
def _f_shr(df):
    """量能衰减：成交额 5 日均 / 20 日均。"""
    a = df["amount"]
    return pm(a.rolling(5).mean() / a.rolling(20).mean())


# ---- C 尾部 / 分布族 ----
@F("dwn20")
def _f_dwn(df):
    """近 20 日下跌天数占比。"""
    return pm((df["close"].pct_change() < 0).rolling(20).mean())


@F("bigdown")
def _f_bd(df):
    """近 10 日单日跌幅 > 2% 的天数。"""
    return pm((df["close"].pct_change() < -0.02).rolling(10).sum())


@F("dnvol")
def _f_dnv(df):
    """下行波动占比：负收益平方和 / 全部收益平方和。"""
    r = df["close"].pct_change()
    dn = (r.clip(upper=0) ** 2).rolling(20).sum()
    tot = (r ** 2).rolling(20).sum().replace(0, np.nan)
    return pm(dn / tot)


@F("skew20")
def _f_skew(df):
    return pm(df["close"].pct_change().rolling(20).skew())


@F("kurt20")
def _f_kurt(df):
    return pm(df["close"].pct_change().rolling(20).kurt())


@F("maxdd20")
def _f_mdd(df):
    """距 20 日最高点的回撤（同 vshi20，独立再验一次）。"""
    return pm(df["close"] / df["high"].rolling(20).max() - 1)


# ---- D 均线族 ----
@F("ma_disp")
def _f_mad(df):
    """均线收敛度：(max−min)/mean of MA5/10/20。低 = 收敛。"""
    c = df["close"]
    m = pd.concat([c.rolling(5).mean(), c.rolling(10).mean(), c.rolling(20).mean()], axis=1)
    return pm((m.max(axis=1) - m.min(axis=1)) / m.mean(axis=1))


@F("ma20_slope")
def _f_mas(df):
    m = df["close"].rolling(20).mean()
    return pm(m / m.shift(5) - 1)


@F("px_ma60")
def _f_p60(df):
    return pm(df["close"] / df["close"].rolling(60).mean() - 1)


@F("px_ma120")
def _f_p120(df):
    return pm(df["close"] / df["close"].rolling(120).mean() - 1)


@F("px_ma250")
def _f_p250(df):
    return pm(df["close"] / df["close"].rolling(250).mean() - 1)


# ---- E 动量族变体 ----
@F("tsi")
def _f_tsi(df):
    """TSI：双重平滑的动量，比 RSI 稳。"""
    d = df["close"].diff()
    s = d.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    a = d.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    return pm(s / a.replace(0, np.nan) * 100)


@F("roc10")
def _f_roc10(df):
    return pm(df["close"] / df["close"].shift(10) - 1)


@F("roc5")
def _f_roc5(df):
    return pm(df["close"] / df["close"].shift(5) - 1)


@F("ppo")
def _f_ppo(df):
    c = df["close"]
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    return pm((e12 - e26) / e26)


@F("macd_acc")
def _f_macc(df):
    """MACD 柱的 3 日变化（负 = 柱在往下走）。"""
    c = df["close"]
    mh = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    h = (mh - mh.ewm(span=9, adjust=False).mean()) / c
    return pm(h - h.shift(3))


@F("aroon")
def _f_aroon(df):
    """阿隆下行：近 25 日创新低的天数占比。"""
    low = df["low"].rolling(25).min()
    return pm((df["low"] <= low).rolling(25).mean())


# ---- F 时间 / 路径族 ----
@F("consec_dn")
def _f_cdn(df):
    """连续下跌天数。"""
    r = (df["close"].pct_change() < 0).astype(float)
    grp = (r != r.shift()).cumsum()
    return pm(r * r.groupby(grp).cumsum())


@F("days_hi20")
def _f_dh(df):
    """距 20 日新高的天数。"""
    c = df["close"]
    idx = np.arange(len(c))
    hi = pd.Series(np.where(c.to_numpy() >= c.rolling(20).max().to_numpy(), idx, np.nan),
                   index=c.index).ffill()
    return pm(pd.Series(idx, index=c.index) - hi)


# ---- G 日内族 ----
@F("clpos")
def _f_clp(df):
    """收盘在日内区间的位置（0 = 收在最低）。"""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return pm((df["close"] - df["low"]) / rng)


@F("upsh")
def _f_ush(df):
    """上影线占比。"""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return pm((df["high"] - df[["open", "close"]].max(axis=1)) / rng)


@F("gap")
def _f_gap(df):
    """向下跳空幅度。"""
    return pm(df["open"] / df["close"].shift(1) - 1)


@F("oc")
def _f_oc(df):
    """日内涨跌 open→close。"""
    return pm(df["close"] / df["open"] - 1)


# ---------------------------------------------------------------- 评估
class Evaluator:
    def __init__(self, h: int = H, tol: float = TOL):
        self.h, self.tol = h, tol
        self.D, self.CL, self.FW = {}, {}, {}
        for b in BOARDS:
            d = load(b)
            self.D[b] = d
            self.CL[b] = d["close"].astype(float)
            self.FW[b] = fwd(self.CL[b], h)

    def base(self, w0, w1) -> float:
        o = []
        for b in BOARDS:
            e, m = self.FW[b]
            idx = self.CL[b].index
            for i in range(len(idx)):
                if not (w0 <= idx[i] <= w1) or i + self.h > len(e) - 1:
                    continue
                o.append(e[i] > 0 and m[i] >= -self.tol)
        return 100 * float(np.mean(o)) if o else float("nan")

    def run(self, S: dict[str, pd.Series], thr: float, cool: int,
            w0, w1, h: int | None = None, cond=None) -> list[tuple]:
        """cond(df, i, s) -> bool：可选的止跌确认层（None = 不过滤）。"""
        h = h or self.h
        out = []
        for b in BOARDS:
            s = S[b].to_numpy(float)
            c = self.CL[b]
            idx = c.index
            e, m = self.FW[b] if h == self.h else fwd(c, h)
            df = self.D[b]
            last = -10 ** 9
            for i in range(1, len(s)):
                if np.isnan(s[i]) or s[i] > thr or i - last < cool:
                    continue
                if not (w0 <= idx[i] <= w1) or i + h > len(e) - 1:
                    continue
                if cond is not None and not cond(df, i, s):
                    continue
                last = i
                out.append((b, str(idx[i].date()), bool(e[i] > 0 and m[i] >= -self.tol),
                            float(e[i])))
        return out

    def score(self, name: str) -> dict[str, pd.Series]:
        """因子函数**自己**返回 0~100 分值（内部已做因果锚映射），此处不再二次映射。

        ⚠️ 这里曾经错误地对因子结果又做了一次 pm()，等于「分位的分位」，
        需要再 500 天预热 → 2013-2016 全变 NaN，直接看不到 2015/2016 那一关。
        direction=-1 时取补（100−v），与 pm(−raw) 在对称锚点下等价。
        """
        fn, d = FACTORS[name]
        out = {}
        for b in BOARDS:
            v = fn(self.D[b])
            out[b] = (100.0 - v) if d < 0 else v
        return out

    def evaluate(self, name: str, thr: float, cool: int = 4,
                 windows=("ALL", "57", "3")) -> dict:
        S = self.score(name)
        W = {"ALL": W_ALL, "57": W_57, "3": W_3}
        res = {"name": name, "thr": thr}
        for tag in windows:
            w0, w1 = W[tag]
            ev = self.run(S, thr, cool, w0, w1)
            k = sum(1 for r in ev if r[2])
            res[tag] = {"n": len(ev), "k": k,
                        "rate": 100 * k / len(ev) if ev else float("nan"),
                        "wilson": wilson(k, len(ev)), "base": self.base(w0, w1),
                        "ev": ev}
        return res

    def by_year(self, ev, y0="2013", y1="2026") -> dict:
        out = {}
        for r in ev:
            y = r[1][:4]
            if not (y0 <= y <= y1):
                continue
            out.setdefault(y, [0, 0])
            out[y][0] += 1
            out[y][1] += 1 if r[2] else 0
        return out


def fmt_row(name: str, thr, r: dict, keys=("ALL", "57", "3")) -> str:
    s = f"{name:<10}{thr:>5.0f}"
    for k in keys:
        v = r[k]
        s += (f"{v['n']:>6}{v['rate']:>8.1f}%{v['wilson']:>7.1f}"
              f"{v['base']:>7.1f}%")
    return s


def header(keys=("ALL", "57", "3")) -> str:
    s = f"{'因子':<10}{'阈值':>5}"
    for k in keys:
        lab = {"ALL": "全程13年", "57": "近5.7年", "3": "近3年"}[k]
        s += f"{lab:>6}{'命中':>9}{'下界':>7}{'基线':>8}"
    return s


def scan(names=None, thr=12.0, cool=4, verbose=True):
    E = Evaluator()
    if verbose:
        print(header())
        print("-" * len(header()))
    rows = {}
    for nm in (names or list(FACTORS)):
        r = E.evaluate(nm, thr, cool)
        rows[nm] = r
        if verbose:
            print(fmt_row(nm, thr, r))
    return E, rows


if __name__ == "__main__":
    args = sys.argv[1:]
    E, rows = scan(args or None)
    print("\n按「全程13年命中率」排序")
    for nm, r in sorted(rows.items(), key=lambda kv: -kv[1]["ALL"]["rate"]):
        v = r["ALL"]
        print(f"  {nm:<10} n={v['n']:<4} {v['rate']:5.1f}%  下界 {v['wilson']:5.1f}%"
              f"  近3年 {r['3']['n']:<4} {r['3']['rate']:5.1f}%")
