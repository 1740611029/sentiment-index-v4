"""SWING-3 —— 小波段「均线拐点」指标（第四个小波段模型，与 SWING / SWING-2 并存）。

与另外两个小波段模型的分工
--------------------------
SWING   抓**回调底**：价格位置 POS 与热度 HEAT 同时跌到极端分位（max，AND 语义）。
SWING-2 抓**动量拐点**：MACD 柱 / 收盘价 跌到自身历史极低分位。
SWING-3 抓**均线拐点**：MA20 的 5 日斜率跌到历史极低分位（均线跌得最急）
        **且当日斜率回升**（跌势趋缓）—— 是「跌速的拐点」，不是「价格的拐点」。

三者判定口径完全一致（T+7 期末 > 0 且期间最深收盘回撤 >= −3%），
但信号几乎不重叠（6 板块 · 近 5.7 年）：
    J(SWING-3, SWING)   = 0.028
    J(SWING-3, SWING-2) = 0.000   ← 与 SWING-2 完全不重叠
    J(SWING,   SWING-2) = 0.154

分值构造
--------
    SWING-3 = _pct_map( MA20 / MA20.shift(5) − 1 )      0~100，越低 = 均线跌得越急
    MA20    = close.rolling(20).mean()

因果锚复用 SENTI-1 的 `model._pct_map`（rolling(750, min_periods=500).quantile().shift(1)），
不复制逻辑。

为什么是「斜率 + 回升确认」这两件套
----------------------------------
1. **斜率**是价量之外的正交信息：它与 SWING 的价格位置（J=0.028）、
   SWING-2 的 MACD 柱（J=0.000）都几乎不重合，所以并集是**净增量**。
2. **回升确认**（分值 > 前一日）是本模型的关键。实测（_explore/s12/s15，9 宽基全程）：
       ma20_slope ≤12 无确认  →  423 次 / 59.1%（2015 年 33%）
       ma20_slope ≤12 + 回升  →  238 次 / 66.8%（2015 年 38%）
   回升确认过滤掉的正是「加速下跌途中」——那是 2015-06 之后连续误判的来源。
   ⚠️ 回升确认是**实时可判定**的（今天 vs 昨天），不是事后挑点（铁律二）。

为什么阈值取 10、冷却取 4
------------------------
全局平台，不是针对某一年调的（_explore/s16，9 宽基全程）：
    阈值  5→65.9%  8→65.5%  **10→66.0%**  12→66.8%  15→62.9%  18→59.4%
    冷却  2→64.8%  3→63.6%  **4→66.8%**   5→63.3%   7→64.2%  10→61.1%
阈值 8~12、冷却 3~5 都在平台内。取 10 / 4 是 6 板块页面口径下并集命中率最高的组合
（≤10 时并集 65.4%，≤12 时 64.7%），且冷却与另两个模型一致，便于比较。

实测（逐日模拟，无未来函数）
----------------------------------
页面口径（6 板块 · 展示窗口 2023-09-20 ~ 2026-09-18）：
| 口径                    | 信号数 | 命中率 | Wilson 下界 | 基线 |
| 现 SWING ≤22            |  64    | 64.1%  | 51.8        | 48.6% |
| 现 SWING-2 ≤12          |  99    | 68.7%  | 59.0        | 48.6% |
| **SWING-3 ≤10 + 回升**  | **47** | **85.1%** | **72.3** | 48.6% |
| 并集（前两个，原默认）    | 143    | 66.4%  | 58.4        | 48.6% |
| **并集（三个，现默认）**  | **189**| **70.9%** | —        | 48.6% |
→ 加入 SWING-3 后**信号 +46 次（+32%）、命中 +4.5pp**，两个维度同时上升。

长历史（9 宽基，剔除中证2000；_explore/s14）：
| 口径                    | 近 5.7 年 2021-01 起 | 全程 2013-2026 |
| 现 SWING ≤22            |  90 / 61.1%          | 159 / 53.5% |
| 现 SWING-2 ≤12          | 114 / 70.2%          | 199 / 57.8% |
| **SWING-3 ≤10 + 回升**  | **72 / 76.4%**（下界65.4）| **115 / 65.2%**（下界56.1）|
| 并集（前两个）           | 172 / 65.1%          | 298 / 56.4% |
| **并集（三个）**         | **242 / 68.2%**      | **409 / 58.9%** |
基线 47.4% / 50.9%。三个窗口方向一致，不是靠某一段撑起来的。

共振分档（近 5.7 年）：全部 72/76.4%、共振≥3 55/81.8%、共振≥4 37/**91.9%**、
共振≥5 24/91.7%。与 SWING 的 S 级一样，共振档是「少而准」，不能当成全部信号的能力。

已知边界（必须知道）
--------------------
1. **2015 年仍然不过关**：6 板块 6/15 = 40%（基线 50.5%）。比现 SWING 的 36%、
   SWING-2 的 33% 略好，但仍是「超卖抄底」范式的通病 —— 加速下跌中途的反弹
   常常撑不过 7 天。回升确认只能**减轻**、不能**消除**。
2. **2019 年偏弱**：4/13 = 30.8%（2019-05 贸易战、2019-08 汇率破 7 的阴跌）。
3. 2016 年 4/5、2021 年 4/4、2024 年 5/5 样本过小，不单独解读。
4. 样本量不大（展示窗口 47 次），Wilson 下界 72.3%，**不宣称稳定能力**。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config as C
from . import data
from .model import _pct_map          # 复用 SENTI-1 的因果锚（不修改其逻辑）

_LO, _HI = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
_LC, _HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]


def _pm(v: pd.Series) -> pd.Series:
    """SENTI-1 的因果锚映射，带上本模块统一的锚点参数。"""
    return _pct_map(v, _LO, _HI, _LC, _HC)


# ---- 冻结参数 ----
MA_WIN = 20         # 均线周期
SLOPE_LB = 5        # 斜率回看（交易日）
THR = 10.0          # 分值阈值：<= 此值且回升才触发
COOL = 4            # 冷却（交易日），与 SWING / SWING-2 同
H = 7               # 判定持有期
TOL = 0.03          # 期间最深回撤容差
RESO_MIN = 3        # 共振标签：含自己在内至少几个板块同时触发
RESO_A = 4
RESO_S = 5

PANEL_DIR = os.path.join(C.DATA_DIR, "panels_swing3")


def build(board_key: str) -> pd.DataFrame:
    """返回该板块面板：close, ms(MA20 斜率原始值), swing3(0~100 因果锚分位)。

    必须在**全历史**上算：因果锚要 rolling(750) 预热（铁律四），
    若先截断到展示窗口，锚点会退化成样本内分位（前视）。
    """
    idx = data.load_index(board_key).set_index("date")
    close = idx["close"].astype(float)

    ma = close.rolling(MA_WIN).mean()
    slope = ma / ma.shift(SLOPE_LB) - 1.0

    out = pd.DataFrame(index=idx.index)
    out["close"] = close
    out["ms"] = slope                          # 原始值，供排查
    out["swing3"] = _pm(slope)                 # 0~100，越低 = 均线跌得越急
    return out[out.index >= pd.Timestamp(C.BACKTEST_START)].copy()


def build_all() -> dict[str, pd.DataFrame]:
    return {b: build(b) for b in C.BOARD_ORDER}


def build_and_cache(force: bool = False) -> dict[str, pd.DataFrame]:
    os.makedirs(PANEL_DIR, exist_ok=True)
    if not force and all(os.path.exists(os.path.join(PANEL_DIR, f"{b}.parquet"))
                         for b in C.BOARD_ORDER):
        return load()
    panels = build_all()
    for b, p in panels.items():
        p.to_parquet(os.path.join(PANEL_DIR, f"{b}.parquet"))
    return panels


def load() -> dict[str, pd.DataFrame]:
    return {b: pd.read_parquet(os.path.join(PANEL_DIR, f"{b}.parquet"))
            for b in C.BOARD_ORDER}


def triggers(p: pd.DataFrame) -> np.ndarray:
    """当日是否**触发**：分值 <= THR 且分值较前一日回升。

    注意与 SWING / SWING-2 的口径差别：那两个模型没有额外条件，
    「分值 <= THR」就等于「触发」；SWING-3 多了回升确认，
    所以共振必须按本函数算，否则共振数会被高估（_explore/s14）。
    """
    s = p["swing3"].to_numpy(dtype=float)
    n = len(s)
    t = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if np.isnan(s[i]) or np.isnan(s[i - 1]):
            continue
        t[i] = (s[i] <= THR) and (s[i] > s[i - 1])
    return t


def resonance(panels: dict[str, pd.DataFrame]) -> pd.Series:
    """每个交易日有多少个板块**触发**了 SWING-3（按各板块自己的日期轴）。"""
    T = pd.DataFrame({b: pd.Series(triggers(p), index=p.index)
                      for b, p in panels.items()}).sort_index()
    return T.fillna(False).sum(axis=1)


def events(p: pd.DataFrame, reso: pd.Series | None = None) -> list[dict]:
    """逐日模拟：t 日收盘触发 → t 日收盘买入，冷却 COOL 个交易日。

    只用当日及历史数据。最后 H 天的信号没有未来数据可判定，
    照常输出但 ret/risk 为 None，页面显示「待验证」。
    """
    s = p["swing3"].to_numpy(dtype=float)
    c = p["close"].to_numpy(dtype=float)
    dates = p.index
    n = len(s)
    rv = reso.reindex(dates).to_numpy(dtype=float) if reso is not None else None
    out = []
    last = -10 ** 9
    for i in range(1, n):
        if np.isnan(s[i]) or s[i] > THR or i - last < COOL:
            continue
        if np.isnan(s[i - 1]) or not (s[i] > s[i - 1]):      # 回升确认
            continue
        last = i
        c0 = c[i]
        rec = {
            "date": str(dates[i].date()),
            "score": round(float(s[i]), 1),
            "src": "maslope",
            "reso": int(rv[i]) if (rv is not None and not np.isnan(rv[i])) else None,
        }
        if i + H <= n - 1:
            seg = c[i + 1:i + H + 1]
            ret = float(seg[-1] / c0 - 1)
            mdd = float(seg.min() / c0 - 1)
            rec.update({
                "ret": round(ret * 100, 2),
                "risk": round(mdd * 100, 2),
                "ok": bool(ret > 0 and mdd >= -TOL),
                "notrap": bool(mdd >= -TOL),
            })
        else:
            rec.update({"ret": None, "risk": None, "ok": None, "notrap": None})
        out.append(rec)
    return out


def baseline(p: pd.DataFrame) -> float:
    """随机基线：随便挑一天买入，满足判定口径的比例。"""
    c = p["close"].to_numpy(dtype=float)
    n = len(c)
    ok = []
    for i in range(n - H):
        seg = c[i + 1:i + H + 1]
        ok.append(seg[-1] / c[i] - 1 > 0 and seg.min() / c[i] - 1 >= -TOL)
    return round(float(np.mean(ok)) * 100, 1) if ok else 0.0


def summary(panels: dict[str, pd.DataFrame]) -> dict:
    reso = resonance(panels)
    per, tot = {}, {"n": 0, "ok": 0, "nt": 0, "pend": 0,
                    "res_n": 0, "res_ok": 0, "res4_n": 0, "res4_ok": 0,
                    "res5_n": 0, "res5_ok": 0}
    for b, p in panels.items():
        ev = events(p, reso)
        done = [e for e in ev if e["ok"] is not None]
        pend = [e for e in ev if e["ok"] is None]
        ok = sum(1 for e in done if e["ok"])
        nt = sum(1 for e in done if e["notrap"])
        res = [e for e in done if (e["reso"] or 0) >= RESO_MIN]
        res4 = [e for e in done if (e["reso"] or 0) >= RESO_A]
        res5 = [e for e in done if (e["reso"] or 0) >= RESO_S]
        per[b] = {"n": len(done), "ok": ok, "nt": nt, "pend": len(pend),
                  "rate": round(ok / len(done) * 100, 1) if done else None,
                  "base": baseline(p)}
        tot["n"] += len(done); tot["ok"] += ok; tot["nt"] += nt; tot["pend"] += len(pend)
        tot["res_n"] += len(res); tot["res_ok"] += sum(1 for e in res if e["ok"])
        tot["res4_n"] += len(res4); tot["res4_ok"] += sum(1 for e in res4 if e["ok"])
        tot["res5_n"] += len(res5); tot["res5_ok"] += sum(1 for e in res5 if e["ok"])
    tot["rate"] = round(tot["ok"] / tot["n"] * 100, 1) if tot["n"] else None
    tot["base"] = round(float(np.mean([v["base"] for v in per.values()])), 1)
    return {"total": tot, "per": per, "thr": THR, "cool": COOL,
            "hold": H, "tol": int(TOL * 100),
            "ma_win": MA_WIN, "slope_lb": SLOPE_LB,
            "reso_min": RESO_MIN, "reso_a": RESO_A, "reso_s": RESO_S}
