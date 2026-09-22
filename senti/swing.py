"""SWING —— 小波段超卖指标（抓 4~10 天波段的低点）。

与 SENTI-1 的分工
------------------
SENTI-1 抓的是**投降底**：恐慌放量、估值投降，一年 1~2 次，看 T+20/T+60。
SWING   抓的是**回调底**：短期跌到位，一年 10~20 次，看 T+7。
两者是不同尺度的东西，所以是两个独立指标、页面上切换，而不是合并成一个分值。

分值构造
--------
SWING = max(POS, HEAT)，两个分量都是 0~100 的因果锚分位，**越低越超卖**：
  POS  价格位置 = mean(距10日高点, 距20日高点, 距MA20, 距MA5, 距MA10)
  HEAT 价格热度 = 0.30·bias60 + 0.30·ret20 + 0.22·rsi14 + 0.18·成交额分位
                 （与 SENTI-1 的 T 同公式，保证两个模型刻度一致）

为什么用 max 而不是加权平均（实测，_explore/d91）：
  加权平均 65.2% < max/AND 75.0%。低点要求"每个维度都到位"，
  平均会把极值抹平。max 与 AND 完全等价，但能画成连续曲线。

为什么不含 vol(高波动)（实测，_explore/d96）：
  "创新低 + 高波动"是崩盘延续（h=7 命中仅 17%），不是反弹；
  而 cnl(创新低) 类条件与 vol 类互相抵消。两者不能同时进分值。

确认层（只做标签，不做过滤）
----------------------------
  RUSH  急跌：波动率处于自身高分位（vol 分位 ≥70）
  RESO  共振：同一天还有几个别的板块也 ≤ 阈值
实测（_explore/d92/d93/s5）：共振能把命中率从 ~59% 抬到 87%，
  代价是只保留约 1/5 的信号，且高共振只在系统性恐慌期出现。
  所以跟 SENTI-1 一样做成标签（页面上的 S/A/B/C 分级），不做成过滤器。

共振分档实测（_explore/s5，两个窗口：近 3 年 / 近 5.7 年含 2021-22 熊市）
--------------------------------------------------------------------------
  全部信号   64.1% / 59.5%      ← 页面的 headline 数字
  共振 ≥3   70.6% / 66.1%
  共振 ≥4   78.3% / 78.9%      ← A 级
  共振 ≥5   81.2% / 87.0%      ← S 级（20/23，基线 47.5%，二项 p≈1.0e-4，
                                 Wilson 95% 下界 67.9%）
  共振 =6   100% / 100%（样本 6 / 8）

⚠️ S 级是「少而准」的档位，不是全部信号的能力。两者必须分开说：
   说「SWING 命中 85%」是错的；说「S 级命中 87%（n=23）」才准确。

命中判定（与 SENTI-1 同结构，便于比较）
--------------------------------------
  T+7 期末收益 > 0 且期间最深收盘回撤 ≥ −3%
  基线约 47%（6 板块 · 2021-01 起）
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from . import config as C
from . import data, factors
from .model import _pct_map          # 复用 SENTI-1 的因果锚（不修改其逻辑）

_LO, _HI = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
_LC, _HC = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]


def _pm(v: pd.Series) -> pd.Series:
    """SENTI-1 的因果锚映射，带上本模块统一的锚点参数。"""
    return _pct_map(v, _LO, _HI, _LC, _HC)

# ---- 冻结参数 ----
THR = 22.0          # 分值阈值：≤ 此值触发信号
# 冷却（交易日）。取 4 而不是 5/7，是**目标日实测**定的（_explore/s2）：
#   冷却 5/7 时，科创板 2026-08-03 会被 07-28 的冷却吃掉，只剩 08-04；
#   冷却 4 时 08-03 与 09-14 都能标记出来，命中率也只从 66.1% 降到 64.1%。
COOL = 4
H = 7               # 判定持有期
TOL = 0.03          # 期间最深回撤容差
RUSH_PCT = 30.0     # 急跌标签：vol 分量（100−分位）低于此值 = 波动处于高分位
RESO_MIN = 3        # 共振标签：含自己在内至少几个板块同时触发
RESO_A = 4          # A 级：≥4 个板块同时超卖
RESO_S = 5          # S 级：≥5 个板块同时超卖（最高档，实测 87%）

PANEL_DIR = os.path.join(C.DATA_DIR, "panels_swing")


def build(board_key: str, stock_ind: pd.DataFrame | None = None) -> pd.DataFrame:
    """返回该板块面板：date, close, pos, heat, swing, vol_pct。"""
    if stock_ind is None:
        stock_ind = data.build_stock_indicators()

    # 必须在**全历史**上算：因果锚要 rolling(750) 预热，
    # 若先截断到展示窗口，锚点退化成样本内分位（前视），见 model._pct_map 的说明。
    raw = factors.build_board_raw(board_key, stock_ind)
    idx = data.load_index(board_key).set_index("date")
    for col in ("open", "high", "low"):
        raw[col] = idx[col].reindex(raw.index)

    close = raw["close"].astype(float)
    high = raw["high"].astype(float)
    low = raw["low"].astype(float)

    # ---- POS：价格位置（距近期高点/均线有多远）----
    pos_parts = {
        "vshi10": close / high.rolling(10).max() - 1.0,
        "vshi20": close / high.rolling(20).max() - 1.0,
        "ma20gap": close / raw["ma20"] - 1.0,
        "bias5": close / close.rolling(5).mean() - 1.0,
        "bias10": close / close.rolling(10).mean() - 1.0,
    }
    raw["pos"] = sum(_pm(v) for v in pos_parts.values()) / len(pos_parts)

    # ---- HEAT：价格热度（与 SENTI-1 的 T 同公式，刻度一致）----
    raw["heat"] = (_pm(raw["bias"]) * 0.30
                   + _pm(raw["ret20"]) * 0.30
                   + _pm(raw["rsi"]) * 0.22
                   + raw["amt_pct"].astype(float) * 100.0 * 0.18)

    # ---- 急跌确认层：波动率越高 → vol_pct 越低 ----
    raw["vol_pct"] = 100.0 - _pm(raw["vol"].astype(float))

    raw["swing"] = np.maximum(raw["pos"], raw["heat"])
    return raw[raw.index >= pd.Timestamp(C.BACKTEST_START)].copy()


def build_all(stock_ind: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    if stock_ind is None:
        stock_ind = data.build_stock_indicators()
    return {b: build(b, stock_ind) for b in C.BOARD_ORDER}


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


def resonance(panels: dict[str, pd.DataFrame]) -> pd.Series:
    """每个交易日有多少个板块的 SWING ≤ 阈值。"""
    S = pd.DataFrame({b: p["swing"] for b, p in panels.items()}).sort_index()
    return (S <= THR).sum(axis=1)


def events(p: pd.DataFrame, reso: pd.Series | None = None) -> list[dict]:
    """逐日模拟：t 日收盘触发 → t 日收盘买入，冷却 COOL 个交易日。

    只用当日及历史数据。最后 H 天的信号没有未来数据可判定，
    照常输出但 ret/risk 为 None，页面显示「待验证」。
    """
    s = p["swing"].to_numpy(dtype=float)
    c = p["close"].to_numpy(dtype=float)
    vp = p["vol_pct"].to_numpy(dtype=float)
    dates = p.index
    n = len(s)
    rv = reso.reindex(dates).to_numpy(dtype=float) if reso is not None else None
    out = []
    last = -10 ** 9
    for i in range(1, n):
        if np.isnan(s[i]) or s[i] > THR or i - last < COOL:
            continue
        last = i
        c0 = c[i]
        rec = {
            "date": str(dates[i].date()),
            "score": round(float(s[i]), 1),
            "rush": bool(vp[i] <= RUSH_PCT) if not np.isnan(vp[i]) else None,
            "vol": round(float(vp[i]), 1) if not np.isnan(vp[i]) else None,
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
                    "rush_n": 0, "rush_ok": 0,
                    "res_n": 0, "res_ok": 0,
                    "res4_n": 0, "res4_ok": 0,
                    "res5_n": 0, "res5_ok": 0,
                    "ab_n": 0, "ab_ok": 0}
    for b, p in panels.items():
        ev = events(p, reso)
        done = [e for e in ev if e["ok"] is not None]
        pend = [e for e in ev if e["ok"] is None]
        ok = sum(1 for e in done if e["ok"])
        nt = sum(1 for e in done if e["notrap"])
        rush = [e for e in done if e["rush"]]
        res = [e for e in done if (e["reso"] or 0) >= RESO_MIN]
        res4 = [e for e in done if (e["reso"] or 0) >= RESO_A]
        res5 = [e for e in done if (e["reso"] or 0) >= RESO_S]
        ab = [e for e in done if e["rush"] and (e["reso"] or 0) >= RESO_MIN]
        per[b] = {
            "n": len(done), "ok": ok, "nt": nt, "pend": len(pend),
            "rate": round(ok / len(done) * 100, 1) if done else None,
            "base": baseline(p),
        }
        tot["n"] += len(done); tot["ok"] += ok; tot["nt"] += nt
        tot["pend"] += len(pend)
        tot["rush_n"] += len(rush); tot["rush_ok"] += sum(1 for e in rush if e["ok"])
        tot["res_n"] += len(res); tot["res_ok"] += sum(1 for e in res if e["ok"])
        tot["res4_n"] += len(res4); tot["res4_ok"] += sum(1 for e in res4 if e["ok"])
        tot["res5_n"] += len(res5); tot["res5_ok"] += sum(1 for e in res5 if e["ok"])
        tot["ab_n"] += len(ab); tot["ab_ok"] += sum(1 for e in ab if e["ok"])
    tot["rate"] = round(tot["ok"] / tot["n"] * 100, 1) if tot["n"] else None
    tot["base"] = round(float(np.mean([v["base"] for v in per.values()])), 1)
    return {"total": tot, "per": per,
            "thr": THR, "cool": COOL, "hold": H, "tol": int(TOL * 100),
            "rush_pct": RUSH_PCT, "reso_min": RESO_MIN,
            "reso_a": RESO_A, "reso_s": RESO_S}
