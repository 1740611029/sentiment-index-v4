"""SWING-2 —— 小波段「动量拐点」指标（第三个模型，与 SWING 并存）。

与 SWING 的分工
--------------
SWING   抓的是**回调底**：价格位置 + 热度同时跌到极端分位（POS / HEAT 双条件 AND）。
SWING-2 抓的是**动量拐点**：MACD 柱跌到自身历史极低分位。

两者判定口径完全一致（T+7 期末 > 0 且期间最深收盘回撤 >= −3%），
但信号**几乎不重叠**：近 5.7 年 Jaccard 只有 0.12（shared 26 / 各自独有 106 与 90）。
所以页面上做成「回调底 / 动量拐点 / 合并」三选，默认合并 —— 合并后信号数接近翻倍。

分值构造
--------
    SWING-2 = _pct_map( macd_hist / close )        0~100，越低 = 动量越超卖
    macd_hist = (EMA12 − EMA26) − EMA9(EMA12 − EMA26)

因果锚复用 SENTI-1 的 `model._pct_map`（rolling(750, min_periods=500).quantile().shift(1)），
不复制逻辑。

为什么除以 close 而不是 ATR
---------------------------
实测（_explore/s7）：
  /close → 5.7 年 n=132 命中 69.7%
  /ATR14 → 5.7 年 n=163 命中 53.0%   ← 归一化的尺度敏感，ATR 归一化会毁掉这个因子

为什么是单因子，不做"技术指标组合"
----------------------------------
把 MACD 与价格位置做加权平均 / AND，在 (信号数, 命中率) 二维上都不如单因子：
  AVG(PX, macd) <= 12  → 106 / 70.0%（信号更少）
  max(PX, macd)  <= 30 → 164 / 65.0%（命中更低）
堆因子会抹平极值 —— 与 SWING 用 max 而非平均是同一个教训（_explore/d91）。

参数怎么定的（全局平台，不是针对某一年调出来的）
------------------------------------------------
  EMA 8/17/9 → 70%；**12/26/9 → 70%**；19/39/9 → 65%；5/35/5 → 63%
  阈值 10 → 207/63.3%；**12 → 222/64.4%**；15 → 245/59.6%（并集口径）
  冷却 2 → 213/67%；**4 → 132/70%**；7 → 99/68%（单模型口径）
  持有期 h=5 → 66.7%；**h=7 → 69.7%**；h=10 → 64.4%

实测（逐日模拟，无未来函数）
----------------------------
| 口径                 | 5.7 年(2021-01起)      | 3 年(展示窗口)         |
| 现 SWING <=22        | 116 / 59.5% (下界50.4) | 64 / 64.1% (下界51.8) |
| **SWING-2 <=12**     | **132 / 69.7%** (61.4) | **99 / 68.7%** (59.0) |
| **并集（去重）**      | **222 / 64.4%** (57.9) | **143 / 66.4%** (58.4)|
基线 47.5% / 48.6%。

已知边界（必须知道）
--------------------
1. **连环崩塌会连续误判**：长历史（2013-2026，9 宽基，剔除中证2000）2015 年 25/72 = 34.7%、
   2016 年 1/16 = 6.2%，远低于基线。这是「超卖抄底」整个范式的通病，不是本模型独有 ——
   同期现 SWING 的价格部分 POS<=15 是 33% / 33%，一样崩（_explore/s6）。
   与 SENTI-1 的已知边界「放到 13 年长历史里会明显变差」是同一件事。
2. **2026 年偏弱**：15/29 = 52%（并集 19/37 = 51%），而 2022/2024/2025 在 65~83%。
   **没有针对年份调参**，只如实披露。
3. 科创板依然最差（14/28 = 50%）：振幅决定"7 天回撤不超 3%"最难满足。
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
THR = 12.0          # 分值阈值：<= 此值触发信号
COOL = 4            # 冷却（交易日），与 SWING 同
H = 7               # 判定持有期
TOL = 0.03          # 期间最深回撤容差
RESO_MIN = 3        # 共振标签：含自己在内至少几个板块同时触发
RESO_A = 4
RESO_S = 5

PANEL_DIR = os.path.join(C.DATA_DIR, "panels_swing2")


def build(board_key: str) -> pd.DataFrame:
    """返回该板块面板：close, mh(MACD柱/收盘价), swing2(0~100 因果锚分位)。

    必须在**全历史**上算：因果锚要 rolling(750) 预热（铁律四），
    若先截断到展示窗口，锚点会退化成样本内分位（前视）。
    """
    idx = data.load_index(board_key).set_index("date")
    close = idx["close"].astype(float)

    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    mh = e12 - e26
    hist = mh - mh.ewm(span=9, adjust=False).mean()

    out = pd.DataFrame(index=idx.index)
    out["close"] = close
    out["mh"] = hist / close                 # 原始值，供排查
    out["swing2"] = _pm(out["mh"])           # 0~100，越低越超卖
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


def resonance(panels: dict[str, pd.DataFrame]) -> pd.Series:
    """每个交易日有多少个板块的 SWING-2 <= 阈值。"""
    S = pd.DataFrame({b: p["swing2"] for b, p in panels.items()}).sort_index()
    return (S <= THR).sum(axis=1)


def events(p: pd.DataFrame, reso: pd.Series | None = None) -> list[dict]:
    """逐日模拟：t 日收盘触发 → t 日收盘买入，冷却 COOL 个交易日。

    只用当日及历史数据。最后 H 天的信号没有未来数据可判定，
    照常输出但 ret/risk 为 None，页面显示「待验证」。
    """
    s = p["swing2"].to_numpy(dtype=float)
    c = p["close"].to_numpy(dtype=float)
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
            "src": "macd",
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
            "reso_min": RESO_MIN, "reso_a": RESO_A, "reso_s": RESO_S}
