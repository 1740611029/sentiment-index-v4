"""面板缓存 + 事件抽取（供 Web 层读取）。"""
from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd

from . import config as C
from . import model, swing

PANEL_DIR = os.path.join(C.DATA_DIR, "panels")
META_PATH = os.path.join(C.DATA_DIR, "meta.json")

H = 20           # 主口径持有期
TOL = 0.03       # 风险容差：底部最深回撤 / 顶部最大踏空
GAP = 20         # 事件冷却期（自然日，旧口径，仅 baseline 之外已不用）

# ---- 实时入场规则（重点：只用当日及历史数据，实盘可执行）----
# 底部：昨日分值 ≤ ENTRY_THR 且今日分值高于昨日 → 今日收盘买入
#       即「极度恐慌之后，情绪掉头」。
#
# ENTRY_THR = 0 就是需求原话「低于 0 是底部」，不是调出来的数。
#   为什么不再用 −6：−6 是在「旧刻度」上调出来的。旧刻度用整个面板的分位做锚点，
#   属于前视函数（见 model._pct_map），换成因果锚之后刻度变了，−6 不再有特殊含义。
#   实测新刻度下阈值 5 / 0 / −3 / −6 / −9 的命中率是 70.4 / 77.3 / 75.0 / 72.2 / 75.0%，
#   基本是平的，−6 并不比 0 好；取 0 既符合需求语义，样本也更多。
# 为什么不用「取每段最深那天」：那天要等整段走完才知道，是事后视角，实盘买不到。
ENTRY_THR = 0.0
COOL_TRADING = 20    # 入场后 20 个交易日内不再重复入场

# ---- 横截面共振（信号的可选确认层，不改变分值本身）----
# 定义：信号日收盘后，全市场还有多少个板块的分值仍在 ≤ ENTRY_THR。
#   reso 高 = 别的板块也还在恐慌 = 系统性投降
#   reso 低 = 只有自己在恐慌     = 局部流动性危机
# 实测（_explore/d67、d68）：
#   reso >= 2 的信号，T+60 盈利 26/26（生产 8/8 + 长历史 8 宽基 18/18），
#   同板块随机买入基线只有 52.6%，二项尾概率 ≈ 1.4e-7；
#   长历史分年度 2015 年 4/4、2016 年 3/3、2020 年 3/3（对照全部信号 2015 年只有 4/12）。
# 代价：只保留约 1/4 的信号（生产 33 → 8，长历史 77 → 18）。
# 所以它**不做成过滤器**，而是做成每个信号上的标签，由使用者自行决定是否只做共振信号。
RESONANCE_MIN = 2
H60 = 60             # 共振统计用的长持有期


def resonance(panels: dict) -> pd.Series:
    """每个交易日，有多少个板块的分值 ≤ ENTRY_THR。"""
    S = pd.DataFrame({b: p["score"] for b, p in panels.items()}).sort_index()
    return (S <= ENTRY_THR).sum(axis=1)


# ---- 确认层 B：估值侧投降（价量之外的新信息源）----
# 破净股占比的 250 日分位。破净率高 = 市场给的价格已经低于净资产 = 估值侧投降。
#
# 为什么需要它：现有模型全是价量，而价量在「流动性危机」和「真恐慌底」上长得几乎一样
# （2024-01-23 中证2000 浮亏 −23.61% 就栽在这里）。估值侧不会说谎。
#
# 实测（_explore/d69~d71，三个样本集，都过了 2015-2016）：
#   bna_pct >= 60 的信号，T+60 盈利
#     交付 6 板块 24/28 = 85.7%   长历史 8 宽基·trail 53/60 = 88.3%
#     长历史 8 宽基·扩展锚 29/29 = 100%（含 2015 年 6/6、2016 年 6/6）
#     合并 53/57；同板块随机买入基线只有 52.9%，p < 1e-6。
#   分年度 T+60：2015 6/6、2016 6/6、2020 4/4、2022 5/5、2024 1/1、2025 7/7。
#     **2015 年从无过滤的 6/12 变成 6/6 —— 这是唯一能救 2015 年的条件。**
#   邻域扰动：门槛 40 / 45 / 50 / 55 / 60 结果完全一致（平台，不是尖峰）。
#   代价：只丢约 15% 的信号（生产 33 → 28），比共振层便宜得多。
#
# 注意：这是**市场级**数据，同一天所有板块读数相同。
#   它做的是「全局择时」（现在是不是全市场级别的投降），
#   而 reso 做的是「横截面」（是不是这个板块自己的问题）。两者互补。
BNA_PCT_MIN = 60.0


def bna_series() -> pd.Series | None:
    from . import data as D
    return D.load_bna_pct()


# ---- 顶部信号已移除（2026-09-22）----
# 原「首破 100」顶部信号实测 2/9 = 22.2%，随机基线 40.1%，优势 −17.9pp，
# p = 0.93；95% 置信区间 [6.3%, 54.7%] 整个跨过基线 —— 统计上与瞎猜无法区分。
# 长历史口径同样不成立（T+60 下跌 15/29 = 51.7%，基线 47.5%，p = 0.39）。
# 唯一能救它的「融资余额分位 ≥70」在近 3 年一次都没触发过（分位全落在 10~18）。
#   → 顶部不是样本偏差，是逻辑不成立，因此整条链路（事件 / 统计 / 页面）全部删除。
# 分值本身仍然照画：>100 只是过热读数的刻度区间，不再产生任何信号。


def build_and_cache(force: bool = False) -> dict[str, pd.DataFrame]:
    os.makedirs(PANEL_DIR, exist_ok=True)
    if not force and all(os.path.exists(os.path.join(PANEL_DIR, f"{b}.parquet")) for b in C.BOARD_ORDER):
        return load()
    panels = model.build_all()
    for b, p in panels.items():
        p.to_parquet(os.path.join(PANEL_DIR, f"{b}.parquet"))
    meta = {
        "built_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_date": str(max(p.index.max() for p in panels.values()).date()),
        "start": C.BACKTEST_START,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return panels


def load() -> dict[str, pd.DataFrame]:
    return {b: pd.read_parquet(os.path.join(PANEL_DIR, f"{b}.parquet")) for b in C.BOARD_ORDER}


def meta() -> dict:
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"built_at": "-", "last_date": "-", "start": C.BACKTEST_START}


def _fwd(close: pd.Series, d, h: int = H):
    i = close.index.get_loc(d)
    if i + h >= len(close):
        return None
    c0 = close.iloc[i]
    seg = close.iloc[i + 1: i + h + 1]
    return (float(seg.iloc[-1] / c0 - 1), float(seg.min() / c0 - 1), float(seg.max() / c0 - 1))


def grade(e: dict) -> str | None:
    """把确认层合成一个置信等级，让使用者不必自己逐列判断。

    底部（两个确认层各管一件事）：
      A = 估值已投降 ✔ 且 全场共振 ✔ —— 市场级时机对、横截面也不是局部问题
      B = 只满足其中一个
      C = 两个都不满足

    实测（T+60 盈利）：
      底部 A 级：生产 8/8；长历史 8 宽基 18/18（含 2015 年 4/4、2016 年 3/3、2020 年 3/3）
      底部 B 级：生产 16/20；长历史 trail 35/42

    数据缺失时返回 None（页面显示「—」，不误导）。
    """
    if e.get("kind") == "bottom":
        cap = e.get("capit"); res = e.get("resonant")
        if cap is None and res is None:
            return None
        cap = bool(cap); res = bool(res)
        if cap and res:
            return "A"
        if cap or res:
            return "B"
        return "C"
    return None


def events(p: pd.DataFrame, reso: pd.Series | None = None,
           bna: pd.Series | None = None) -> list[dict]:
    """底部信号（逐日模拟，只用当日及历史数据，无未来函数）。

    昨日分值 ≤ ENTRY_THR 且今日分值 > 昨日  → 今日收盘买入
    入场后 COOL_TRADING 个交易日内不再重复入场。

    reso 传入时（由 resonance(panels) 得到），会额外给底部信号打上
    「当日还有几个板块也在 ≤0」的横截面共振标签 reso / resonant。

    顶部信号已于 2026-09-22 移除（实测无效，见文件顶部说明），本函数只返回底部。
    """
    s = p["score"].to_numpy(dtype=float)
    c = p["close"].to_numpy(dtype=float)
    dates = p.index
    n = len(s)
    rv = reso.reindex(dates).to_numpy(dtype=float) if reso is not None else None
    bv = (bna.reindex(dates, method="ffill").to_numpy(dtype=float)
          if bna is not None else None)
    bot = []
    last_b = -10 ** 9
    for i in range(1, n - H):
        if np.isnan(s[i]) or np.isnan(s[i - 1]):
            continue
        c0 = c[i]
        seg = c[i + 1:i + H + 1]
        ret = float(seg[-1] / c0 - 1)
        mdd = float(seg.min() / c0 - 1)
        if i - last_b >= COOL_TRADING and s[i - 1] <= ENTRY_THR and s[i] > s[i - 1]:
            last_b = i
            r60 = round(float(c[i + H60] / c0 - 1) * 100, 2) if i + H60 <= n - 1 else None
            r0 = int(rv[i]) if (rv is not None and not np.isnan(rv[i])) else None
            b0 = float(bv[i]) if (bv is not None and not np.isnan(bv[i])) else None
            bot.append({
                "bna": round(b0, 1) if b0 is not None else None,
                "capit": bool(b0 is not None and b0 >= BNA_PCT_MIN),
                "date": str(dates[i].date()),
                "prev": round(float(s[i - 1]), 1),
                "score": round(float(s[i]), 1),
                "ret": round(ret * 100, 2),
                "ret60": r60,
                "risk": round(mdd * 100, 2),
                "ok": bool(ret > 0 and mdd >= -TOL),
                "notrap": bool(mdd >= -TOL),
                "reso": r0,
                "resonant": bool(r0 is not None and r0 >= RESONANCE_MIN),
                "kind": "bottom",
            })
            bot[-1]["grade"] = grade(bot[-1])
    return bot


def baseline(p: pd.DataFrame) -> float:
    """随机基线：同板块随便挑一天买入，满足底部判定口径的比例。"""
    close = p["close"]
    ok = []
    for i in range(len(close) - H):
        c0 = close.iloc[i]
        seg = close.iloc[i + 1: i + H + 1]
        ok.append(seg.iloc[-1] / c0 - 1 > 0 and seg.min() / c0 - 1 >= -TOL)
    return round(float(np.mean(ok)) * 100, 1) if ok else 0.0


_swing_cache: dict = {}


def swing_panels() -> dict[str, pd.DataFrame]:
    """小波段面板（带进程内缓存，避免首页每次请求都读一遍 parquet）。"""
    if not _swing_cache:
        _swing_cache.update(swing.build_and_cache())
    return _swing_cache


def summary(panels: dict[str, pd.DataFrame]) -> dict:
    tot = {"b_n": 0, "b_ok": 0, "b_nt": 0,
           "r_n": 0, "r_ok": 0, "r_nt": 0, "r_win60": 0, "r_n60": 0,
           "c_n": 0, "c_ok": 0, "c_nt": 0, "c_win60": 0, "c_n60": 0,
           "rc_n": 0, "rc_win60": 0, "rc_n60": 0,
           **{f"gb{g}_{k}": 0 for g in "ABC" for k in ("n", "ok", "n60", "win60")}}
    per = {}
    reso = resonance(panels)
    bna = bna_series()
    for b, p in panels.items():
        bot = events(p, reso, bna)
        bok = sum(1 for e in bot if e["ok"])
        bnt = sum(1 for e in bot if e["notrap"])
        sub = [e for e in bot if e["resonant"]]
        sub60 = [e for e in sub if e["ret60"] is not None]
        cap = [e for e in bot if e["capit"]]
        cap60 = [e for e in cap if e["ret60"] is not None]
        both = [e for e in bot if e["capit"] and e["resonant"]]
        both60 = [e for e in both if e["ret60"] is not None]
        tot["b_n"] += len(bot); tot["b_ok"] += bok; tot["b_nt"] += bnt
        tot["r_n"] += len(sub); tot["r_ok"] += sum(1 for e in sub if e["ok"])
        tot["r_nt"] += sum(1 for e in sub if e["notrap"])
        tot["r_n60"] += len(sub60); tot["r_win60"] += sum(1 for e in sub60 if e["ret60"] > 0)
        tot["c_n"] += len(cap); tot["c_ok"] += sum(1 for e in cap if e["ok"])
        tot["c_nt"] += sum(1 for e in cap if e["notrap"])
        tot["c_n60"] += len(cap60); tot["c_win60"] += sum(1 for e in cap60 if e["ret60"] > 0)
        tot["rc_n"] += len(both)
        tot["rc_n60"] += len(both60)
        tot["rc_win60"] += sum(1 for e in both60 if e["ret60"] > 0)
        # 置信等级统计
        for g in ("A", "B", "C"):
            gb = [e for e in bot if e["grade"] == g]
            gb60 = [e for e in gb if e["ret60"] is not None]
            tot[f"gb{g}_n"] += len(gb)
            tot[f"gb{g}_ok"] += sum(1 for e in gb if e["ok"])
            tot[f"gb{g}_n60"] += len(gb60)
            tot[f"gb{g}_win60"] += sum(1 for e in gb60 if e["ret60"] > 0)
        per[b] = {
            "bottom": {"n": len(bot), "ok": bok, "nt": bnt},
            "reso": {"n": len(sub), "ok": sum(1 for e in sub if e["ok"]),
                     "nt": sum(1 for e in sub if e["notrap"]),
                     "n60": len(sub60), "win60": sum(1 for e in sub60 if e["ret60"] > 0)},
            "capit": {"n": len(cap), "ok": sum(1 for e in cap if e["ok"]),
                      "nt": sum(1 for e in cap if e["notrap"]),
                      "n60": len(cap60), "win60": sum(1 for e in cap60 if e["ret60"] > 0)},
            "base_bottom": baseline(p),
        }
    return {"total": tot, "per": per, "entry_thr": ENTRY_THR,
            "reso_min": RESONANCE_MIN, "bna_min": BNA_PCT_MIN,
            "bna_latest": (None if bna is None or len(bna) == 0
                           else round(float(bna.dropna().iloc[-1]), 1)),
            "bna_date": (None if bna is None or len(bna) == 0
                         else str(bna.dropna().index[-1].date())),
            "swing": swing.summary(swing_panels())}


def to_json(panels: dict[str, pd.DataFrame]) -> dict:
    out = {}
    reso = resonance(panels)
    bna = bna_series()
    sp = swing_panels()
    sreso = swing.resonance(sp)
    for b, p in panels.items():
        bot = events(p, reso, bna)
        q = sp[b]
        out[b] = {
            "name": C.BOARDS[b]["name"],
            "desc": C.BOARDS[b]["desc"],
            "dates": [str(d.date()) for d in p.index],
            "score": [round(float(x), 2) for x in p["score"]],
            "close": [round(float(x), 2) for x in p["close"]],
            "L": [round(float(x), 1) for x in p["L"]],
            "T": [round(float(x), 1) for x in p["T"]],
            "bottom": bot,
            # ---- 小波段 SWING ----
            "swing": [round(float(x), 2) for x in q["swing"]],
            "swing_pos": [round(float(x), 1) for x in q["pos"]],
            "swing_heat": [round(float(x), 1) for x in q["heat"]],
            "swing_events": swing.events(q, sreso),
        }
    return out
