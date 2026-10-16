"""板块卡「建议持有几层仓」：状态识别 + 冻结映射（纯函数，无 I/O）。

层数语义：「如果今天才开仓，开几层」的开仓强度 —— 绝对值，不是目标仓位、
不是增减量；封顶 MAX_LAYERS（永不喊满仓）；没有新建议 ≠ 卖掉。

映射表由 _explore/s39.py（状态桶胜率）与 _explore/s40.py（当日指标数值区分度 +
盈亏比）的校准输出冻结：「点估计落档 且 Wilson 下界 > 同板块基线，子桶 n<8 退父桶」，
第二轮起加「下界处期望收益 > 0」的盈亏比闸门（DELIVERY.md §21）。
改映射必须附 s39/s40 重跑证据。

文案禁区（AGENTS.md §5）：禁「减仓 / 卖出 / 买入」；SENTI-1 >100（贪婪区）
不产生任何建议（顶部信号已证伪，见 store.py 文件头）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import swing, swing2, swing3

MAX_LAYERS = 8

# ---- 冻结映射（2026-10-16 定稿；第二轮按 s40 盈亏比证据修订，见 DELIVERY.md §21.8）----
# 分档规则：点估计落档（≥85→8 / 75~85→7 / 65~75→5 / 55~65→3 / 其余→0）
# 且 事件级/连续段级 Wilson 下界 > 同板块基线（否则降一档）；子桶 n<8 退父桶。
# 第二轮起加盈亏比闸门：Wilson 下界处的期望收益 ≤0 的桶不给层
# （期望 = p·均盈 − (1−p)·均亏，Kelly 口径在 s40 输出上手工核算）。
# 基线：小机会 T+7 随机日 48.3%；大机会 T+20 随机日 41.5%。
#
# 小机会（T+7 ok；信号级 / 事件级）：
#   S 级提示日  81.1% n=37 / 84.2% n=19（LB 62.4）        → 7
#   A 级提示日  87.5% n=48 / 91.4% n=35（LB 77.6）        → 8
#   B 级提示日  60.0% n=60 / 54.3% n=46（LB 40.2 ≤ 基线）→ 0（降档）
#   C 级提示日  50.9% n=53 / 50.0% n=40 ≈ 基线            → 0
#   B/C 深破（相对跌破≥50%）63.2% n=57：下界 50.2 处期望 −0.31% → 0（s40，盈亏比闸门）
#   「几个模型同时在线下」「跌破深度」在小机会侧无单调区分度（s40 A/B 段），不入规则
#   持有 d1_3（源事件 S/A）日级 73.2% n=194（段级 LB 65.4）→ 5
#   持有 d4_7：nmod0 55.0% n=100 下界不过基线；nmod≥1 78.6% n=14 但均亏 10.1%、
#             下界处期望 −2.46% → 0（s40 修订：持有只留前 3 天）
#   持有层数 = min(持有档, 源事件档)：B/C 级提示后不延续
SWING_MAP = {
    ("event", "S"): 7, ("event", "A"): 8, ("event", "B"): 0, ("event", "C"): 0,
    ("hold", "d1_3"): 5, ("hold", "d4_7"): 0,
}
# 大机会（T+20 ok / T+60 win；窗口内等级样本稀少，A/C 档引用已登记证据）：
#   A 级提示日  窗口 n=5 不足；长历史已登记 T+60：生产 8/8 + 8 宽基 18/18
#               （store.py 文件头 / DELIVERY.md §5），父桶 82.4% n=17      → 8
#   B 级提示日  91.7% n=12（LB 64.6），T+60 win 90.0% n=10                 → 8
#   C 级提示日  窗口 0 样本（DELIVERY.md §6.5：近年破净率高、估值层全过）；
#               「C 级置信度回落到历史平均」（DELIVERY.md §5）              → 0
#   持有 d1_10  日级 76.5% n=170（段级 70.6% n=17，LB 46.9 > 基线 41.5）   → 7
#   持有 d11_20 日级 70.6% n=170（段级 82.4% n=17，LB 59.0）               → 5
#   持有层数 = min(持有档, 源事件档)，再按当日分值调制（s40 C 段）：
#     0 < score ≤ 40 → 维持（持有期内 T+20 76~89%、T+60 93~100%、盈亏比 5~13）
#     score > 40     → 3（T+60 回落 54.6%≈基线：温度已回暖，不再开新仓强度）
#     score ≤ 0      → 3（跌回恐慌区，视同 panic 档）
#   panic（score≤0 且近 20 日无事件）：T+20 85.7% n=21（LB 65.4）、
#     T+60 89.5% 盈亏比 7.2；下界处 Kelly≈64%，对折（无 2015 样本折价）→ 3
#   edge（0<score≤20 无事件）T+20 38.7~45.4% ≤ 基线 → 0（触发线 0 的边界有信息量）
SENTI_MAP = {
    ("event", "A"): 8, ("event", "B"): 8, ("event", "C"): 0,
    ("hold", "d1_10"): 7, ("hold", "d11_20"): 5,
    ("panic", ""): 3,
}
SENTI_HOLD_WARM = 40.0      # 持有期内当日分值 > 40 → 降档（s40 C 段）


def swing_grade(ures: int | None) -> str:
    """并集事件等级：当日有信号的板块数 ≥5/4/2 → S/A/B，否则 C（与前端 gradeOf 同口径）。"""
    u = ures or 1
    return "S" if u >= 5 else ("A" if u >= 4 else ("B" if u >= 2 else "C"))


def senti_grade(capit: bool | None, resonant: bool | None) -> str | None:
    """SENTI-1 事件等级（capit/resonant 由调用方算好；数据缺失传 None）。"""
    from . import store                       # 延迟 import：store 也 import 本模块
    return store.grade({"kind": "bottom", "capit": capit, "resonant": resonant})


def scan_fires(s: np.ndarray, kind: str) -> list[int]:
    """逐日模拟扫描触发下标（含冷却），不截断末尾、不算收益。

    入场规则与各模型 events() 完全一致（全链路唯一的扫描实现，盘中也走这里）：
      swing / swing2: s[i] <= THR
      swing3:         s[i] <= THR 且 s[i] > s[i-1]（回升确认）
      senti:          s[i-1] <= ENTRY_THR 且 s[i] > s[i-1]（昨日恐慌 + 今日掉头）
    """
    from . import store
    mod = {"swing": swing, "swing2": swing2, "swing3": swing3}.get(kind)
    thr = mod.THR if mod else None
    cool = store.COOL_TRADING if kind == "senti" else mod.COOL
    fires: list[int] = []
    last = -10 ** 9
    for i in range(1, len(s)):
        if np.isnan(s[i]):
            continue
        prev = s[i - 1]
        if kind == "senti":
            ok = not np.isnan(prev) and prev <= store.ENTRY_THR and s[i] > prev
        elif kind == "swing3":
            ok = not np.isnan(prev) and s[i] <= thr and s[i] > prev
        else:
            ok = s[i] <= thr
        if ok and i - last >= cool:
            last = i
            fires.append(i)
    return fires


# ---- 状态构造（生产 / 盘中 / s39 三处共用，防口径漂移）----

def swing_st(days: int | None, grade: str | None) -> dict:
    """小机会状态：days = 距末次并集事件的交易日数（0=事件当日），grade = 该事件等级。"""
    if days == 0:
        return {"state": "event", "grade": grade or "C", "days": 0}
    if days is not None and 1 <= days <= swing.H:
        return {"state": "hold", "seg": "d1_3" if days <= 3 else "d4_7",
                "days": days, "grade": grade}
    return {"state": "none"}


def senti_st(days: int | None, grade: str | None) -> dict:
    """大机会状态：days = 距末次底部事件的交易日数（0=事件当日）。"""
    from . import store
    if days == 0:
        return {"state": "event", "grade": grade or "C", "days": 0}
    if days is not None and 1 <= days <= store.H:
        return {"state": "hold", "seg": "d1_10" if days <= 10 else "d11_20",
                "days": days, "grade": grade}
    return {"state": "none"}


# ---- 生产口径：最新一天的状态 ----

def swing_state(uev: list[dict], dates: pd.Index) -> dict:
    """从并集事件列表推导最新一天的小机会状态（union_events 含最近日，无需扫描）。"""
    if not len(uev):
        return {"state": "none"}
    e = uev[-1]
    pos_of = {str(d.date()): i for i, d in enumerate(dates)}
    i_ev = pos_of.get(e["date"])
    if i_ev is None:
        return {"state": "none"}
    return swing_st(len(dates) - 1 - i_ev, swing_grade(e.get("ures")))


def _senti_grade_at(p: pd.DataFrame, i: int,
                    reso: pd.Series | None, bna: pd.Series | None) -> str | None:
    from . import store
    r0 = None
    if reso is not None:
        rv = reso.reindex(p.index).to_numpy(dtype=float)
        if not np.isnan(rv[i]):
            r0 = int(rv[i])
    b0 = None
    if bna is not None and len(bna):
        bv = bna.reindex(p.index, method="ffill").to_numpy(dtype=float)
        if not np.isnan(bv[i]):
            b0 = float(bv[i])
    return senti_grade(None if b0 is None else b0 >= store.BNA_PCT_MIN,
                       None if r0 is None else r0 >= store.RESONANCE_MIN)


def senti_state(p: pd.DataFrame, reso: pd.Series | None, bna: pd.Series | None) -> dict:
    """SENTI-1 最新一天状态。

    store.events() 只扫到 n-H（store.py:192，最近 20 天没有事件记录），
    这里用 scan_fires 按同一入场规则自行扫描，覆盖到最新一天。
    状态里带当日分值（hold 调制与 panic 判定要用，s40 口径）。
    """
    from . import store
    s = p["score"].to_numpy(dtype=float)
    if len(s) < 2:
        return {"state": "none"}
    today = float(s[-1]) if not np.isnan(s[-1]) else None
    fires = scan_fires(s, "senti")
    if fires:
        days = len(s) - 1 - fires[-1]
        if days <= store.H:
            st = senti_st(days, _senti_grade_at(p, fires[-1], reso, bna))
            st["score"] = today
            return st
    if today is not None and today <= store.ENTRY_THR:
        return {"state": "panic", "score": today}
    return {"state": "none"}


# ---- 状态 → 建议 ----
# 持有期层数 = min(持有档, 源事件档)：低等级提示（swing B/C、senti C）后不延续仓位。
# 依据（s39 分组）：swing BC 源事件后 d1_3 58.2% / d4_7 43.2% ≈ 基线，无延续价值。
# senti 持有期再按当日分值调制（s40）：>40 或 ≤0 都降到 panic 档 3 层。

def swing_advice(st: dict) -> dict:
    if st["state"] == "event":
        g = st["grade"]
        layers = SWING_MAP[("event", g)]
        if layers == 0:
            return {"layers": 0, "label": "观望", "why": f"{g} 级提示历史胜率≈基线"}
        return {"layers": layers, "label": f"{g} 级提示日", "why": ""}
    if st["state"] == "hold":
        g = st.get("grade") or "C"
        layers = min(SWING_MAP[("hold", st["seg"])], SWING_MAP[("event", g)])
        if layers == 0:
            why = ("低等级提示不延续" if SWING_MAP[("event", g)] == 0
                   else "持有第 4~7 天盈亏比≈0（s40）")
            return {"layers": 0, "label": "观望", "why": why}
        return {"layers": layers, "label": f"提示后第 {st['days']} 天",
                "why": f"{swing.H} 天持有期内"}
    return {"layers": 0, "label": "观望", "why": ""}


def senti_advice(st: dict) -> dict:
    from . import store
    if st["state"] == "event":
        g = st["grade"]
        layers = SENTI_MAP[("event", g)]
        if layers == 0:
            return {"layers": 0, "label": "观望", "why": f"{g} 级置信度≈历史平均"}
        return {"layers": layers, "label": f"{g} 级提示日", "why": ""}
    if st["state"] == "hold":
        g = st.get("grade") or "C"
        base = min(SENTI_MAP[("hold", st["seg"])], SENTI_MAP[("event", g)])
        if base == 0:
            return {"layers": 0, "label": "观望", "why": "低等级提示不延续"}
        sc = st.get("score")
        if sc is not None and sc > SENTI_HOLD_WARM:
            return {"layers": SENTI_MAP[("panic", "")], "label": f"提示后第 {st['days']} 天",
                    "why": "温度已回暖 >40，强度降档"}
        if sc is not None and sc <= store.ENTRY_THR:
            return {"layers": SENTI_MAP[("panic", "")], "label": f"提示后第 {st['days']} 天",
                    "why": "温度跌回恐慌区，按恐慌档"}
        return {"layers": base, "label": f"提示后第 {st['days']} 天",
                "why": f"{store.H} 天持有期内"}
    if st["state"] == "panic":
        return {"layers": SENTI_MAP[("panic", "")], "label": "恐慌区",
                "why": "分值≤0 未出提示"}
    return {"layers": 0, "label": "观望", "why": ""}
