"""SENTI-1 —— 单一情绪模型（恐贪指数）。

设计要点（全部只用当日及历史数据，无未来函数）
------------------------------------------------
1) 两套子口径，各自 0~100 绝对刻度（按本板块自身历史分位锚定，而非滚动 z）
   L  广度情绪 = Σ w·占优度(b20,b60,r5,nh,lim,rsi,bias,ret20)
   T  价格过热 = 0.30·bias + 0.30·ret20 + 0.22·rsi + 0.18·成交额分位
   U = 0.5·L + 0.5·T

2) 底部确认项 cf_b = 超跌程度 × 放量程度
   —— 实测：恐慌放量型的底（2024-02、2025-04）命中率显著高于单纯超跌

3) score = U − K_B·cf_b
   溢出区（<0 / >100）为绝对刻度自然溢出，不是人为阈值

为什么不用滚动 z-score：滚动 z 测的是「相对近期的意外程度」，
阴跌磨底时永远不极端，实测极值日与真实拐点完全错相位（见 _explore/d3）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data, factors

# ---- 冻结的超参数 ----
L_WEIGHTS = {"b20": .18, "b60": .15, "r5": .11, "nh": .13,
             "lim": .09, "rsi": .14, "bias": .12, "ret20": .08}
T_WEIGHTS = {"bias": .30, "ret20": .30, "rsi": .22, "amt": .18}
W_L = 0.5            # U = W_L·L + (1−W_L)·T
K_B = 25.0           # 底部确认项 A（恐慌放量型）强度
CF_AMT_LO = 0.60     # 放量确认：成交额分位起点
CF_L_HI = 12.0       # 超跌确认：L 低于此值才计入
CLIP_LO, CLIP_HI = -30.0, 140.0

# ---- 底部确认项 B：磨底型 —— 已于 2026-09-20 判定无效，强度置 0 ----
# 原设计：距 250 日高点深度回撤 + 个股走势高度同步（截面离散度低）时额外下修分值，
#         用来抓 2024-09 那种「不暴跌、慢慢磨」的底。
#
# 为什么作废：换成因果锚之后，实测 cf_g 与**失败**绑定而不是成功。
#   生产 22 次信号里，失败组 cf_g 均值 0.333、命中组 0.098（失败组是命中组的 3.4 倍）。
#   放到长历史（8 宽基 × 2013~2026，扩展因果锚）验证同样成立：
#     K_G = 40 → 56.5%  30 → 54.5%  20 → 56.8%  10 → 65.7%  **0 → 77.1%**
#     最难区段 2015-2016：30 → 23.1%  **0 → 55.6%**
#   K_G ≤ 0 结果完全一致（平台，不是尖峰），说明这一项不该压低分值。
# 语义结论：**「磨底」不是底**。深度回撤 + 走势高度同步，意味着市场还在慢慢往下走，
#   并没有出现恐慌放量式的投降。真正有效的是确认项 A（cf_b）。
# cf_g 仍照常计算并保留在面板里，便于观察，但不参与分值。
K_G = 0.0            # 磨底确认强度（置 0 = 不参与分值）
GR_DD_A, GR_DD_W = 90.0, 12.0     # 回撤分位斜坡：90→102 线性 0→1（保留，仅供观察）
GR_DP_A, GR_DP_W = 75.0, 12.0     # 同步性分位斜坡：75→87 线性 0→1（保留，仅供观察）


ANCHOR_WIN = int(C.MODEL.get("anchor_window", 750))   # 锚点窗口（交易日）
ANCHOR_MIN = int(C.MODEL.get("anchor_min", 500))      # 历史不足则不给出分值


def _pct_map(v: pd.Series, lo_q: float, hi_q: float, lo_clip: float, hi_clip: float) -> pd.Series:
    """把因子按「因果锚点」锚定到 0~100 绝对刻度。

    锚点 = 过去 ANCHOR_WIN 个交易日的 p2/p98，并整体 shift(1)，
    保证 t 时刻的刻度只由 t 之前的数据决定。

    为什么必须因果：旧实现用「整个面板」的分位数当锚点，等于用 2026 年的数据
    去给 2023 年的分值定刻度 —— 这是前视函数。后果有两个：
      1) 每天新增数据都会重算并改写整条历史曲线（实测最大改写 2.7 分），
         昨天显示的 −10 今天可能变成 −3，指标没法跟踪；
      2) 回测里调出来的阈值（如 −6）依赖未来的数据，实盘不可复现。
    改成因果锚后，已发布的历史分值永不改变。
    """
    x = v.astype(float)
    lo = x.rolling(ANCHOR_WIN, min_periods=ANCHOR_MIN).quantile(lo_q / 100.0).shift(1)
    hi = x.rolling(ANCHOR_WIN, min_periods=ANCHOR_MIN).quantile(hi_q / 100.0).shift(1)
    d = (hi - lo).replace(0.0, np.nan)
    return (100.0 * (x - lo) / d).clip(lo_clip, hi_clip)


def build(board_key: str, stock_ind: pd.DataFrame | None = None) -> pd.DataFrame:
    """返回该板块完整面板：date, close, 各因子, L, T, U, cf_b, score。"""
    if stock_ind is None:
        stock_ind = data.build_stock_indicators()

    # 注意：必须在**全历史**上计算，锚点窗口需要预热段；
    # 展示窗口的截断放在最后，否则锚点会退化成样本内分位（前视）。
    raw = factors.build_board_raw(board_key, stock_ind)

    idx = data.load_index(board_key).set_index("date")
    for col in ("open", "high", "low"):
        raw[col] = idx[col].reindex(raw.index)

    lo_q, hi_q = C.MODEL["anchor_lo"], C.MODEL["anchor_hi"]
    lc, hc = C.MODEL["map_clip_lo"], C.MODEL["map_clip_hi"]

    # ---- L：广度情绪 ----
    num = pd.Series(0.0, index=raw.index)
    den = 0.0
    for k, w in L_WEIGHTS.items():
        num = num + _pct_map(raw[k], lo_q, hi_q, lc, hc) * w
        den += w
    raw["L"] = num / den

    # ---- T：价格过热 ----
    raw["T"] = (_pct_map(raw["bias"], lo_q, hi_q, lc, hc) * T_WEIGHTS["bias"]
                + _pct_map(raw["ret20"], lo_q, hi_q, lc, hc) * T_WEIGHTS["ret20"]
                + _pct_map(raw["rsi"], lo_q, hi_q, lc, hc) * T_WEIGHTS["rsi"]
                + raw["amt_pct"] * 100.0 * T_WEIGHTS["amt"])

    raw["U"] = W_L * raw["L"] + (1 - W_L) * raw["T"]

    # ---- 底部确认 A：超跌 × 放量（恐慌放量型，如 2024-02、2025-04）----
    amt_q = raw["amt_pct"].astype(float)
    cf_amt = ((amt_q - CF_AMT_LO) / (1.0 - CF_AMT_LO)).clip(0, 1).fillna(0.0)
    cf_low = ((CF_L_HI - raw["L"]) / CF_L_HI).clip(0, 1).fillna(0.0)
    raw["cf_b"] = (cf_amt * cf_low).fillna(0.0)

    # ---- 底部确认 B：深度回撤 × 高同步（磨底型，如 2024-09）----
    raw["dd250"] = raw["close"] / raw["close"].rolling(250, min_periods=60).max() - 1.0
    f_disp = (100.0 - _pct_map(raw["disp"], lo_q, hi_q, lc, hc)).fillna(0.0)   # 同步性（越高越同步）
    f_dd = (100.0 - _pct_map(raw["dd250"], lo_q, hi_q, lc, hc)).fillna(0.0)    # 回撤深度（越高越深）
    g1 = ((f_dd - GR_DD_A) / GR_DD_W).clip(0, 1)
    g2 = ((f_disp - GR_DP_A) / GR_DP_W).clip(0, 1)
    raw["cf_g"] = (g1 * g2).fillna(0.0)

    raw["score"] = (raw["U"] - K_B * raw["cf_b"] - K_G * raw["cf_g"]).clip(CLIP_LO, CLIP_HI)
    return raw[raw.index >= pd.Timestamp(C.BACKTEST_START)].copy()


def build_all(stock_ind: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    if stock_ind is None:
        stock_ind = data.build_stock_indicators()
    return {b: build(b, stock_ind) for b in C.BOARD_ORDER}
