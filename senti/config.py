"""全局配置：板块定义、路径、模型参数。"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 上游可复用数据（v1 / v2 项目的缓存，只读） ----
UPSTREAM = {
    "v1_stocks": r"D:\情绪指标\data\cache\stocks",
    "v2_stocks": r"D:\情绪指标2\data\cache\stocks",
    "v2_index": r"D:\情绪指标2\data\cache\index",
    "v2_universe": r"D:\情绪指标2\data\cache\universe",
}

# ---- 本项目自身目录 ----
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# ---- 板块定义（6 个，顺序即页面顺序） ----
# key: 板块标识; index_file: v2 指数缓存文件名; uni: 成分股清单文件名
BOARDS = {
    "SH":     {"name": "大盘",     "index": "SH_INDEX", "uni": "SH_INDEX", "desc": "上证指数 000001.SH"},
    "STAR":   {"name": "科创板",   "index": "STAR",     "uni": "STAR",     "desc": "科创50 000688.SH"},
    "CHINEXT": {"name": "创业板",  "index": "CHINEXT",  "uni": "CHINEXT",  "desc": "创业板指 399006.SZ"},
    "CSI1000": {"name": "中证1000", "index": "CSI1000", "uni": "CSI1000",  "desc": "中证1000 000852.SH"},
    "CSI2000": {"name": "中证2000", "index": "CSI2000", "uni": "CSI2000",  "desc": "中证2000 932000.CSI"},
    "HS300":  {"name": "沪深300",  "index": "HS300",    "uni": "HS300",    "desc": "沪深300 000300.SH"},
}
BOARD_ORDER = ["SH", "STAR", "CHINEXT", "CSI2000", "CSI1000", "HS300"]

# ---- 回测 / 展示窗口 ----
BACKTEST_START = "2023-09-20"   # 近 3 年（当前 2026-09-20）

# ---- 历史起点：锚点窗口需要预热，实际数据要比展示窗口更早开始 ----
# 展示仍从 BACKTEST_START 起；更早的段落只用于计算锚点，不出现在图上。
HIST_START = "2019-01-01"

# ---- 模型参数（SENTI-1，单一模型，全板块统一） ----
MODEL = {
    # 因子绝对刻度锚点：本板块自身分位 → 0 / 100
    "anchor_lo": 2.0,
    "anchor_hi": 98.0,
    "map_clip_lo": -25.0,   # 单因子映射截断
    "map_clip_hi": 125.0,
    # 锚点窗口（交易日）：t 时刻的刻度只由过去这么多天的数据决定。
    # 必须因果，否则新增数据会改写已发布的历史分值。
    "anchor_window": 750,
    "anchor_min": 500,      # 历史不足这么多天时不给出分值
}

# ---- 因子权重（等权为默认，校准后可调整；和为 1） ----
WEIGHTS = {
    "b20":   0.14,   # 站上 MA20 占比
    "b60":   0.12,   # 站上 MA60 占比
    "r5":    0.10,   # 近 5 日上涨占比
    "nh":    0.12,   # 60 日净新高（新高占比 − 新低占比）
    "lim":   0.10,   # 近 5 日涨停净占比
    "amt":   0.10,   # 成交额 250 日分位
    "rsi":   0.10,   # 指数 RSI14
    "bias":  0.10,   # 指数乖离率 close/MA60 − 1
    "ret20": 0.06,   # 指数 20 日收益
    "vol":   0.06,   # 指数 20 日波动率（反向计入）
}
# 反向因子：值越大越恐慌
INVERT = {"vol": True}

WEB_PORT = 8779
