"""d94 —— 约束驱动：先诊断目标日，再按"必须覆盖目标日"筛选候选。

d93 暴露的核心矛盾：
  命中率 82~92% 的方案全是 reso≥3，只在 2022/2024 的系统性恐慌期出信号，
  每板块每年 <1.5 次，而且 2026-08-03 / 09-14 一次都没触发。
  但用户点名的就是这两个点 —— 所以共振过滤这条路必须放弃。

做法：
  1) 先打印科创板在 2026-08-03 / 09-14 两天的**全部因子分位**，看清"这两天到底是什么状态"；
  2) 再把"必须覆盖这两天"作为硬约束去筛候选，在满足约束的里面挑命中率最高的。
"""
from __future__ import annotations
import os, sys, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C
from d81 import short_factors, norm
from d84 import groups_mean
from d86 import build as build_breadth, board_breadth
from d89 import z_factors, EX_IDX
from d92 import prepare, hits_for, sim
from d80 import build_full

W0 = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2026-09-18")
T1, T2 = "2026-08-03", "2026-09-14"

if __name__ == "__main__":
    panels = build_full()
    wide = build_breadth()
    D = prepare(panels, wide)
    names = D[C.BOARD_ORDER[0]]["names"]
    cols = D["STAR"]["cols"]
    print("=== 科创板在目标日的因子分位（0~100，越低越超卖）===")
    row1 = cols.loc[T1]; row2 = cols.loc[T2]
    ref = cols.loc[W0:].median()
    df = pd.DataFrame({"08-03": row1, "09-14": row2, "全样本中位": ref})
    df["两天最大"] = df[["08-03", "09-14"]].max(axis=1)
    print(df.sort_values("两天最大").round(1).to_string())
    print("\n（「两天最大」越小 = 该因子在两天都很低，适合做 AND 条件）")
