"""d82 —— 判定口径网格扫描：找一个基线合理、语义贴合"4~10 天小波段低点"的口径。

基线太低（如 24%）的口径说明判定条件本身就是小概率事件，
即便信号很准也凑不出 85%，而且对低波动板块（大盘/沪深300）系统性不公平。
基线太高（>70%）的口径则没有区分度，命中率再高也没意义。
目标：基线落在 40%~60%，这样 85% 才是真的 edge。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from senti import config as C
from d80 import build_full, fwd_stats

W0 = pd.Timestamp("2023-09-20")


def main():
    panels = build_full()
    stats = {b: fwd_stats(panels[b]["close"]) for b in C.BOARD_ORDER}
    win = {b: s.loc[s.index >= W0] for b, s in stats.items()}

    cands = []
    for h in (5, 7, 10):
        # 需要按 h 重算
        st = {b: fwd_stats(panels[b]["close"], h).loc[lambda d: d.index >= W0]
              for b in C.BOARD_ORDER}
        g = pd.concat([s["gain"] for s in st.values()]).dropna()
        m = pd.concat([s["mdd"] for s in st.values()]).dropna()
        e = pd.concat([s["end"] for s in st.values()]).dropna()
        for gain in (0.02, 0.03, 0.04):
            for tol in (0.02, 0.03, 0.05):
                a = (g >= gain) & (m >= -tol)
                b = (e > 0) & (m >= -tol)
                cands.append({"h": h, "gain": gain, "tol": tol,
                              "抓波段": a.mean() * 100,
                              "期末正+不套": b.mean() * 100})
        cands.append({"h": h, "gain": np.nan, "tol": np.nan,
                      "抓波段": np.nan, "期末正+不套": np.nan})
        # 纯口径
        pure = pd.DataFrame([{
            "h": h, "gain": "—", "tol": "—",
            "抓波段": (g >= 0.03).mean() * 100,
            "期末正+不套": (e > 0).mean() * 100}])
        cands.append(pure.iloc[0].to_dict())

    df = pd.DataFrame(cands).dropna(subset=["h"])
    df["组合"] = df.apply(lambda r: f"涨≥{r['gain']:.0%} 且回撤≥−{r['tol']:.0%}"
                          if isinstance(r["gain"], float) and not np.isnan(r["gain"])
                          else "纯（只看涨/只看期末）", axis=1)
    print(f"基线（6 板块合并，n≈{len(pd.concat([s['gain'] for s in win.values()]).dropna())}）\n")
    print(df[["h", "组合", "抓波段", "期末正+不套"]].to_string(index=False,
          float_format=lambda x: f"{x:.1f}"))

    # 分板块：低波动板块是否被口径系统性歧视
    print("\n分板块（h=7）:")
    for b in C.BOARD_ORDER:
        s = fwd_stats(panels[b]["close"], 7).loc[lambda d: d.index >= W0].dropna()
        print(f"  {C.BOARDS[b]['name']:<9}"
              f" 涨≥3% {100*(s['gain']>=0.03).mean():5.1f}%"
              f"  期末>0 {100*(s['end']>0).mean():5.1f}%"
              f"  涨≥3%且回撤≥-3% {100*((s['gain']>=0.03)&(s['mdd']>=-0.03)).mean():5.1f}%")


if __name__ == "__main__":
    main()
