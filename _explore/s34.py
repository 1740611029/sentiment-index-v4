"""s34 —— 第二十六轮：外部信息源的收口（QVIX 决定性补测 + 其余源批量筛除）

背景
----
s33 显示 QVIX 过滤器在分半检验上崩了：
    前半 2015-2020 = 44.4%(268)  后半 2021-2026 = 96.8%(31)
前半反而**低于**基准 48.9%。这是典型的「近期制度依赖」。
但在写下否决结论之前，必须先排除一种可能：**前半样本里 QVIX 也有信息，
只是被别的因素盖住了**（例如 2015 年 QVIX 常年 >30，阈值失去了区分度）。
本脚本 A 段用「组内分位剂量反应」来正面回答：**在前半内部，QVIX 高低还有区分度吗？**

同时把 s31 探测到的其余可达外部源一次性筛完，避免下一轮重复：
    国债收益率（10Y / 期限利差）、北向资金净买、新成立基金份额、全A估值 PE 分位

口径
----
· 外部序列一律 **shift(1)**（北向当日盘后公布、基金份额按成立日、估值按月末），
  与项目既有约定一致（AGENTS.md §6）。
· 剂量反应用**组内四分位**（不是阈值），免疫阈值挑选。
· 每个候选都要过：三窗口不劣 + 分半检验 + 2015/2016。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s8
import s9                                        # noqa: F401
import s10                                       # noqa: F401
import s12                                       # noqa: F401
import s17
import s18
import s19
import s24
import s32
from s8 import wilson, BOARDS                    # noqa: E402

import akshare as ak                             # noqa: E402

CACHE_DIR = s8.C.CACHE_DIR

W_Q = (pd.Timestamp("2015-02-09"), pd.Timestamp("2026-09-18"))
W_57 = (pd.Timestamp("2021-01-01"), pd.Timestamp("2026-09-18"))
W_3 = (pd.Timestamp("2023-09-20"), pd.Timestamp("2026-09-18"))
H1 = (pd.Timestamp("2015-02-09"), pd.Timestamp("2020-12-31"))
H2 = (pd.Timestamp("2021-01-01"), pd.Timestamp("2026-09-18"))


# ------------------------------------------------------------------ 工具
def st(ev, w0, w1):
    sub = [r for r in ev if w0 <= pd.Timestamp(r[1]) <= w1]
    n = len(sub)
    k = sum(1 for r in sub if r[2])
    return n, k, (100 * k / n if n else float("nan")), wilson(k, n)


def yr(ev, y):
    sub = [r for r in ev if r[1][:4] == y]
    return f"{sum(1 for r in sub if r[2])}/{len(sub)}" if sub else "—"


def mk_series(E, raw: pd.Series) -> dict:
    """市场级外部序列 → 各板块日期轴对齐（reindex + ffill），并 shift(1) 防前视。"""
    r = raw.dropna().sort_index().shift(1)
    return {b: r.reindex(E.CL[b].index).ffill() for b in BOARDS}


def dose(lev, ser, label, nb=4):
    """剂量反应：按外部变量在**该区间内部**的分位把信号分 nb 组，看各组命中率是否单调。"""
    rows = []
    for tag, (w0, w1) in (("全程", W_Q), ("前半", H1), ("后半", H2)):
        sub = [r for r in lev if w0 <= pd.Timestamp(r[1]) <= w1]
        vals = np.array([ser[r[0]].loc[pd.Timestamp(r[1])] for r in sub], float)
        ok = np.array([r[2] for r in sub], bool)
        m = ~np.isnan(vals)
        vals, ok = vals[m], ok[m]
        if len(vals) < 40:
            rows.append((tag, "样本不足", "", ""))
            continue
        qs = np.quantile(vals, np.linspace(0, 1, nb + 1))
        cells = []
        for j in range(nb):
            lo, hi = qs[j], qs[j + 1]
            sel = (vals >= lo) & (vals <= hi) if j == nb - 1 else (vals >= lo) & (vals < hi)
            k, n = int(ok[sel].sum()), int(sel.sum())
            cells.append(f"Q{j+1} {100*k/n:.0f}%({n})" if n else f"Q{j+1} —")
        base = 100 * ok.mean()
        rows.append((tag, "  ".join(cells), f"{base:.1f}%", f"{len(vals)}"))
    print(f"\n  ── {label}　剂量反应（组内四分位，高→低看是否单调）")
    print(f"  {'区间':<6}{'Q1(低) → Q4(高)':<52}{'区间基准':>10}{'样本':>7}")
    for tag, c, b, n in rows:
        print(f"  {tag:<6}{c:<52}{b:>10}{n:>7}")


def ftest(E, lev, ser, cond, label):
    """把 cond 当过滤器套在并集上，报三窗口 + 分半 + 2015/2016。"""
    keep = []
    for b, dt, ok, rt in lev:
        v = ser[b].loc[pd.Timestamp(dt)]
        if pd.isna(v) or not cond(v):
            continue
        keep.append((b, dt, ok, rt))
    n, k, r, wl = st(keep, *W_Q)
    if n == 0:
        print(f"  {label:<30}{'0 个信号，无效':>20}")
        return None
    print(f"  {label:<30}{n:>7}{r:>8.1f}%{wl:>8.1f}{yr(keep, '2015'):>9}"
          f"{yr(keep, '2016'):>9}"
          f"{f'{st(keep, *W_57)[2]:.1f}%({st(keep, *W_57)[0]})':>14}"
          f"{f'{st(keep, *W_3)[2]:.1f}%({st(keep, *W_3)[0]})':>14}"
          f"{f'{st(keep, *H1)[2]:.1f}%({st(keep, *H1)[0]})':>15}"
          f"{f'{st(keep, *H2)[2]:.1f}%({st(keep, *H2)[0]})':>15}")
    return keep


def head2():
    print(f"  {'过滤条件':<30}{'保留':>7}{'命中':>9}{'下界':>8}{'2015':>9}{'2016':>9}"
          f"{'近5.7年':>14}{'近3年':>14}{'前半':>15}{'后半':>15}")


# ------------------------------------------------------------------ 取数
def fetch_cached(fname, fn, **kw):
    p = os.path.join(CACHE_DIR, fname)
    if os.path.exists(p):
        return pd.read_parquet(p)
    for a in range(3):
        try:
            d = fn(**kw)
            break
        except Exception as e:
            if a == 2:
                print(f"  ✘ {fname}: {type(e).__name__} {str(e)[:60]}")
                return None
            time.sleep(1.0)
    if d is None or len(d) == 0:
        return None
    d.to_parquet(p, index=False)
    return d


def main():
    E = s8.Evaluator()
    WQ = s32.wide(pd.read_parquet(s32.CACHE))
    inc = s18.merge_all(*[s17.sig_dates(E, E.score(nm), thr, 4, s17.COND_FN[cn])
                          for nm, thr, cn in s17.INC])
    lev = [(b, dt, ok, rt) for b, d in inc.items() for dt, (ok, rt) in d.items()]

    # ---------------------------------------------------------- A
    print("=" * 118)
    print("A. QVIX 决定性补测：在前半（2015-2020）内部，QVIX 高低还有区分度吗？")
    print("=" * 118)
    qv = mk_series(E, WQ["SH50ETF"])
    dose(lev, qv, "50ETF QVIX 绝对水平")
    print("\n  解读：若前半内部四分位命中率**没有单调性**，说明 QVIX 在 2015-2020 是纯噪声；")
    print("        s33 里「QVIX≥30 前半 44.4%」不是被阈值掩盖，而是它本来就没信息。")

    # ---------------------------------------------------------- B
    print("\n" + "=" * 118)
    print("B. 其余外部源：取数 + 质检")
    print("=" * 118)

    bond = fetch_cached("bond_rate.parquet",
                        ak.bond_zh_us_rate, start_date="20130101")
    nb = fetch_cached("hsgt_north.parquet",
                      ak.stock_hsgt_hist_em, symbol="北向资金")
    fund = fetch_cached("fund_new.parquet", ak.fund_new_found_em)
    val = fetch_cached("a_valuation.parquet", ak.stock_a_ttm_lyr)

    ext = {}
    if bond is not None:
        b = bond.rename(columns={"日期": "date"})
        b["date"] = pd.to_datetime(b["date"])
        b = b.set_index("date").sort_index()
        y10 = pd.to_numeric(b["中国国债收益率10年"], errors="coerce")
        sp = pd.to_numeric(b.get("中国国债收益率10年-2年"), errors="coerce")
        print(f"  ✔ 国债收益率   {len(y10.dropna())} 行  "
              f"{y10.dropna().index.min().date()} ~ {y10.dropna().index.max().date()}  "
              f"10Y 中位 {y10.median():.2f}")
        ext["10Y国债收益率"] = y10
        ext["10Y国债20日变化"] = y10 - y10.rolling(20).mean()
        ext["期限利差10Y-2Y"] = sp
    if nb is not None:
        n = nb.rename(columns={"日期": "date"})
        n["date"] = pd.to_datetime(n["date"])
        n = n.set_index("date").sort_index()
        nf = pd.to_numeric(n["当日成交净买额"], errors="coerce")
        last_ok = nf.dropna().index.max()
        print(f"  ✔ 北向净买     {len(nf.dropna())} 行  最后一个有效值 {last_ok.date()}")
        if last_ok < pd.Timestamp("2024-12-31"):
            print(f"     ⚠️ 2024-08 起沪深港通改按季披露，日频已断 → **只能做 2015-2024 的检验**")
        ext["北向净买(5日累计)"] = nf.rolling(5).sum()
    if fund is not None:
        f = fund.rename(columns={"成立日期": "date"})
        f["date"] = pd.to_datetime(f["date"], errors="coerce")
        f["sh"] = pd.to_numeric(f["募集份额"], errors="coerce")
        fs = f.dropna(subset=["date"]).groupby("date")["sh"].sum().sort_index()
        ext["新成立基金份额(20日累计)"] = fs.rolling(20).sum()
        print(f"  ✔ 新成立基金   {len(fs)} 个成立日  "
              f"{fs.index.min().date()} ~ {fs.index.max().date()}  （按成立日聚合，再 20 日累计）")
    if val is not None:
        v = val.rename(columns={"date": "date"})
        v["date"] = pd.to_datetime(v["date"])
        v = v.set_index("date").sort_index()
        pe = pd.to_numeric(v["middlePETTM"], errors="coerce")
        print(f"  ✔ 全A估值PE    {len(pe.dropna())} 行  "
              f"{pe.dropna().index.min().date()} ~ {pe.dropna().index.max().date()}  "
              f"（**月频**，只作慢变量参考）")
        ext["全A中位PE(月频)"] = pe

    # ---------------------------------------------------------- C
    print("\n" + "=" * 118)
    print("C. 剂量反应：这些源在并集信号上到底有没有区分度？")
    print("=" * 118)
    for lab, raw in ext.items():
        dose(lev, mk_series(E, raw), lab)

    # ---------------------------------------------------------- D
    print("\n" + "=" * 118)
    print("D. 作为过滤器套在并集上（含分半检验）")
    print("=" * 118)
    head2()
    n0, k0, r0, w0_ = st(lev, *W_Q)
    print(f"  {'【基准】无过滤':<30}{n0:>7}{r0:>8.1f}%{w0_:>8.1f}{yr(lev, '2015'):>9}"
          f"{yr(lev, '2016'):>9}"
          f"{f'{st(lev, *W_57)[2]:.1f}%({st(lev, *W_57)[0]})':>14}"
          f"{f'{st(lev, *W_3)[2]:.1f}%({st(lev, *W_3)[0]})':>14}"
          f"{f'{st(lev, *H1)[2]:.1f}%({st(lev, *H1)[0]})':>15}"
          f"{f'{st(lev, *H2)[2]:.1f}%({st(lev, *H2)[0]})':>15}")
    print("  " + "-" * 116)
    if "10Y国债收益率" in ext:
        s = mk_series(E, ext["10Y国债收益率"])
        for t in (2.7, 2.9, 3.1):
            ftest(E, lev, s, lambda x, t=t: x <= t, f"10Y ≤ {t}（利率低=宽松）")
    if "10Y国债20日变化" in ext:
        s = mk_series(E, ext["10Y国债20日变化"])
        for t in (-0.05, 0.0, 0.05):
            ftest(E, lev, s, lambda x, t=t: x <= t, f"10Y 低于MA20 且差≤{t}")
    if "期限利差10Y-2Y" in ext:
        s = mk_series(E, ext["期限利差10Y-2Y"])
        for t in (0.4, 0.6, 0.8):
            ftest(E, lev, s, lambda x, t=t: x >= t, f"期限利差 ≥ {t}（陡=宽松预期）")
    if "北向净买(5日累计)" in ext:
        s = mk_series(E, ext["北向净买(5日累计)"])
        for t in (0.0, 50.0, 100.0):
            ftest(E, lev, s, lambda x, t=t: x >= t, f"北向5日累计 ≥ {t:g} 亿（外资在买）")
    if "新成立基金份额(20日累计)" in ext:
        s = mk_series(E, ext["新成立基金份额(20日累计)"])
        sub = [s[r[0]].loc[pd.Timestamp(r[1])] for r in lev]
        sub = [x for x in sub if not np.isnan(x)]
        if sub:
            for q in (0.25, 0.5, 0.75):
                t = float(np.quantile(sub, q))
                ftest(E, lev, s, lambda x, t=t: x <= t,
                      f"基金发行冰点 ≤P{int(q*100)}")
    if "全A中位PE(月频)" in ext:
        s = mk_series(E, ext["全A中位PE(月频)"])
        sub = [s[r[0]].loc[pd.Timestamp(r[1])] for r in lev]
        sub = [x for x in sub if not np.isnan(x)]
        if sub:
            for q in (0.25, 0.5):
                t = float(np.quantile(sub, q))
                ftest(E, lev, s, lambda x, t=t: x <= t, f"全A中位PE ≤P{int(q*100)}（估值低）")

    # ---------------------------------------------------------- E
    print("\n" + "=" * 118)
    print("E. 结论")
    print("=" * 118)
    print("  （见 DELIVERY.md 第 16 节）")


if __name__ == "__main__":
    main()
