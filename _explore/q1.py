"""统计 SENTI-1（大波段）与 SWING（小波段）的测试时间范围与样本数量。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from senti import config as C, store, swing

print("=" * 78)
print("【1】面板（展示窗口）时间范围与数据量")
print("=" * 78)

P = store.load()
S = swing.load()

print(f"{'板块':<10}{'SENTI-1 起':>12}{'止':>12}{'行数':>7}   {'SWING 起':>12}{'止':>12}{'行数':>7}")
for b in C.BOARD_ORDER:
    p, s = P[b], S[b]
    print(f"{C.BOARDS[b]['name']:<10}"
          f"{str(p.index.min().date()):>12}{str(p.index.max().date()):>12}{len(p):>7}   "
          f"{str(s.index.min().date()):>12}{str(s.index.max().date()):>12}{len(s):>7}")

allmin = min(p.index.min() for p in P.values())
allmax = max(p.index.max() for p in P.values())
print(f"\n统一区间: {allmin.date()} ~ {allmax.date()}  (交易日 ~{len(P['SH'])})")
print(f"锚点预热起点 HIST_START = {C.HIST_START}（只用于算刻度，不进图）")
print(f"展示起点 BACKTEST_START = {C.BACKTEST_START}")

# ---- 有效分值（非 NaN）行数 ----
print("\n有效分值行数（NaN = 锚点预热不足）:")
for b in C.BOARD_ORDER:
    print(f"  {C.BOARDS[b]['name']:<10} SENTI-1 {P[b]['score'].notna().sum():>5} / {len(P[b]):<5}"
          f"  SWING {S[b]['swing'].notna().sum():>5} / {len(S[b])}")

# =========================================================
print()
print("=" * 78)
print("【2】SENTI-1 大波段：信号数量")
print("=" * 78)
reso = store.resonance(P)
bna = store.bna_series()
mg = store.margin_series()
tot_b = tot_t = 0
rows = []
for b in C.BOARD_ORDER:
    bot, top = store.events(P[b], reso, bna, mg)
    tot_b += len(bot); tot_t += len(top)
    ok = sum(1 for e in bot if e["ok"])
    rows.append((C.BOARDS[b]["name"], len(bot), ok, len(top), sum(1 for e in top if e["ok"])))
for n, nb, ok, nt, okt in rows:
    print(f"  {n:<10} 底部 {nb:>3} 次（命中 {ok:>2}）   顶部 {nt:>3} 次（命中 {okt}）")
print(f"  {'合计':<10} 底部 {tot_b:>3} 次   顶部 {tot_t:>3} 次")

allb, allt = [], []
for b in C.BOARD_ORDER:
    bot, top = store.events(P[b], reso, bna, mg)
    for e in bot: e["board"] = b; allb.append(e)
    for e in top: e["board"] = b; allt.append(e)
print(f"\n底部信号日期范围: {min(e['date'] for e in allb)} ~ {max(e['date'] for e in allb)}")
if allt:
    print(f"顶部信号日期范围: {min(e['date'] for e in allt)} ~ {max(e['date'] for e in allt)}")
g = {}
for e in allb: g[e["grade"]] = g.get(e["grade"], 0) + 1
print(f"底部分级: {g}")
print(f"底部带完整 T+60 的: {sum(1 for e in allb if e['ret60'] is not None)} / {len(allb)}")

# =========================================================
print()
print("=" * 78)
print("【3】SWING 小波段：信号数量")
print("=" * 78)
sm = swing.summary(S)
print(f"{'板块':<10}{'信号':>6}{'已判定':>8}{'待验证':>8}{'命中':>6}{'命中率':>8}{'基线':>8}")
for b in C.BOARD_ORDER:
    v = sm["per"][b]
    print(f"{C.BOARDS[b]['name']:<10}{v['n']+v['pend']:>6}{v['n']:>8}{v['pend']:>8}"
          f"{v['ok']:>6}{str(v['rate'])+'%':>8}{str(v['base'])+'%':>8}")
t = sm["total"]
print(f"{'合计':<10}{t['n']+t['pend']:>6}{t['n']:>8}{t['pend']:>8}{t['ok']:>6}"
      f"{str(t['rate'])+'%':>8}{str(t['base'])+'%':>8}")

res = swing.resonance(S)
allev = []
for b in C.BOARD_ORDER:
    for e in swing.events(S[b], res):
        e["board"] = b; allev.append(e)
done = [e for e in allev if e["ok"] is not None]
pend = [e for e in allev if e["ok"] is None]
print(f"\n信号日期范围: {min(e['date'] for e in allev)} ~ {max(e['date'] for e in allev)}")
print(f"已判定 {len(done)} 个 / 待验证 {len(pend)} 个（待验证 = "
      f"{[e['date'] for e in pend]}）")
for lab, lo in [("S 级 共振≥5", 5), ("A 级 共振≥4", 4), ("B 级 共振2~3", 2), ("全部", 0)]:
    sub = [e for e in done if (e["reso"] or 0) >= lo]
    if lab == "B 级 共振2~3":
        sub = [e for e in done if 2 <= (e["reso"] or 0) <= 3]
    if sub:
        print(f"  {lab:<12} n={len(sub):>3}  命中 {sum(1 for e in sub if e['ok']):>3} "
              f"= {sum(1 for e in sub if e['ok'])/len(sub)*100:.1f}%")
