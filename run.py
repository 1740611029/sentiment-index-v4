"""A股板块恐贪情绪指数 v4 —— 命令行入口。

  python run.py refresh   抓新数据 + 重建全部面板（**唯一会联网的命令**）
  python run.py build     重建个股指标 + 面板缓存（只读本地缓存）
  python run.py update    重建面板缓存（复用个股指标）
  python run.py stats     打印各板块命中率
  python run.py serve     启动网页（默认 8779）

⚠️ update / build 都**不联网**：数据链的源头是本地缓存，重算一万遍日期也不会前进。
   要让页面日期往前走，必须 `refresh`。详见 senti/fetch.py 的模块说明。
"""
from __future__ import annotations

import argparse
import sys

from senti import config as C, data, model, store, web


def cmd_build(args):
    data.build_stock_indicators(force=True)
    store.build_and_cache(force=True)
    # ⚠️ 必须显式重建：store.build_and_cache() 只管 SENTI-1 面板，
    # 三个小波段各有独立模块与面板目录，且 build_and_cache() 默认 force=False。
    store.rebuild_swing_panels(force=True)
    print("完成：个股指标 + SENTI-1 面板 + 三个小波段面板已重建")


def cmd_update(args):
    store.build_and_cache(force=True)
    print("完成：面板缓存已重建")
    print("⚠️ 注意：update 不抓新数据，last_date 不会前进。要更新数据请用 refresh。")


def cmd_refresh(args):
    """抓新数据 → 重建全部面板。这是本项目唯一会联网的命令。"""
    from senti import fetch

    r = fetch.refresh(lookback=args.lookback, workers=args.workers,
                      skip_stocks=args.index_only)

    print("⑤ 重建个股指标长表 ...")
    data.build_stock_indicators(force=True)
    print("⑥ 重建 SENTI-1 面板 ...")
    store.build_and_cache(force=True)
    print("⑦ 重建三个小波段面板（必须 force，否则继续用旧面板）...")
    store.rebuild_swing_panels(force=True)

    m = store.meta()
    print(f"完成：last_date = {m['last_date']}　built_at = {m['built_at']}")

    # 指数接口进度不一致 → 说清「为什么没更新到接口最新的那天」，避免误以为又没生效
    cap = r["cap"]
    pre = r.get("before_align") or {}
    behind = {b: d for b, d in pre.items() if d > cap}
    if behind:
        print(f"\n⚠️ 有板块的指数接口还没出 {cap} 之后的数据，本次统一更新到 {cap}：")
        for b, d in sorted(behind.items(), key=lambda kv: kv[1]):
            print(f"     {C.BOARDS[b]['name']:<8} 接口已有 {d}")
        print("  → 这是正常的（各交易所指数发布时点不同，深市常比沪市晚）。")
        print("    等接口补齐后（一般当晚）再跑一次即可自动补上，数据不会丢。")
    elif m["last_date"] < cap:
        print(f"⚠️ 面板末日 {m['last_date']} 早于数据末日 {cap}，请检查面板重建是否报错。")


def cmd_stats(args):
    panels = store.load()
    s = store.summary(panels)
    thr = s["entry_thr"]
    print(f"{'板块':<10}{'底部命中':>11}{'未被套':>10}{'基线':>9}")
    for b in C.BOARD_ORDER:
        p = s["per"][b]
        bt = p["bottom"]
        bs = f"{bt['ok']}/{bt['n']}"
        nt = f"{bt['nt']}/{bt['n']}"
        print(f"{C.BOARDS[b]['name']:<10}{bs:>11}{nt:>10}"
              f"{p['base_bottom']:>8.1f}%")
    t = s["total"]
    pct = lambda a, b: (a / b * 100) if b else 0.0
    print(f"\n合计（近3年 {C.BACKTEST_START} ~ 今，判定 T+{store.H}，容差 {store.TOL:.0%}）")
    print(f"  入场规则：昨日分值 ≤ {thr:.0f} 且今日回升 → 今日收盘买入"
          f"（入场后 {store.COOL_TRADING} 交易日冷却）")
    print(f"  底部  命中 {t['b_ok']}/{t['b_n']} = {pct(t['b_ok'],t['b_n']):.0f}%"
          f"   未被套超3% {t['b_nt']}/{t['b_n']} = {pct(t['b_nt'],t['b_n']):.0f}%")
    print("  顶部信号已移除：实测 2/9 = 22.2%，随机基线 40.1%，p = 0.93，与瞎猜无差异")

    # ---- 小波段 SWING ----
    sw = s.get("swing")
    if sw:
        st = sw["total"]
        print(f"\n小波段 SWING（近3年 {C.BACKTEST_START} ~ 今）")
        print(f"  入场规则：SWING 超卖度 ≤ {sw['thr']:.0f} → 当日收盘买入"
              f"（冷却 {sw['cool']} 交易日，判定 T+{sw['hold']}，容差 {sw['tol']}%）")
        print(f"{'板块':<10}{'命中':>11}{'未被套':>10}{'基线':>9}")
        for b in C.BOARD_ORDER:
            v = sw["per"][b]
            print(f"{C.BOARDS[b]['name']:<10}{str(v['ok'])+'/'+str(v['n']):>11}"
                  f"{str(v['nt'])+'/'+str(v['n']):>10}{v['base']:>8.1f}%")
        print(f"  合计  命中 {st['ok']}/{st['n']} = {st['rate']}%"
              f"   未被套 {st['nt']}/{st['n']}"
              f"   待验证 {st['pend']}   基线 {st['base']}%")
        print(f"  分层  共振≥{sw['reso_min']} {st['res_ok']}/{st['res_n']}"
              f" = {pct(st['res_ok'],st['res_n']):.1f}%"
              f"　共振≥4 {st['res4_ok']}/{st['res4_n']}"
              f" = {pct(st['res4_ok'],st['res4_n']):.1f}%"
              f"　共振≥5(S级) {st['res5_ok']}/{st['res5_n']}"
              f" = {pct(st['res5_ok'],st['res5_n']):.1f}%")
        print("  （共振越强越准、但信号越少；等级只做标签不过滤，两个目标日都在全部信号里）")
        print("  注意：上表是近 3 年窗口。拉长到 2021-01 起（含 2021-22 熊市，n=116）"
              "全部信号降到 59.5%，而共振≥5 反而升到 87.0%（20/23）——"
              "S 级这档跨窗口最稳。")

    # ---- 小波段 SWING-2（动量拐点）----
    sw2 = s.get("swing2")
    un = s.get("union")
    if sw2:
        t2 = sw2["total"]
        print(f"\n小波段 SWING-2（动量拐点，第二个小波段模型）")
        print(f"  入场规则：SWING-2 = MACD柱/收盘价的因果锚分位 ≤ {sw2['thr']:.0f}"
              f" → 当日收盘买入（冷却 {sw2['cool']}，判定 T+{sw2['hold']}，容差 {sw2['tol']}%）")
        print(f"  合计  命中 {t2['ok']}/{t2['n']} = {t2['rate']}%"
              f"   未被套 {t2['nt']}/{t2['n']}   待验证 {t2['pend']}   基线 {t2['base']}%")

    # ---- 小波段 SWING-3（均线拐点）----
    sw3 = s.get("swing3")
    if sw3:
        t3 = sw3["total"]
        print(f"\n小波段 SWING-3（均线拐点，第三个小波段模型）")
        print(f"  入场规则：SWING-3 = MA20 的 5 日斜率的因果锚分位 ≤ {sw3['thr']:.0f}"
              f" 且当日回升 → 当日收盘买入"
              f"（冷却 {sw3['cool']}，判定 T+{sw3['hold']}，容差 {sw3['tol']}%）")
        print(f"{'板块':<10}{'命中':>11}{'未被套':>10}{'基线':>9}")
        for b in C.BOARD_ORDER:
            v = sw3["per"][b]
            print(f"{C.BOARDS[b]['name']:<10}{str(v['ok'])+'/'+str(v['n']):>11}"
                  f"{str(v['nt'])+'/'+str(v['n']):>10}{v['base']:>8.1f}%")
        print(f"  合计  命中 {t3['ok']}/{t3['n']} = {t3['rate']}%"
              f"   未被套 {t3['nt']}/{t3['n']}"
              f"   待验证 {t3['pend']}   基线 {t3['base']}%")
        print(f"  分层  共振≥{sw3['reso_min']} {t3['res_ok']}/{t3['res_n']}"
              f" = {pct(t3['res_ok'],t3['res_n']):.1f}%"
              f"　共振≥4 {t3['res4_ok']}/{t3['res4_n']}"
              f" = {pct(t3['res4_ok'],t3['res4_n']):.1f}%"
              f"　共振≥5(S级) {t3['res5_ok']}/{t3['res5_n']}"
              f" = {pct(t3['res5_ok'],t3['res5_n']):.1f}%")
        print("  与另两个模型几乎不重叠：J(SWING-3,SWING)=0.03、J(SWING-3,SWING-2)=0.00")

    # ---- 三模型并集（页面默认视图）----
    if un:
        tu = un["total"]
        print(f"\n并集（SWING ∪ SWING-2 ∪ SWING-3，同日去重，页面默认视图）")
        print(f"{'板块':<10}{'命中':>11}{'未被套':>10}{'基线':>9}")
        for b in C.BOARD_ORDER:
            v = un["per"][b]
            print(f"{C.BOARDS[b]['name']:<10}{str(v['ok'])+'/'+str(v['n']):>11}"
                  f"{str(v['nt'])+'/'+str(v['n']):>10}{tu['base']:>8.1f}%")
        print(f"  合计  命中 {tu['ok']}/{tu['n']} = {tu['rate']}%"
              f"   未被套 {tu['nt']}/{tu['n']}   待验证 {tu['pend']}   基线 {tu['base']}%")
        print(f"  → 对比原两模型合并（143 次 / 66.4%）：信号 +32%、命中 +4.5pp，"
              f"两个维度同时上升")
        # ---- 事件级（同板块 ±4 自然日只算第一枪）----
        eblk = s.get("event")
        if eblk:
            print(f"\n  事件级（同板块 ±{eblk['gap_days']} 自然日算一件事，只算第一枪）")
            print(f"  {'口径':<14}{'信号数':>8}{'事件数':>8}{'信号命中':>10}"
                  f"{'事件命中':>10}{'下界':>9}")
            for k_, lab in (("swing", "SWING"), ("swing2", "SWING-2"),
                            ("swing3", "SWING-3"), ("union", "三模型合并")):
                e_ = eblk[k_]
                sig = (un["total"] if k_ == "union"
                       else (s.get(k_) or {}).get("total", {}))
                sn = sig.get("n")
                sr = sig.get("rate")
                print(f"  {lab:<14}{str(sn):>8}{e_['n']:>8}"
                      f"{(str(sr) + '%') if sr is not None else '—':>10}"
                      f"{(str(e_['rate']) + '%') if e_['rate'] is not None else '—':>10}"
                      f"{(str(e_['wilson']) + '%') if e_['wilson'] is not None else '—':>9}")
            print("  → 信号次数回答「页面画了几个点」，事件次数才是独立的入场机会数；")
            print("    同一波下跌三个模型隔几天各触发一次，按同日去重会重复计数。")
        # ---- 市场级（跨板块 ±5 自然日算一次）----
        mblk = s.get("market")
        if mblk:
            print(f"\n  市场级（跨板块 ±{mblk['gap_days']} 自然日算一件事，只算第一枪）")
            print("  → 这一层回答「这套系统一年动手几次」，含「某板块单独崩」的局部机会。")
            print(f"  {'口径':<14}{'件数':>7}{'命中':>9}{'下界':>8}{'年化':>8}"
                  f"{'其中长期阴跌(≥8天)':>20}")
            for k_, lab in (("swing", "SWING"), ("swing2", "SWING-2"),
                            ("swing3", "SWING-3"), ("union", "三模型合并"),
                            ("bottom", "SENTI-1底部")):
                e_ = mblk[k_]
                if not e_["n"]:
                    continue
                lg = e_["long"]
                lgtxt = (f"{lg['ok']}/{lg['n']} = {lg['rate']}%"
                         if lg["n"] else "无")
                print(f"  {lab:<14}{e_['n']:>7}"
                      f"{(str(e_['rate']) + '%') if e_['rate'] is not None else '—':>9}"
                      f"{(str(e_['wilson']) + '%') if e_['wilson'] is not None else '—':>8}"
                      f"{e_['n']/3:>8.1f}{lgtxt:>20}")
            print("  ⚠️ 「长期阴跌」（簇跨度 ≥8 天）的历史命中率只有 19~36%，")
            print("     但它在第一枪时不可知，**不能当过滤器**（_explore/s26 已证）。")
            print("  ⚠️ 「第一枪」口径对信号更密的模型天然不利（最早一枪更早 = 更接近")
            print("     下跌途中），所以这一层只看「几次机会、效果如何」，不做模型排名。")
        print("  已知边界：2015/2016 连环崩塌会连续误判（SWING-2 长历史 34.7%、"
              "SWING-3 2015 年 40%），这是「超卖抄底」整个范式的通病，"
              "同期现 SWING 价格部分也是 33% / 33%。")


def main():
    ap = argparse.ArgumentParser(description="A股板块恐贪情绪指数 v4")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("build", help="重建全部（个股指标 + 面板）")
    sub.add_parser("update", help="重建面板缓存（不抓新数据）")
    pr = sub.add_parser("refresh", help="抓新数据 + 重建全部面板（联网）")
    pr.add_argument("--lookback", type=int, default=120,
                    help="个股增量抓取的日历日回溯（默认 120，需覆盖缓存末日到今天）")
    pr.add_argument("--workers", type=int, default=8, help="个股抓取并发数（默认 8）")
    pr.add_argument("--index-only", action="store_true",
                    help="只抓指数、跳过个股（调试用；广度因子会缺当天数据）")
    sub.add_parser("stats", help="打印命中率")
    p = sub.add_parser("serve", help="启动网页")
    p.add_argument("--port", type=int, default=C.WEB_PORT)
    p.add_argument("--no-browser", action="store_true")

    args = ap.parse_args()
    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "update":
        cmd_update(args)
    elif args.cmd == "refresh":
        cmd_refresh(args)
    elif args.cmd == "stats":
        cmd_stats(args)
    elif args.cmd == "serve":
        web.serve(port=args.port, open_browser=not args.no_browser)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
