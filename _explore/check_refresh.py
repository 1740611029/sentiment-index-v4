"""刷新校验：证明 `run.py refresh` 只追加、不改写历史（AGENTS.md 铁律八）。

用法
----
    python _explore/check_refresh.py --save     # 刷新**前**跑，把当前状态存成基线
    python _explore/check_refresh.py --check    # 刷新**后**跑，与基线比对（退出码 0/1）
    python _explore/check_refresh.py            # 只打印当前状态

为什么需要它
------------
本项目的**历史段是拼接/合成出来的**（`index_long` 的长历史、中证2000 的等权合成、
上游 v1/v2 的前复权缓存），它们的基准与成分股清单都和「今天」不同。
所以刷新只能是**纯追加**：新增日期、旧值一个都不能变。

判定方法（按日期取交集逐点比，不按位置）
----------------------------------------
对每个板块，取「基线日期 ∩ 当前日期」，要求这些日期上的：

1. **4 个模型的分值**（`score` / `swing` / `swing2` / `swing3`）逐点相等
2. **信号**（SENTI-1 底部 / SWING / SWING-2 / SWING-3 / 并集）的
   「日期 + 分值 + src + 共振数」逐条相等
3. **基线里的日期一个都不能少**（少了 = 历史被截断）

⚠️ 为什么**不**比对「命中率」这类统计数字：
   待验证信号的 T+7 到期后，`ok` / `ret` 会从 null 变成值 —— 这是**正常的**，
   命中率的分子分母本来就会随新数据变化。硬比会天天误报。
   真正的不变量是「**历史分值不变**」，所以只比这个。
   统计数字的变化会**打印出来供人工看一眼**，但不作为失败条件。

⚠️ 信号只比「入场时就已确定的字段」（date / score / src / reso），
   不比 `ok` / `ret` / `risk` / `notrap`（这些随未来数据推进而变）。

⚠️ 基线文件在 `_explore/_smoke_tmp/`（已 gitignore）—— 它是「上一次刷新前」的快照，
   随每次刷新更新，不该入库。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from senti import config as C, store  # noqa: E402

BASE = os.path.join(ROOT, "_explore", "_smoke_tmp", "refresh_baseline.json")
MODELS = ("score", "swing", "swing2", "swing3")
SIGNALS = ("bottom", "swing_events", "swing2_events", "swing3_events", "union_events")
# 信号里「入场时就已确定」的字段 —— 只有这些参与比对
ENTRY_KEYS = ("date", "score", "src", "reso", "ures")


def snapshot() -> dict:
    panels = store.load()
    payload = store.to_json(panels)
    s = store.summary(panels)

    snap = {"meta": store.meta(), "boards": {}, "stats": {}}
    for b in C.BOARD_ORDER:
        p = payload[b]
        board = {"dates": list(p["dates"])}
        for k in MODELS:
            board[k] = list(p[k])
        for k in SIGNALS:
            board[k] = [{q: e.get(q) for q in ENTRY_KEYS} for e in p[k]]
        snap["boards"][b] = board

    t = s["total"]
    snap["stats"]["bottom"] = [t["b_n"], t["b_ok"], t["b_nt"]]
    for k in ("swing", "swing2", "swing3", "union"):
        blk = s[k]["total"]
        snap["stats"][k] = [blk["n"], blk["ok"], blk["nt"]]
    for grp in ("event", "market"):
        for k, v in s[grp].items():
            if isinstance(v, dict):
                snap["stats"][f"{grp}_{k}"] = [v["n"], v["ok"]]
    return snap


def _by_date(dates: list, values: list) -> dict:
    return {d: v for d, v in zip(dates, values)}


def check(cur: dict, base: dict) -> tuple[bool, list[str], list[str], list[str]]:
    """返回 (是否通过, 失败项, 提示项, 统计变动项)。"""
    bad: list[str] = []
    warn: list[str] = []

    for b in C.BOARD_ORDER:
        ob, nb = base["boards"][b], cur["boards"][b]
        od, nd = set(ob["dates"]), set(nb["dates"])
        name = C.BOARDS[b]["name"]
        base_last = max(od)

        # ---- 3a. 基线里的日期一个都不能少（少了 = 历史被截断）----
        lost = sorted(od - nd)
        if lost:
            bad.append(f"[{name}] ⛔ 历史日期丢失 {len(lost)} 天："
                       f"{lost[:3]} … {lost[-1:]}")
        # ---- 3b. 也不能凭空多出「基线窗口之内」的日期（交易日历被改写）----
        injected = sorted(d for d in nd - od if d <= base_last)
        if injected:
            bad.append(f"[{name}] ⛔ 历史窗口内多出日期：{injected[:3]}")

        shared = sorted(od & nd)
        if not shared:
            bad.append(f"[{name}] ⛔ 与基线没有任何共同日期")
            continue
        if not (nd - od):
            warn.append(f"[{name}] ⚠️ 没有新增日期（last_date 未前进）")

        # ---- 1. 4 个模型的分值逐点比对（按各模型自身的日期交集）----
        for k in MODELS:
            a = _by_date(ob["dates"], ob[k])
            c = _by_date(nb["dates"], nb[k])
            if len(a) != len(ob["dates"]) or len(c) != len(nb["dates"]):
                bad.append(f"[{name}] ⛔ 基线/当前快照不一致：{k} 的日期数与分值数不等"
                           f"（{len(ob['dates'])}/{len(ob[k])}、"
                           f"{len(nb['dates'])}/{len(nb[k])}）")
                continue
            diffs = [(d, a[d], c[d]) for d in sorted(set(a) & set(c)) if a[d] != c[d]]
            if diffs:
                d, x, y = diffs[0]
                bad.append(f"[{name}] ⛔ {k} 历史被改写 {len(diffs)} 处，"
                           f"例 {d}: {x} → {y}")

        # ---- 2. 信号逐条比对（只比入场字段）----
        for k in SIGNALS:
            a = {e["date"]: e for e in ob[k]}
            c = {e["date"]: e for e in nb[k]}
            miss = sorted(set(a) - set(c))
            if miss:
                bad.append(f"[{name}] ⛔ {k} 历史信号消失：{miss[:3]}")
            # 历史窗口内凭空多出信号 = 模型被改了（新增日期上的信号是正常的）
            extra = sorted(d for d in set(c) - set(a) if d <= base_last)
            if extra:
                bad.append(f"[{name}] ⛔ {k} 历史窗口内多出信号：{extra[:3]}")
            chg = [d for d in sorted(set(a) & set(c)) if a[d] != c[d]]
            if chg:
                bad.append(f"[{name}] ⛔ {k} 信号内容变了 @{chg[0]}："
                           f"{a[chg[0]]} → {c[chg[0]]}")

    # ---- 统计数字：只打印，不作为失败条件 ----
    stat_notes = []
    for k, v in base["stats"].items():
        nv = cur["stats"].get(k)
        if nv != v:
            stat_notes.append(f"{k}: {v} → {nv}")

    return (not bad), bad, warn, stat_notes


def show(cur: dict) -> None:
    m = cur["meta"]
    print(f"last_date = {m['last_date']}　built_at = {m['built_at']}")
    print(f"\n{'板块':<10}{'天数':>6}{'末日':>14}{'SENTI-1':>9}{'SWING':>8}"
          f"{'SWING-2':>9}{'SWING-3':>9}{'信号数':>8}")
    for b in C.BOARD_ORDER:
        p = cur["boards"][b]
        n = len(p["dates"])
        nsig = sum(len(p[k]) for k in SIGNALS)
        print(f"{C.BOARDS[b]['name']:<10}{n:>6}{p['dates'][-1]:>14}"
              f"{p['score'][-1]:>9.1f}{p['swing'][-1]:>8.1f}"
              f"{p['swing2'][-1]:>9.1f}{p['swing3'][-1]:>9.1f}{nsig:>8}")
    print("\n统计数字（信号数 / 命中 / 未被套）：")
    for k, v in cur["stats"].items():
        print(f"  {k:<20}{v}")


def main() -> int:
    ap = argparse.ArgumentParser(description="刷新校验：证明只追加、不改写历史")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--save", action="store_true", help="把当前状态存成基线（刷新前跑）")
    g.add_argument("--check", action="store_true", help="与基线比对（刷新后跑）")
    args = ap.parse_args()

    cur = snapshot()

    if args.save:
        os.makedirs(os.path.dirname(BASE), exist_ok=True)
        with open(BASE, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False)
        n = sum(len(cur["boards"][b]["dates"]) for b in C.BOARD_ORDER)
        print(f"✔ 基线已保存：{BASE}")
        print(f"   last_date = {cur['meta']['last_date']}　共 {n} 行"
              f"　（现在可以去跑 run.py refresh，回来跑 --check）")
        return 0

    if args.check:
        if not os.path.exists(BASE):
            print(f"✘ 找不到基线 {BASE}\n  请先在刷新**前**跑一次 --save")
            return 2
        with open(BASE, encoding="utf-8") as f:
            base = json.load(f)
        ok, bad, warn, notes = check(cur, base)
        print(f"基线 last_date = {base['meta']['last_date']}"
              f"　当前 last_date = {cur['meta']['last_date']}\n")
        for m in bad:
            print("  " + m)
        for m in warn:
            print("  " + m)
        if notes:
            print("  统计数字变动（正常：待验证信号到期后 ok/ret 本来就会变，不作为失败条件）：")
            for m in notes:
                print("    " + m)
        print()
        if ok:
            print("✔ 通过：纯追加，历史未被改写")
            print("  （共同日期上的分值 / 信号逐点一致，基线日期一个没丢）")
            return 0
        print("✘ 失败：刷新改写了历史 —— 这是 bug，不是「数据变好了」")
        print("  排查方向：① 是否用全序列覆盖了本地历史 ② 是否重算了历史段（合成/复权）")
        print("            ③ 因果锚是否漏了 .shift(1) ④ 广度分母的股票集合是否变了")
        return 1

    show(cur)
    return 0


if __name__ == "__main__":
    sys.exit(main())
