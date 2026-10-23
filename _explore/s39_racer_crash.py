"""复现 / 验证：并发创建 py_mini_racer.MiniRacer 是否会让进程直接崩在 mini_racer.dll。

背景（2026-10-23 排查「每天第一次点 更新数据 必报错」）
--------------------------------------------------------
`akshare.stock.stock_zh_a_sina.stock_zh_a_daily()` 每调用一次就 **新建一个 V8 isolate**：

    r = requests.get(...)
    js_code = py_mini_racer.MiniRacer()      # ← 这里
    js_code.eval(hk_js_decode)

而 `senti/fetch.py::fetch_stocks()` 用 `ThreadPoolExecutor(max_workers=8)` 并发调它，
一次 refresh ≈ 4500 次调用 = 4500 个 isolate，且前 8 个是**同时**创建的。

Windows 事件日志（Application, Id 1000）实测：
    错误模块 mini_racer.dll / 异常代码 0x80000003 / 偏移 0x14fe5f8
    进程存活 3~63 秒（= 第一波 8 个 isolate 创建的时候）

0x80000003 = STATUS_BREAKPOINT = V8 的 IMMEDIATE_CRASH()（MSVC 下就是 __debugbreak()），
即 V8 内部某个 CHECK 失败。**不是 Python 异常，try/except 拦不住，进程直接死。**

本脚本不联网、只做「建 isolate + eval 解密函数」，两种模式对比：

    python _explore/s39_racer_crash.py new     # 现状：每次调用新建 isolate
    python _explore/s39_racer_crash.py reuse   # 修法：每线程复用 + 首次创建加锁

崩溃时退出码是 -2147483645（0x80000003），正常结束时打印 OK。
"""
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import py_mini_racer
from akshare.stock.stock_zh_a_sina import hk_js_decode

WORKERS = 8
ROUNDS = 300          # 300 轮 × 8 线程 = 2400 次「取一次日线」的 V8 开销

_tls = threading.local()
_init_lock = threading.Lock()
_Real = py_mini_racer.MiniRacer


def _reuse_factory(*a, **kw):
    """按线程复用同一个 isolate；首次创建串行化（V8 全局初始化不是线程安全的）。"""
    inst = getattr(_tls, "racer", None)
    if inst is None:
        with _init_lock:
            inst = _Real(*a, **kw)
        _tls.racer = inst
    return inst


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "new"
    if mode == "reuse":
        py_mini_racer.MiniRacer = _reuse_factory

    def one(_i: int) -> int:
        js = py_mini_racer.MiniRacer()
        js.eval(hk_js_decode)
        return 1

    t0 = time.time()
    total = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(one, i) for i in range(ROUNDS * WORKERS)]
        for i, f in enumerate(futs, 1):
            f.result()
            total += 1
            if i % (WORKERS * 20) == 0:
                print(f"  mode={mode}  {i}/{ROUNDS * WORKERS} 次  "
                      f"{time.time() - t0:.1f}s", flush=True)
    print(f"OK mode={mode}  {total} 次调用  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
