"""把个股日线补到 2013 年，用于：
  1) 让 p2/p98 分位锚点在长历史上稳定（根治 −6 阈值不可移植的问题）
  2) 拿到 2015 / 2018 / 2022 等其它熊市样本

只下载「各板块成分股的并集」，不下载全市场，量约 2000~2500 只。
已有缓存覆盖 2022-09 起，所以只补 2013-01-01 ~ 2023-06-30 这段。
"""
import sys, os, json, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"; os.environ["no_proxy"] = "*"
import pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
from senti import config as C, data

DEST = os.path.join(C.DATA_DIR, "cache", "stocks_hist")
os.makedirs(DEST, exist_ok=True)
START, END = "20130101", "20230630"
WORKERS = 8

# ---------- 1) 收集需要下载的股票代码 ----------
print("收集成分股并集 ...", flush=True)
codes = set()
try:
    for b in C.BOARD_ORDER:
        codes |= set(data.load_universe(b))
        print(f"  {C.BOARDS[b]['name']:<8} {len(codes)}")
except Exception as e:
    print("  universe FAIL", e)
vdir = os.path.join(C.DATA_DIR, "cache", "valid")
for f in os.listdir(vdir) if os.path.isdir(vdir) else []:
    if f.endswith("_cons.json"):
        codes |= set(json.load(open(os.path.join(vdir, f), encoding="utf-8")))
print("  并集股票数:", len(codes), flush=True)

todo = [c for c in sorted(codes) if not os.path.exists(os.path.join(DEST, f"{c}.parquet"))]
print("  待下载:", len(todo), flush=True)

# ---------- 2) 探测接口 ----------
print("\n探测 stock_zh_a_hist ...", flush=True)
try:
    t = ak.stock_zh_a_hist(symbol="600519", period="daily",
                           start_date="20130101", end_date="20130630", adjust="qfq")
    print("  OK rows:", len(t), "列:", list(t.columns))
except Exception as e:
    print("  FAIL", type(e).__name__, str(e)[:120])
    raise SystemExit(1)

# ---------- 3) 批量下载 ----------
lock = threading.Lock()
stat = {"ok": 0, "empty": 0, "fail": 0}
fails = []


def one(code):
    for attempt in range(3):
        try:
            d = ak.stock_zh_a_hist(symbol=code, period="daily",
                                   start_date=START, end_date=END, adjust="qfq")
            if d is None or len(d) == 0:
                with lock: stat["empty"] += 1
                return None
            d = d.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                                  "最高": "high", "最低": "low",
                                  "成交量": "volume", "成交额": "amount"})
            d["date"] = pd.to_datetime(d["date"])
            d = d[["date", "open", "high", "low", "close", "volume", "amount"]]
            d.to_parquet(os.path.join(DEST, f"{code}.parquet"), index=False)
            with lock: stat["ok"] += 1
            return code
        except Exception:
            time.sleep(0.6 * (attempt + 1))
    with lock:
        stat["fail"] += 1
        fails.append(code)
    return None


print(f"\n开始下载 {len(todo)} 只（{START} ~ {END}，{WORKERS} 线程）...", flush=True)
t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = {ex.submit(one, c): c for c in todo}
    for i, f in enumerate(as_completed(futs), 1):
        if i % 200 == 0:
            el = time.time() - t0
            print(f"  进度 {i}/{len(todo)}  ok={stat['ok']} empty={stat['empty']} "
                  f"fail={stat['fail']}  {el:.0f}s  预计剩余 {el/i*(len(todo)-i):.0f}s", flush=True)

print(f"\n完成：ok={stat['ok']} empty={stat['empty']} fail={stat['fail']}  用时 {time.time()-t0:.0f}s")
if fails:
    json.dump(fails, open(os.path.join(DEST, "_fails.json"), "w"), encoding="utf-8")
    print("  失败代码已存 _fails.json，前 20:", fails[:20])
print("  输出目录:", DEST)
