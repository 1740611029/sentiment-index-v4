"""把个股日线补到 2013 年（走新浪，东财 push2 在本环境不通）。

目的：
 1) 让 p2/p98 分位锚点在长历史上稳定，根治「−6 阈值不可移植」
 2) 拿到 2015 / 2018 / 2022 等其它熊市样本

范围：排除上证综指（2229 只）与科创板（2019 才有），只下有长历史意义的板块成分股，
      约 2300 只。同时把各板块**指数**日线也补到 2013。
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
IDEST = os.path.join(C.DATA_DIR, "cache", "index_hist")
os.makedirs(DEST, exist_ok=True); os.makedirs(IDEST, exist_ok=True)
S_START, S_END = "20130101", "20230630"
I_START, I_END = "20130101", "20260920"
WORKERS = 8

# 有长历史意义的板块（排除 大盘/上证综指、科创板）
KEEP_BOARDS = ["HS300", "CSI1000", "CSI2000", "CHINEXT"]
EXTRA_CONS = ["SZ50", "CSI100", "CSI500", "CSI800", "SH180"]
INDEX_MAP = {  # board_key -> sina symbol
    "HS300": "sh000300", "CSI1000": "sh000852", "CSI2000": "sh000932",
    "CHINEXT": "sz399006", "STAR": "sh000688", "SH": "sh000001",
    "SZ50": "sh000016", "CSI100": "sh000903", "CSI500": "sh000905",
    "CSI800": "sh000906", "SH180": "sh000010",
}

print("收集成分股并集 ...", flush=True)
codes = set()
for b in KEEP_BOARDS:
    codes |= set(data.load_universe(b))
    print(f"  {C.BOARDS[b]['name']:<8} 累计 {len(codes)}")
vdir = os.path.join(C.DATA_DIR, "cache", "valid")
if os.path.isdir(vdir):
    for f in os.listdir(vdir):
        if f.endswith("_cons.json"):
            k = f.replace("_cons.json", "")
            if k in EXTRA_CONS:
                codes |= set(json.load(open(os.path.join(vdir, f), encoding="utf-8")))
print("  并集股票数:", len(codes), flush=True)
with open(os.path.join(DEST, "_codes.json"), "w", encoding="utf-8") as f:
    json.dump(sorted(codes), f, ensure_ascii=False)

todo = [c for c in sorted(codes) if not os.path.exists(os.path.join(DEST, f"{c}.parquet"))]
print("  待下载:", len(todo), flush=True)

lock = threading.Lock()
stat = {"ok": 0, "empty": 0, "fail": 0}; fails = []


def sym(code):
    return ("sh" if code.startswith("6") else "sz") + code


def one(code):
    for attempt in range(3):
        try:
            d = ak.stock_zh_a_daily(symbol=sym(code), start_date=S_START,
                                    end_date=S_END, adjust="qfq")
            if d is None or len(d) == 0:
                with lock: stat["empty"] += 1
                return None
            keep = [c for c in ("date", "open", "high", "low", "close", "volume", "amount")
                    if c in d.columns]
            d = d[keep].copy()
            d["date"] = pd.to_datetime(d["date"])
            if "amount" not in d.columns:
                d["amount"] = d["volume"] * d["close"]
            d.to_parquet(os.path.join(DEST, f"{code}.parquet"), index=False)
            with lock: stat["ok"] += 1
            return code
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    with lock:
        stat["fail"] += 1; fails.append(code)
    return None


print(f"\n下载个股日线 {len(todo)} 只（{S_START}~{S_END}，{WORKERS} 线程）...", flush=True)
t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(one, c) for c in todo]
    for i, f in enumerate(as_completed(futs), 1):
        if i % 300 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  ok={stat['ok']} empty={stat['empty']} fail={stat['fail']}"
                  f"  {el:.0f}s  剩余约 {el/i*(len(todo)-i):.0f}s", flush=True)
print(f"  个股完成 ok={stat['ok']} empty={stat['empty']} fail={stat['fail']}  {time.time()-t0:.0f}s", flush=True)
if fails:
    with open(os.path.join(DEST, "_fails.json"), "w", encoding="utf-8") as f:
        json.dump(fails, f)

print("\n下载指数日线 ...", flush=True)
igot = {}
for k, s in INDEX_MAP.items():
    fp = os.path.join(IDEST, f"{k}.parquet")
    if os.path.exists(fp):
        igot[k] = len(pd.read_parquet(fp)); print(f"  {k:<9} 已缓存 {igot[k]}"); continue
    try:
        d = ak.stock_zh_index_daily(symbol=s)
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"] >= pd.Timestamp("2013-01-01")]
        if "amount" not in d.columns:
            d["amount"] = d["volume"] * d["close"]
        d.to_parquet(fp, index=False)
        igot[k] = len(d)
        print(f"  {k:<9} {s}  {len(d)} 行  {d['date'].min().date()} ~ {d['date'].max().date()}")
    except Exception as e:
        print(f"  {k:<9} FAIL {type(e).__name__}: {str(e)[:60]}")
    time.sleep(0.2)

print("\n完成。个股目录:", DEST, " 指数目录:", IDEST)
