import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"; os.environ["no_proxy"] = "*"
import akshare as ak


def t(tag, fn):
    t0 = time.time()
    try:
        r = fn()
        print(f"  OK   {tag:<40} rows={len(r)}  {time.time()-t0:.1f}s")
        if hasattr(r, "columns"): print(f"       列: {list(r.columns)}")
        if hasattr(r, "head"): print(r.tail(2).to_string())
        return r
    except Exception as e:
        print(f"  FAIL {tag:<40} {type(e).__name__}: {str(e)[:80]}")
        return None


print("个股历史日线数据源探测")
print("-" * 100)
t("新浪 stock_zh_a_daily sh600519", lambda: ak.stock_zh_a_daily(
    symbol="sh600519", start_date="20130101", end_date="20130630", adjust="qfq"))
t("腾讯 stock_zh_a_hist_tx", lambda: ak.stock_zh_a_hist_tx(
    symbol="sh600519", start_date="20130101", end_date="20130630", adjust="qfq"))
t("网易 stock_zh_a_hist_163", lambda: ak.stock_zh_a_hist_163(symbol="600519"))
t("东财 stock_zh_a_hist(对照，预期失败)", lambda: ak.stock_zh_a_hist(
    symbol="600519", period="daily", start_date="20130101", end_date="20130630", adjust="qfq"))
