"""构建 6 个交付板块的长历史指数（供锚点窗口预热用）。

  SH / STAR / CHINEXT / CSI1000 / HS300 —— 新浪长历史与 v2 缓存完全一致（比率 1.0000），直接拼
  CSI2000                              —— 该指数 2023-08 才发布，新浪返回的是错误序列。
                                          用「当前成分股等权收益指数」合成 2023-06 之前那段，
                                          并按 2023-06-01 的取值对齐到生产口径（连续无跳变）。

输出：data/cache/index_long/{board}.parquet
注意：预热段只用于锚点计算，页面展示从 BACKTEST_START 开始，不会显示合成段的价格。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from senti import config as C, data
import longhist

OUT = os.path.join(C.DATA_DIR, "cache", "index_long")
os.makedirs(OUT, exist_ok=True)
IH = os.path.join(C.DATA_DIR, "cache", "index_hist")
MAP = {"SH": "SH", "STAR": "STAR", "CHINEXT": "CHINEXT",
       "CSI1000": "CSI1000", "CSI2000": "CSI2000", "HS300": "HS300"}
MIN_STOCKS = 100


def synth_eqweight(codes, upto):
    """成分股等权日收益指数：只用 upto 之前的数据。"""
    px = {}
    n = 0
    for c in codes:
        try:
            df = longhist.load_merged(c)
        except Exception:
            continue
        if df is None or len(df) == 0 or "close" not in df:
            continue
        d = df[["date", "close"]].copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"] < upto]
        if len(d) < 30:
            continue
        px[c] = d.set_index("date")["close"].astype(float)
        n += 1
        if n % 500 == 0:
            print(f"      读取 {n}/{len(codes)}", flush=True)
    if not px:
        return None
    print(f"      合成用 {len(px)} 只成分股", flush=True)
    wide = pd.DataFrame(px).sort_index()
    ret = wide.pct_change(fill_method=None)
    cnt = ret.notna().sum(axis=1)
    r = ret.mean(axis=1, skipna=True).where(cnt >= MIN_STOCKS)
    r = r.dropna()
    if len(r) < 200:
        return None
    return (1.0 + r).cumprod()


def main():
    for bk, fn in MAP.items():
        base = data.load_index(bk)
        base["date"] = pd.to_datetime(base["date"])
        cut = base["date"].min()
        print(f"\n【{bk}】v2 缓存 {len(base)} 行，起点 {cut.date()}", flush=True)
        if bk != "CSI2000":
            p = os.path.join(IH, f"{fn}.parquet")
            h = pd.read_parquet(p); h["date"] = pd.to_datetime(h["date"])
            for col in ("open", "high", "low", "close"):
                if col not in h.columns:
                    h[col] = h["close"]
            h = h[["date", "open", "high", "low", "close", "volume", "amount"]]
            pre = h[h["date"] < cut].copy()
            j = base.set_index("date")["close"].to_frame("v").join(
                h.set_index("date")["close"].to_frame("h"), how="inner").dropna()
            ratio = float((j["v"] / j["h"]).median())
            for col in ("open", "high", "low", "close"):
                pre[col] = pre[col] * ratio
            print(f"   拼接 {len(pre)} 行（比率 {ratio:.6f}）", flush=True)
        else:
            codes = data.load_universe("CSI2000")
            print(f"   合成中证2000：{len(codes)} 只成分股", flush=True)
            cum = synth_eqweight(codes, cut)
            if cum is None:
                print("   合成失败，回退为仅 v2 缓存"); pre = None
            else:
                v0 = float(base["close"].iloc[0])
                scale = v0 / float(cum.iloc[-1])
                pre = pd.DataFrame({
                    "date": cum.index,
                    "close": cum.to_numpy() * scale,
                })
                pre["open"] = pre["close"]; pre["high"] = pre["close"]; pre["low"] = pre["close"]
                pre["volume"] = np.nan; pre["amount"] = np.nan
                # 最末一天与 v2 起点重合，去掉避免重复
                pre = pre[pre["date"] < cut]
                print(f"   合成 {len(pre)} 行  {pre['date'].min().date()} ~ {pre['date'].max().date()}"
                      f"  末值 {pre['close'].iloc[-1]:.2f}（v2 起点 {v0:.2f}）", flush=True)
        out = base if pre is None or len(pre) == 0 else pd.concat(
            [pre[base.columns], base], ignore_index=True)
        out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        fp = os.path.join(OUT, f"{bk}.parquet")
        out.to_parquet(fp)
        print(f"   写出 {fp}  {len(out)} 行  {out['date'].min().date()} ~ {out['date'].max().date()}", flush=True)


if __name__ == "__main__":
    main()
