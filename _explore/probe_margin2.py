"""第二轮探测：必须找到覆盖 2015-2016 的杠杆/投降类数据，否则不能采（铁律）。"""
import akshare as ak
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def show(name, fn):
    try:
        d = fn()
        print("=" * 118)
        print(f"✓ {name}   shape={d.shape}")
        print("  列：", list(d.columns))
        print(d.head(3).to_string())
        print("  ...")
        print(d.tail(3).to_string())
        # 找日期列
        for c in d.columns:
            if "date" in c.lower() or "日期" in c or "时间" in c:
                try:
                    s = pd.to_datetime(d[c].astype(str), errors="coerce")
                    print(f"  >> 日期列 [{c}]  范围 {s.min()} ~ {s.max()}   唯一值 {s.nunique()}")
                except Exception:
                    pass
                break
    except Exception as e:
        print("=" * 118)
        print(f"✗ {name}   {type(e).__name__}: {e}")


show("macro_china_market_margin_sh", ak.macro_china_market_margin_sh)
show("macro_china_market_margin_sz", ak.macro_china_market_margin_sz)
show("stock_a_below_net_asset_statistics", ak.stock_a_below_net_asset_statistics)
show("stock_account_statistics_em", lambda: ak.stock_account_statistics_em(symbol="新增投资者数量"))
