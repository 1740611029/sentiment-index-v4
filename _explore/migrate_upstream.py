"""一次性迁移：把上游 v1/v2 个股缓存合并进 stocks_hist/，然后项目不再依赖外部目录。

精确复现当前 load_stock() 的拼接逻辑：
  1. v1 + v2 → cache_df（v2 优先，与 load_stock 的 concat + keep="last" 一致）
  2. 已在 stocks_hist/ 的股票：
     - 重叠期常数比率稳定（CV < 0.02）→ hist 对齐到 cache_df 基准后拼接
     - 比率不稳定 → 保存 cache_df（与 load_stock 比率不稳定时返回 cache_df 一致）
  3. 不在 stocks_hist/ 的股票：直接保存 cache_df

用法：
  python _explore/migrate_upstream.py --dry-run   # 预览
  python _explore/migrate_upstream.py              # 执行
"""
from __future__ import annotations

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1_DIR = r"D:\情绪指标\data\cache\stocks"
V2_DIR = r"D:\情绪指标2\data\cache\stocks"
HIST_DIR = os.path.join(ROOT, "data", "cache", "stocks_hist")

CV_THRESHOLD = 0.02
MIN_OVERLAP = 30


def ratio_to(base: pd.Series, other: pd.Series, min_rows: int = MIN_OVERLAP) -> float | None:
    """重叠期常数比率 other → base，与 data._ratio_to 完全一致。"""
    j = pd.concat([base.rename("a"), other.rename("b")], axis=1).dropna()
    if len(j) < min_rows:
        return None
    r = (j["a"] / j["b"]).replace([np.inf, -np.inf], np.nan).dropna()
    if not len(r) or not r.mean() or (r.std() / abs(r.mean())) > CV_THRESHOLD:
        return None
    return float(r.median())


def merge_v1_v2(v1_path: str | None, v2_path: str | None) -> pd.DataFrame:
    """合并 v1 + v2，v2 优先（与 load_stock 行为一致）。"""
    frames = []
    for p in (v1_path, v2_path):
        if p and os.path.exists(p):
            d = pd.read_parquet(p)
            d["date"] = pd.to_datetime(d["date"])
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def process_stock(code: str, cache_df: pd.DataFrame, hist_path: str | None,
                  dry_run: bool) -> str:
    """处理单只股票，返回操作描述。"""
    if hist_path is None or not os.path.exists(hist_path):
        if not dry_run:
            cache_df.to_parquet(os.path.join(HIST_DIR, f"{code}.parquet"), index=False)
        return "added"

    hist_df = pd.read_parquet(hist_path)
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    hist_df = hist_df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    if len(cache_df) == 0:
        return "keep_hist"

    lo = max(hist_df["date"].min(), cache_df["date"].min())
    hi = min(hist_df["date"].max(), cache_df["date"].max())

    if lo >= hi:
        # 无重叠 → 当前 load_stock 的 hist-vs-cache ratio 为 None → 返回 cache_df
        # 迁移必须保存 cache_df 才能复现同样行为
        if not dry_run:
            cache_df.to_parquet(os.path.join(HIST_DIR, f"{code}.parquet"), index=False)
        return "cache_only_no_overlap"

    A = hist_df[(hist_df["date"] >= lo) & (hist_df["date"] <= hi)].set_index("date")["close"].astype(float)
    B = cache_df[(cache_df["date"] >= lo) & (cache_df["date"] <= hi)].set_index("date")["close"].astype(float)
    j = pd.concat([A.rename("h"), B.rename("v")], axis=1).dropna()

    if len(j) < MIN_OVERLAP:
        if not dry_run:
            cache_df.to_parquet(os.path.join(HIST_DIR, f"{code}.parquet"), index=False)
        return "cache_only_short_overlap"

    r = (j["v"] / j["h"]).replace([np.inf, -np.inf], np.nan).dropna()
    if not len(r) or not r.mean():
        if not dry_run:
            cache_df.to_parquet(os.path.join(HIST_DIR, f"{code}.parquet"), index=False)
        return "cache_only_bad_ratio"

    cv = r.std() / abs(r.mean())
    if cv >= CV_THRESHOLD:
        if not dry_run:
            cache_df.to_parquet(os.path.join(HIST_DIR, f"{code}.parquet"), index=False)
        return f"cache_only_unstable_cv{cv:.4f}"

    med = float(r.median())

    if not dry_run:
        pre = hist_df[hist_df["date"] < cache_df["date"].min()].copy()
        if abs(med - 1.0) > 1e-6:
            for c in ("open", "high", "low", "close"):
                if c in pre.columns:
                    pre[c] = pre[c] * med
        out = pd.concat([pre, cache_df], ignore_index=True)
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        out.to_parquet(os.path.join(HIST_DIR, f"{code}.parquet"), index=False)

    return f"merged_ratio{med:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只报告不写入")
    args = ap.parse_args()

    if not os.path.isdir(V1_DIR):
        print(f"ERROR: v1 目录不存在: {V1_DIR}")
        sys.exit(1)
    if not os.path.isdir(V2_DIR):
        print(f"ERROR: v2 目录不存在: {V2_DIR}")
        sys.exit(1)

    v1_files = {os.path.splitext(os.path.basename(p))[0].zfill(6): p
                for p in glob.glob(os.path.join(V1_DIR, "*.parquet"))}
    v2_files = {os.path.splitext(os.path.basename(p))[0].zfill(6): p
                for p in glob.glob(os.path.join(V2_DIR, "*.parquet"))}
    hist_files = {os.path.splitext(os.path.basename(p))[0].zfill(6): p
                  for p in glob.glob(os.path.join(HIST_DIR, "*.parquet"))}

    all_codes = sorted(set(v1_files) | set(v2_files))
    print(f"v1: {len(v1_files)}, v2: {len(v2_files)}, union: {len(all_codes)}")
    print(f"stocks_hist: {len(hist_files)}")
    print(f"{'[DRY RUN] ' if args.dry_run else ''}处理 {len(all_codes)} 只股票...")
    print()

    stats = {"merged_ratio": 0, "added": 0, "cache_only_no_overlap": 0,
             "cache_only_short_overlap": 0, "cache_only_bad_ratio": 0,
             "cache_only_unstable": 0, "keep_hist": 0}
    unstable_codes = []

    for i, code in enumerate(all_codes):
        v1p = v1_files.get(code)
        v2p = v2_files.get(code)
        cache_df = merge_v1_v2(v1p, v2p)
        if len(cache_df) == 0:
            continue

        hist_path = hist_files.get(code)
        result = process_stock(code, cache_df, hist_path, args.dry_run)

        if result.startswith("merged_ratio"):
            stats["merged_ratio"] += 1
        elif result == "added":
            stats["added"] += 1
        elif result == "cache_only_no_overlap":
            stats["cache_only_no_overlap"] += 1
        elif result == "cache_only_short_overlap":
            stats["cache_only_short_overlap"] += 1
        elif result == "cache_only_bad_ratio":
            stats["cache_only_bad_ratio"] += 1
        elif result.startswith("cache_only_unstable"):
            stats["cache_only_unstable"] += 1
            unstable_codes.append(code)
        elif result == "keep_hist":
            stats["keep_hist"] += 1

        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(all_codes)}")

    print()
    print("=" * 60)
    print(f"迁移{'预览' if args.dry_run else '完成'}：")
    print(f"  合并（比率稳定）: {stats['merged_ratio']}")
    print(f"  新增（不在 hist）: {stats['added']}")
    print(f"  仅缓存（无重叠）: {stats['cache_only_no_overlap']}")
    print(f"  仅缓存（重叠不足）: {stats['cache_only_short_overlap']}")
    print(f"  仅缓存（比率异常）: {stats['cache_only_bad_ratio']}")
    print(f"  仅缓存（比率不稳）: {stats['cache_only_unstable']}")
    print(f"  保留 hist（无上游）: {stats['keep_hist']}")

    if unstable_codes:
        print(f"\n比率不稳的股票（{len(unstable_codes)} 只）: {unstable_codes[:20]}")

    if not args.dry_run:
        new_count = len(glob.glob(os.path.join(HIST_DIR, "*.parquet")))
        print(f"\nstocks_hist 文件数: {len(hist_files)} → {new_count}")


if __name__ == "__main__":
    main()
