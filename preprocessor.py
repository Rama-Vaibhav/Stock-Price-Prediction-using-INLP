"""
Unified Preprocessing Pipeline
================================
Combines:
  1. Raw CSV clean-up & merging  (from preprocess_datasets.py)
  2. News cleaning + Price feature engineering  (from preprocess_datasets_final.py)

into a single, scalable, crash-resilient pipeline optimized for large datasets
(3+ years of data, ~100K+ articles).

Design Principles:
  • SCALABLE        — Chunked CSV reading (never loads full corpus into RAM at once).
  • CRASH-SAFE      — Each step writes intermediate outputs; re-run skips completed steps.
  • FAST            — multiprocessing for CPU-bound work (technical features),
                      vectorized pandas ops for text cleaning.
  • CONFIGURABLE    — Ticker/sector maps are dictionaries at the top; date window is a constant.
  • EXTENSIBLE      — Add new raw source CSVs to datasets/ and re-run; they auto-merge.

Pipeline:
  Phase 1 — Raw CSV Ingest & Merge
      • Scan datasets/ for *_news.csv source files.
      • Normalize columns to (date, title, news, url).
      • Per-file dedup + cross-file global dedup on URL.
      • Output: datasets/combined_market_news.csv

  Phase 2 — News Cleaning
      • HTML entity decode, ALSO READ removal, whitespace normalization.
      • Short article filtering (< 100 chars).
      • Source domain extraction.
      • Date parsing & date-range enforcement.
      • Output: datasets/processed/cleaned_news.csv

  Phase 3 — Price Cleaning & Feature Engineering
      • Holiday/stale row removal.
      • Sector tagging.
      • 13 technical indicators computed per ticker (multiprocessing).
      • Output: datasets/processed/cleaned_prices.csv,
               datasets/processed/price_features.csv

  Phase 4 — Calendar & News Volume
      • Trading calendar.
      • Daily news volume.
      • Output: datasets/processed/trading_calendar.csv,
               datasets/processed/daily_news_volume.csv

Date Window: January 1, 2023  →  February 28, 2026

Environment:
    conda activate ml

Usage:
    python unified_preprocessor.py              # full pipeline
    python unified_preprocessor.py --skip-merge  # skip Phase 1 (already merged)
    python unified_preprocessor.py --force       # re-run all steps even if outputs exist
"""

import os
import re
import html
import glob
import argparse
import warnings
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent   # Project root (now script is at root)
DATASET_DIR = BASE_DIR / "datasets"
OUT_DIR     = DATASET_DIR / "processed"

NEWS_COMBINED = DATASET_DIR / "combined_market_news.csv"
PRICES_RAW    = DATASET_DIR / "nifty50_historical_prices.csv"

# ── Date window  ──────────────────────────────────────────────────────────────
# Change these to extend the range (scraper + yfinance must match)
START_DATE = "2023-01-01"
END_DATE   = "2026-02-28"

# ── Target CSV columns ────────────────────────────────────────────────────────
TARGET_COLUMNS = ["date", "title", "news", "url"]

# ── Nifty 50 → Sector mapping (as of Dec 2025 composition) ────────────────────
# Add / remove tickers here when the index rebalances.
TICKER_SECTOR = {
    "ADANIENT": "Metals & Mining", "ADANIPORTS": "Services",
    "APOLLOHOSP": "Healthcare", "ASIANPAINT": "Consumer Durables",
    "AXISBANK": "Financial Services", "BAJAJ-AUTO": "Auto",
    "BAJFINANCE": "Financial Services", "BAJAJFINSV": "Financial Services",
    "BEL": "Capital Goods", "BHARTIARTL": "Telecom",
    "CIPLA": "Healthcare", "COALINDIA": "Oil Gas & Fuels",
    "DRREDDY": "Healthcare", "EICHERMOT": "Auto",
    "ETERNAL": "Consumer Services", "GRASIM": "Construction Materials",
    "HCLTECH": "IT", "HDFCBANK": "Financial Services",
    "HDFCLIFE": "Financial Services", "HINDALCO": "Metals & Mining",
    "HINDUNILVR": "FMCG", "ICICIBANK": "Financial Services",
    "INDIGO": "Services", "INFY": "IT",
    "ITC": "FMCG", "JIOFIN": "Financial Services",
    "JSWSTEEL": "Metals & Mining", "KOTAKBANK": "Financial Services",
    "LT": "Construction", "M&M": "Auto",
    "MARUTI": "Auto", "MAXHEALTH": "Healthcare",
    "NESTLEIND": "FMCG", "NTPC": "Power",
    "ONGC": "Oil Gas & Fuels", "POWERGRID": "Power",
    "RELIANCE": "Oil Gas & Fuels", "SBILIFE": "Financial Services",
    "SHRIRAMFIN": "Financial Services", "SBIN": "Financial Services",
    "SUNPHARMA": "Healthcare", "TCS": "IT",
    "TATACONSUM": "FMCG", "TATASTEEL": "Metals & Mining",
    "TECHM": "IT", "TITAN": "Consumer Durables",
    "TMPV": "Auto", "TRENT": "Consumer Services",
    "ULTRACEMCO": "Construction Materials", "WIPRO": "IT",
}


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: RAW CSV INGEST & MERGE
# ══════════════════════════════════════════════════════════════════════════════

def phase1_merge_raw_sources(force: bool = False):
    """
    Scan datasets/ for per-source news CSVs, normalize and deduplicate,
    write combined_market_news.csv.
    """
    print("\n" + "=" * 70)
    print("PHASE 1 — Raw CSV Ingest & Merge")
    print("=" * 70)

    if NEWS_COMBINED.exists() and not force:
        existing = _count_lines(NEWS_COMBINED) - 1  # minus header
        print(f"  combined_market_news.csv already exists ({existing:,} rows).")
        print(f"  Use --force to regenerate. Skipping Phase 1.")
        return

    # Find all per-source CSVs (pattern: *_news.csv, excluding the combined file)
    all_csvs = sorted(DATASET_DIR.glob("*_news.csv"))
    all_csvs = [f for f in all_csvs if f.name != "combined_market_news.csv"]

    if not all_csvs:
        print("  No *_news.csv source files found in datasets/. Nothing to merge.")
        return

    frames = []
    for filepath in all_csvs:
        print(f"\n  Processing: {filepath.name}")
        try:
            df = pd.read_csv(filepath, on_bad_lines="skip", engine="python")
            initial = len(df)
            print(f"    Read {initial:,} rows")

            if initial == 0:
                continue

            # ── Normalize column names ──
            col_lower = [str(c).strip().lower() for c in df.columns]
            has_header = any(k in c for c in col_lower
                            for k in ["date", "title", "news", "url"])

            if not has_header:
                # Headers are actually data — push them down
                df.loc[-1] = df.columns.tolist()
                df.index = df.index + 1
                df = df.sort_index()

            if len(df.columns) < 4:
                print(f"    ✗ Fewer than 4 columns — skipping.")
                continue

            # Keep first 4 columns → rename to standard
            df = df.iloc[:, :4]
            df.columns = TARGET_COLUMNS

            # ── Type coercion & cleaning ──
            df = df.dropna(subset=["title", "news", "date"])
            df["title"] = df["title"].astype(str).str.replace(r"[\n\r]+", ". ", regex=True).str.strip()
            df["news"]  = df["news"].astype(str).str.replace(r"[\n\r]+", ". ", regex=True).str.strip()

            # Drop stubs
            df = df[(df["title"].str.len() > 3) & (df["news"].str.len() > 5)]

            # ── Date parsing (coerce unparseable → NaT → drop) ──
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")

            # ── Per-file URL dedup ──
            df = df.drop_duplicates(subset=["url"], keep="first")

            print(f"    Cleaned: {len(df):,} rows (dropped {initial - len(df):,})")
            frames.append(df)

        except Exception as e:
            print(f"    ✗ Failed: {e}")

    if not frames:
        print("\n  No valid data to merge.")
        return

    # ── Global merge & dedup ──
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["url"], keep="first")

    # Sort newest-first
    combined["_dt"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values("_dt", ascending=False).drop(columns=["_dt"])
    combined.reset_index(drop=True, inplace=True)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(NEWS_COMBINED, index=False)

    print(f"\n  Merged: {before:,} → {len(combined):,} (global URL dedup)")
    print(f"  ✓ Saved: {NEWS_COMBINED.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: NEWS CLEANING
# ══════════════════════════════════════════════════════════════════════════════

# Pre-compile heavy regex once
_ALSO_READ_RE = re.compile(
    r"ALSO\s*READ\s*[:\-–]?\s*.{5,300}?(?=\.\s+[A-Z]|\.\s*$)",
    re.DOTALL,
)
_MULTI_SPACE_RE = re.compile(r"\s+")


def _clean_text_column(series: pd.Series) -> pd.Series:
    """Vectorized text cleaning: HTML decode → ALSO READ removal → whitespace."""
    s = series.apply(html.unescape)
    s = s.apply(lambda x: _ALSO_READ_RE.sub("", x))
    s = s.str.replace(_MULTI_SPACE_RE, " ", regex=True).str.strip()
    return s


def phase2_clean_news(force: bool = False):
    """Clean combined_market_news.csv → cleaned_news.csv."""
    print("\n" + "=" * 70)
    print("PHASE 2 — News Cleaning")
    print("=" * 70)

    out_path = OUT_DIR / "cleaned_news.csv"
    if out_path.exists() and not force:
        n = _count_lines(out_path) - 1
        print(f"  cleaned_news.csv exists ({n:,} rows). Use --force to redo. Skipping.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not NEWS_COMBINED.exists():
        print("  ✗ combined_market_news.csv not found. Run Phase 1 first.")
        return

    print("  Loading combined news...")
    # For very large files, use chunked reading to keep RAM in check
    file_size_mb = NEWS_COMBINED.stat().st_size / (1024 * 1024)
    if file_size_mb > 500:
        # Chunked processing for 500 MB+ files
        print(f"  Large file ({file_size_mb:.0f} MB) — processing in chunks...")
        _clean_news_chunked(out_path)
    else:
        df = pd.read_csv(NEWS_COMBINED)
        print(f"  Loaded {len(df):,} articles ({file_size_mb:.0f} MB)")
        df = _clean_news_dataframe(df)
        df.to_csv(out_path, index=False)
        print(f"  ✓ Saved {len(df):,} articles → {out_path.name}")


def _clean_news_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Full in-memory news cleaning."""
    print("  [1/5] Decoding HTML + removing ALSO READ...")
    df["title"] = _clean_text_column(df["title"])
    df["news"]  = _clean_text_column(df["news"])

    print("  [2/5] Removing short articles (< 100 chars)...")
    before = len(df)
    df = df[df["news"].str.len() >= 100].copy()
    print(f"         {before:,} → {len(df):,} (dropped {before - len(df):,})")

    print("  [3/5] Extracting source domain...")
    df["source"] = df["url"].str.extract(
        r"https?://(?:www\.)?([\w\-]+\.[\w]+)", expand=False
    )

    print("  [4/5] Parsing dates...")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["date"])

    # Enforce date window
    mask = (df["date"] >= START_DATE) & (df["date"] <= END_DATE)
    df = df[mask].copy()
    print(f"         Kept {len(df):,} in date range [{START_DATE} → {END_DATE}]")

    print("  [5/5] Sorting by date ascending...")
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    return df


def _clean_news_chunked(out_path: Path, chunk_size: int = 50_000):
    """Chunk-based cleaning for files too large to fit in RAM."""
    first_chunk = True
    total_rows = 0
    reader = pd.read_csv(NEWS_COMBINED, chunksize=chunk_size)

    for i, chunk in enumerate(reader, 1):
        print(f"  Chunk {i}: {len(chunk):,} rows...")
        cleaned = _clean_news_dataframe(chunk)
        cleaned.to_csv(
            out_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        total_rows += len(cleaned)
        first_chunk = False

    # Final sort (read → sort → overwrite)
    print(f"  Final sort of {total_rows:,} rows...")
    df = pd.read_csv(out_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(df):,} cleaned articles → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: PRICE CLEANING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def phase3_price_features(force: bool = False):
    """Clean OHLCV data + compute technical features per ticker using multiprocessing."""
    print("\n" + "=" * 70)
    print("PHASE 3 — Price Cleaning & Feature Engineering")
    print("=" * 70)

    out_clean = OUT_DIR / "cleaned_prices.csv"
    out_feat  = OUT_DIR / "price_features.csv"

    if out_feat.exists() and not force:
        n = _count_lines(out_feat) - 1
        print(f"  price_features.csv exists ({n:,} rows). Use --force to redo. Skipping.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PRICES_RAW.exists():
        print(f"  ✗ {PRICES_RAW.name} not found. Run yfinance extractor first.")
        return

    print("  Loading raw prices...")
    df = pd.read_csv(PRICES_RAW)
    print(f"  {len(df):,} rows, {df['Ticker'].nunique()} tickers")

    # ── Clean ──────────────────────────────────────────────────────────────
    print("  Parsing dates...")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    print("  Removing market-holiday / stale rows...")
    before = len(df)
    holiday_mask = (df["Volume"] == 0) & (df["Open"] == df["Close"])
    holiday_dates = df.loc[holiday_mask, "Date"].unique()
    df = df[~holiday_mask].copy()
    print(f"    Removed {before - len(df)} stale rows across "
          f"{len(holiday_dates)} date(s)")

    print("  Adding Sector column...")
    df["Sector"] = df["Ticker"].map(TICKER_SECTOR)
    unmapped = df["Sector"].isna().sum()
    if unmapped > 0:
        bad_tickers = df.loc[df["Sector"].isna(), "Ticker"].unique()
        print(f"    ⚠ {unmapped} rows have unmapped tickers: {list(bad_tickers)}")
        print(f"    → Add them to TICKER_SECTOR dict at the top of this file.")

    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Save cleaned prices
    df.to_csv(out_clean, index=False)
    print(f"  ✓ Saved {len(df):,} cleaned rows → {out_clean.name}")

    # ── Compute technical features with multiprocessing ────────────────────
    print("\n  Computing technical features per ticker (multiprocessing)...")
    tickers = sorted(df["Ticker"].unique())
    n_workers = min(mp.cpu_count(), len(tickers))
    print(f"    {len(tickers)} tickers across {n_workers} processes")

    # Split into per-ticker DataFrames
    ticker_dfs = [df[df["Ticker"] == t].copy() for t in tickers]

    with mp.Pool(n_workers) as pool:
        results = pool.map(_compute_features_for_ticker, ticker_dfs)

    features = pd.concat(results, ignore_index=True)
    features = features.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Round floats
    float_cols = features.select_dtypes(include=[np.floating]).columns
    features[float_cols] = features[float_cols].round(4)

    features.to_csv(out_feat, index=False)
    print(f"  ✓ Saved {len(features):,} rows × {len(features.columns)} cols → {out_feat.name}")


def _compute_features_for_ticker(tk: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 13 technical indicators for a single ticker.
    Designed to be called by multiprocessing.Pool.map().
    """
    tk = tk.sort_values("Date").copy()

    c = tk["Close"]
    h = tk["High"]
    l = tk["Low"]
    o = tk["Open"]
    v = tk["Volume"].astype(float)

    # Returns
    tk["Daily_Return_Pct"] = c.pct_change() * 100
    tk["Log_Return"]       = np.log(c / c.shift(1))

    # Intraday range
    tk["Intraday_Range_Pct"] = ((h - l) / o * 100).round(4)

    # Moving Averages
    tk["SMA_5"]   = c.rolling(5).mean()
    tk["SMA_10"]  = c.rolling(10).mean()
    tk["SMA_20"]  = c.rolling(20).mean()
    tk["EMA_12"]  = c.ewm(span=12, adjust=False).mean()

    # Volatility
    tk["Volatility_10"] = tk["Daily_Return_Pct"].rolling(10).std()

    # RSI 14
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta.clip(upper=0))
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    tk["RSI_14"] = 100 - (100 / (1 + rs))

    # VWAP
    tk["VWAP"] = (h + l + c) / 3

    # Volume SMA
    tk["Volume_SMA_10"] = v.rolling(10).mean()

    # Multi-day changes
    tk["Pct_Change_5d"]  = c.pct_change(periods=5)  * 100
    tk["Pct_Change_10d"] = c.pct_change(periods=10) * 100

    return tk


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: CALENDAR & NEWS VOLUME
# ══════════════════════════════════════════════════════════════════════════════

def phase4_calendar_and_volume(force: bool = False):
    """Build trading calendar and daily news volume summaries."""
    print("\n" + "=" * 70)
    print("PHASE 4 — Trading Calendar & News Volume")
    print("=" * 70)

    out_cal = OUT_DIR / "trading_calendar.csv"
    out_vol = OUT_DIR / "daily_news_volume.csv"

    if out_cal.exists() and out_vol.exists() and not force:
        print(f"  Both files exist. Use --force to redo. Skipping.")
        return

    clean_prices_path = OUT_DIR / "cleaned_prices.csv"
    clean_news_path   = OUT_DIR / "cleaned_news.csv"

    if not clean_prices_path.exists():
        print(f"  ✗ cleaned_prices.csv not found. Run Phase 3 first.")
        return
    if not clean_news_path.exists():
        print(f"  ✗ cleaned_news.csv not found. Run Phase 2 first.")
        return

    # ── Trading calendar ──
    print("  Building trading calendar...")
    prices = pd.read_csv(clean_prices_path, usecols=["Date", "Ticker"],
                          parse_dates=["Date"])
    cal = (prices.groupby("Date")
           .agg(Num_Tickers=("Ticker", "nunique"))
           .reset_index()
           .sort_values("Date"))
    cal["Day_Of_Week"] = cal["Date"].dt.day_name()
    cal.to_csv(out_cal, index=False)
    print(f"  ✓ {len(cal)} trading days → {out_cal.name}")

    # ── Daily news volume ──
    print("  Building daily news volume...")
    news = pd.read_csv(clean_news_path, usecols=["date", "title", "source"],
                        parse_dates=["date"])
    vol = (news.groupby("date")
           .agg(
               Num_Articles=("title", "count"),
               Sources=("source", lambda x: ", ".join(sorted(x.unique()))),
           )
           .reset_index()
           .rename(columns={"date": "Date"})
           .sort_values("Date"))
    vol.to_csv(out_vol, index=False)
    print(f"  ✓ {len(vol)} calendar days → {out_vol.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _count_lines(filepath: Path) -> int:
    """Count lines in a file without loading it into memory."""
    count = 0
    with open(filepath, "rb") as f:
        for _ in f:
            count += 1
    return count


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified preprocessing pipeline for stock forecasting project.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all steps even if output files already exist.",
    )
    global START_DATE, END_DATE

    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip Phase 1 (raw CSV merge). Useful when combined file is ready.",
    )
    parser.add_argument(
        "--start-date", default=None,
        help=f"Override start date (YYYY-MM-DD). Default: {START_DATE}",
    )
    parser.add_argument(
        "--end-date", default=None,
        help=f"Override end date (YYYY-MM-DD). Default: {END_DATE}",
    )
    args = parser.parse_args()
    if args.start_date:
        START_DATE = args.start_date
    if args.end_date:
        END_DATE = args.end_date

    print("=" * 70)
    print("  UNIFIED PREPROCESSING PIPELINE")
    print(f"  Date window: {START_DATE} → {END_DATE}")
    print(f"  CPU cores available: {mp.cpu_count()}")
    print("=" * 70)

    # Phase 1: Merge raw CSVs
    if not args.skip_merge:
        phase1_merge_raw_sources(force=args.force)
    else:
        print("\n  Phase 1 skipped (--skip-merge).")

    # Phase 2: Clean news
    phase2_clean_news(force=args.force)

    # Phase 3: Price cleaning + features
    phase3_price_features(force=args.force)

    # Phase 4: Calendar + news volume
    phase4_calendar_and_volume(force=args.force)

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("  PREPROCESSING COMPLETE — Summary")
    print("=" * 70)

    for f in sorted(OUT_DIR.glob("*.csv")):
        rows = _count_lines(f) - 1
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:35s}  {rows:>8,} rows  ({size_kb:>8,.0f} KB)")

    print(f"\n  Output directory: {OUT_DIR.relative_to(BASE_DIR)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
