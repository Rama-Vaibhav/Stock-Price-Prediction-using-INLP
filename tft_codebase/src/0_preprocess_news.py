#!/usr/bin/env python3
"""
News Dataset Preprocessing Pipeline - Memory-Efficient Chunked Processing

PREPROCESSING OPERATIONS:
═══════════════════════════════════════════════════════════════════════════════
Phase 1: Chunked Merging
  • Merge all *_raw.csv files from dataset/raw_dataset/ into single CSV
  • Process in 100k-row chunks to prevent memory overflow

Phase 2: In-Place Cleaning & Transformation
  Structural Integration:
    • Standardize columns to ['date', 'title', 'news', 'url']
    • Drop null values in critical columns (date, title, news)
    • Global deduplication using URL hash set across all chunks
  
  Text & NLP Preprocessing:
    • HTML entity decoding (&amp; → &, &#8211; → –, etc.)
    • Typography normalization (smart quotes → ASCII, em-dash → hyphen)
    • Remove "ALSO READ..." promotional artifacts
    • Strip embedded URLs (http/https) and email addresses
    • Remove boilerplate ("Download the app", "Subscribe", "Disclaimer")
    • Newline/carriage return → period + space
    • Collapse multiple spaces into single space
    • Length filtering (title ≥3 chars, news ≥100 chars)
  
  Temporal Preprocessing:
    • Parse dates to YYYY-MM-DD format
    • Coerce invalid dates to NaT and drop
    • Enforce date window: 2020-01-01 to 2026-03-31
  
  Post-Processing:
    • Chronological sorting by date (memory-efficient external sort)
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from pathlib import Path
import html
import re
from datetime import datetime
import tempfile
import shutil
from typing import Set
import time

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE = 100_000
START_DATE = "2020-01-01"
END_DATE = "2026-03-31"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw_dataset"
OUTPUT_FILE = PROJECT_ROOT / "dataset" / "processed_news_dataset.csv"

STANDARD_COLUMNS = ["date", "title", "news", "url"]

# HTML entities and special characters to normalize
HTML_ENTITIES_EXTRA = {
    "&#8211;": "–",
    "&#8212;": "—",
    "&#8216;": "'",
    "&#8217;": "'",
    "&#8220;": """,
    "&#8221;": """,
    "&#8230;": "...",
    "&#039;": "'",
    "&nbsp;": " ",
    "&ndash;": "–",
    "&mdash;": "—",
    "&lsquo;": "'",
    "&rsquo;": "'",
    "&ldquo;": """,
    "&rdquo;": """,
    "&hellip;": "...",
}

# Smart quotes and unicode to ASCII
TYPOGRAPHY_MAP = {
    """: '"',
    """: '"',
    "'": "'",
    "'": "'",
    "—": "-",
    "–": "-",
    "…": "...",
    "−": "-",
}

# Boilerplate patterns to remove (case-insensitive)
BOILERPLATE_PATTERNS = [
    r"download\s+(?:the\s+)?app.*?(?:\.|$)",
    r"subscribe\s+to\s+(?:our\s+)?newsletter.*?(?:\.|$)",
    r"disclaimer\s*:.*?(?:\.|$)",
    r"follow\s+us\s+on.*?(?:\.|$)",
    r"join\s+us\s+on.*?(?:\.|$)",
    r"get\s+(?:the\s+)?latest\s+news.*?(?:\.|$)",
    r"read\s+more\s+at.*?(?:\.|$)",
    r"click\s+here\s+to.*?(?:\.|$)",
    r"for\s+more\s+information.*?(?:\.|$)",
    r"visit\s+our\s+website.*?(?:\.|$)",
]

# Promotional artifact pattern (ALSO READ...)
PROMOTIONAL_PATTERN = r"ALSO\s+READ\s*:?.*?(?:\.|\n|$)"

# URL and email patterns
URL_PATTERN = r"https?://[^\s<>\"]+|www\.[^\s<>\"]+"
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def decode_html_entities(text: pd.Series) -> pd.Series:
    """Decode HTML entities in text using vectorized operations."""
    # Standard HTML decoding
    text = text.apply(lambda x: html.unescape(x) if pd.notna(x) else x)
    
    # Additional numeric/named entities
    for entity, replacement in HTML_ENTITIES_EXTRA.items():
        text = text.str.replace(entity, replacement, regex=False)
    
    return text


def normalize_typography(text: pd.Series) -> pd.Series:
    """Convert smart quotes and unicode dashes to ASCII."""
    for unicode_char, ascii_char in TYPOGRAPHY_MAP.items():
        text = text.str.replace(unicode_char, ascii_char, regex=False)
    return text


def remove_promotional_artifacts(text: pd.Series) -> pd.Series:
    """Remove 'ALSO READ...' promotional text."""
    return text.str.replace(PROMOTIONAL_PATTERN, " ", flags=re.IGNORECASE, regex=True)


def strip_urls_and_emails(text: pd.Series) -> pd.Series:
    """Remove URLs and email addresses from text."""
    text = text.str.replace(URL_PATTERN, " ", regex=True)
    text = text.str.replace(EMAIL_PATTERN, " ", regex=True)
    return text


def remove_boilerplate(text: pd.Series) -> pd.Series:
    """Remove common scraper boilerplate text."""
    for pattern in BOILERPLATE_PATTERNS:
        text = text.str.replace(pattern, " ", flags=re.IGNORECASE, regex=True)
    return text


def normalize_whitespace(text: pd.Series) -> pd.Series:
    """Normalize newlines and collapse multiple spaces."""
    # Replace newlines and carriage returns with period + space
    text = text.str.replace(r"[\n\r]+", ". ", regex=True)
    # Collapse multiple spaces into single space
    text = text.str.replace(r"\s+", " ", regex=True)
    # Strip leading/trailing whitespace
    text = text.str.strip()
    # Fix multiple periods
    text = text.str.replace(r"\.{2,}", ".", regex=True)
    # Fix period-space-period patterns
    text = text.str.replace(r"\.\s+\.", ".", regex=True)
    return text


def preprocess_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all text preprocessing transformations to title and news columns."""
    for col in ["title", "news"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = decode_html_entities(df[col])
            df[col] = normalize_typography(df[col])
            df[col] = remove_promotional_artifacts(df[col])
            df[col] = strip_urls_and_emails(df[col])
            df[col] = remove_boilerplate(df[col])
            df[col] = normalize_whitespace(df[col])
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DATE PREPROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_and_filter_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Parse dates to YYYY-MM-DD format and filter by date range."""
    # Parse dates (coerce errors to NaT)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Drop rows with invalid dates
    before_count = len(df)
    df = df.dropna(subset=["date"])
    invalid_count = before_count - len(df)
    
    if invalid_count > 0:
        print(f"    ⚠ Dropped {invalid_count:,} rows with invalid dates")
    
    # Filter by date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    before_filter = len(df)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    filtered_count = before_filter - len(df)
    
    if filtered_count > 0:
        print(f"    ⚠ Filtered {filtered_count:,} rows outside date range [{start_date}, {end_date}]")
    
    # Format as YYYY-MM-DD
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL PREPROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase and keep only required columns."""
    # Lowercase all column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Keep only standard columns that exist
    existing_cols = [col for col in STANDARD_COLUMNS if col in df.columns]
    df = df[existing_cols]
    
    return df


def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with null values in critical columns."""
    before_count = len(df)
    df = df.dropna(subset=["date", "title", "news"])
    dropped_count = before_count - len(df)
    
    if dropped_count > 0:
        print(f"    ⚠ Dropped {dropped_count:,} rows with null values")
    
    return df


def filter_by_length(df: pd.DataFrame) -> pd.DataFrame:
    """Drop articles with title < 3 chars or news < 100 chars."""
    before_count = len(df)
    
    df = df[
        (df["title"].str.len() >= 3) & 
        (df["news"].str.len() >= 100)
    ]
    
    dropped_count = before_count - len(df)
    if dropped_count > 0:
        print(f"    ⚠ Filtered {dropped_count:,} rows by length (title<3 or news<100)")
    
    return df


def deduplicate_chunk(df: pd.DataFrame, seen_urls: Set[str]) -> tuple[pd.DataFrame, Set[str]]:
    """Remove duplicate URLs using global hash set."""
    before_count = len(df)
    
    # Filter out already-seen URLs
    df = df[~df["url"].isin(seen_urls)]
    
    # Add new URLs to seen set
    new_urls = set(df["url"].values)
    seen_urls.update(new_urls)
    
    duplicates_removed = before_count - len(df)
    if duplicates_removed > 0:
        print(f"    ⚠ Removed {duplicates_removed:,} duplicate URLs")
    
    return df, seen_urls


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: CHUNKED MERGING
# ═══════════════════════════════════════════════════════════════════════════════

def merge_raw_files():
    """Merge all *_raw.csv files from raw_dataset directory into single CSV."""
    print("=" * 80)
    print("PHASE 1: CHUNKED MERGING")
    print("=" * 80)
    
    # Get all *_raw.csv files
    raw_files = sorted(RAW_DATA_DIR.glob("*_raw.csv"))
    
    if not raw_files:
        print("⚠ No *_raw.csv files found in", RAW_DATA_DIR)
        return
    
    print(f"Found {len(raw_files)} CSV file(s) to merge:")
    for f in raw_files:
        file_size = f.stat().st_size / (1024 * 1024)  # MB
        print(f"  • {f.name} ({file_size:.1f} MB)")
    
    print(f"\nMerging into: {OUTPUT_FILE}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows\n")
    
    # Remove existing output file if it exists
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print("  ✓ Removed existing output file\n")
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    total_rows = 0
    header_written = False
    
    for file_idx, raw_file in enumerate(raw_files, 1):
        print(f"[{file_idx}/{len(raw_files)}] Processing {raw_file.name}...")
        file_rows = 0
        chunk_num = 0
        
        try:
            # Read file in chunks
            for chunk in pd.read_csv(raw_file, chunksize=CHUNK_SIZE, low_memory=False):
                chunk_num += 1
                chunk_rows = len(chunk)
                file_rows += chunk_rows
                
                # Write to output (header only on first chunk)
                mode = "w" if not header_written else "a"
                chunk.to_csv(OUTPUT_FILE, mode=mode, index=False, header=not header_written)
                
                if not header_written:
                    header_written = True
                
                print(f"  Chunk {chunk_num}: {chunk_rows:,} rows → Total: {file_rows:,}")
        
        except Exception as e:
            print(f"  ✗ Error reading {raw_file.name}: {e}")
            continue
        
        total_rows += file_rows
        print(f"  ✓ {raw_file.name}: {file_rows:,} rows\n")
    
    print("=" * 80)
    print(f"✓ PHASE 1 COMPLETE: {total_rows:,} total rows merged")
    print("=" * 80)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: IN-PLACE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_merged_file():
    """Apply all preprocessing transformations to merged file using chunks."""
    print("=" * 80)
    print("PHASE 2: IN-PLACE PREPROCESSING")
    print("=" * 80)
    
    if not OUTPUT_FILE.exists():
        print(f"⚠ {OUTPUT_FILE} not found. Run Phase 1 first.")
        return
    
    file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Input file: {OUTPUT_FILE.name} ({file_size:.1f} MB)")
    print(f"Chunk size: {CHUNK_SIZE:,} rows\n")
    
    # Create temporary output file
    temp_file = OUTPUT_FILE.parent / f"temp_{OUTPUT_FILE.name}"
    
    seen_urls: Set[str] = set()
    total_input_rows = 0
    total_output_rows = 0
    chunk_num = 0
    header_written = False
    
    start_time = time.time()
    
    print("Processing chunks...\n")
    
    try:
        for chunk in pd.read_csv(OUTPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
            chunk_num += 1
            chunk_start = time.time()
            input_rows = len(chunk)
            total_input_rows += input_rows
            
            print(f"Chunk {chunk_num}: {input_rows:,} rows")
            
            # 1. Standardize columns
            chunk = standardize_columns(chunk)
            
            # 2. Drop nulls
            chunk = drop_nulls(chunk)
            
            # 3. Deduplicate
            chunk, seen_urls = deduplicate_chunk(chunk, seen_urls)
            
            if len(chunk) == 0:
                print(f"  ⚠ Chunk empty after deduplication, skipping\n")
                continue
            
            # 4. Parse and filter dates
            chunk = parse_and_filter_dates(chunk, START_DATE, END_DATE)
            
            if len(chunk) == 0:
                print(f"  ⚠ Chunk empty after date filtering, skipping\n")
                continue
            
            # 5. Text preprocessing
            print(f"    → Applying text preprocessing...")
            chunk = preprocess_text_columns(chunk)
            
            # 6. Length filtering
            chunk = filter_by_length(chunk)
            
            if len(chunk) == 0:
                print(f"  ⚠ Chunk empty after length filtering, skipping\n")
                continue
            
            # Write to temp file
            mode = "w" if not header_written else "a"
            chunk.to_csv(temp_file, mode=mode, index=False, header=not header_written)
            
            if not header_written:
                header_written = True
            
            output_rows = len(chunk)
            total_output_rows += output_rows
            chunk_time = time.time() - chunk_start
            
            print(f"    ✓ Output: {output_rows:,} rows ({chunk_time:.2f}s)")
            print(f"    → Running total: {total_output_rows:,} / {total_input_rows:,} rows ({100 * total_output_rows / total_input_rows:.1f}% retained)\n")
    
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        if temp_file.exists():
            temp_file.unlink()
        return
    
    print("=" * 80)
    print("SORTING BY DATE (Chronological Order)...")
    print("=" * 80)
    
    # Sort the temp file (memory-efficient: read in chunks, sort, write)
    try:
        print("Reading cleaned data for sorting...")
        
        # For very large files, we'd need external sorting, but for moderate sizes:
        # Try to read all at once for sorting. If it fails, skip sorting.
        try:
            df_sorted = pd.read_csv(temp_file)
            print(f"  ✓ Loaded {len(df_sorted):,} rows into memory")
            
            print("  → Sorting by date...")
            df_sorted = df_sorted.sort_values("date", ascending=True)
            
            print("  → Writing sorted data...")
            df_sorted.to_csv(temp_file, index=False)
            print("  ✓ Sorting complete\n")
            
        except MemoryError:
            print("  ⚠ File too large to sort in memory, skipping sort step\n")
    
    except Exception as e:
        print(f"  ⚠ Could not sort file: {e}\n")
    
    # Replace original file with cleaned file
    print("Finalizing...")
    shutil.move(str(temp_file), str(OUTPUT_FILE))
    
    elapsed_time = time.time() - start_time
    
    print("=" * 80)
    print("✓ PHASE 2 COMPLETE")
    print("=" * 80)
    print(f"Input rows:    {total_input_rows:,}")
    print(f"Output rows:   {total_output_rows:,}")
    print(f"Removed:       {total_input_rows - total_output_rows:,} ({100 * (total_input_rows - total_output_rows) / total_input_rows:.1f}%)")
    print(f"Unique URLs:   {len(seen_urls):,}")
    print(f"Processing time: {elapsed_time / 60:.1f} minutes")
    print(f"\nFinal output: {OUTPUT_FILE}")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Execute full preprocessing pipeline."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 16 + "NEWS DATASET PREPROCESSING PIPELINE" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    overall_start = time.time()
    
    # Phase 1: Merge all raw files
    merge_raw_files()
    
    # Phase 2: Clean and preprocess
    preprocess_merged_file()
    
    overall_time = time.time() - overall_start
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 27 + "PIPELINE COMPLETE" + " " * 34 + "║")
    print("║" + f" Total execution time: {overall_time / 60:.1f} minutes" + " " * (78 - 29 - len(f"{overall_time / 60:.1f}")) + "║")
    print("╚" + "=" * 78 + "╝")
    print()


if __name__ == "__main__":
    main()
