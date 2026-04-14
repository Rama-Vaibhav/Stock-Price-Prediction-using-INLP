#!/usr/bin/env python3
"""
Nifty Stock Price Scraper from Yahoo Finance
=============================================
Downloads historical stock prices for Nifty 50/100/200 constituents.

Features:
* Fetches current constituent lists from NSE India
* Handles ticker symbol changes/renamings
* Parallel processing using all 12 CPU threads
* Checkpoint/resume capability
* Rate limiting to avoid Yahoo Finance blocks
* Comprehensive error handling

Usage:
    conda activate ml
    cd src/scrapers
    python nifty_yfinance_scraper.py --index nifty50
    python nifty_yfinance_scraper.py --index nifty100
    python nifty_yfinance_scraper.py --index nifty200
    python nifty_yfinance_scraper.py --index all
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Date range for historical data
START_DATE = "2020-01-01"
END_DATE = "2026-03-31"

# Parallel processing (utilize all 12 threads)
MAX_WORKERS = 12

# Rate limiting (requests per second to avoid Yahoo Finance blocks)
RATE_LIMIT_DELAY = 0.5  # seconds between requests

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Output directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "stock_dataset"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoint"

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nifty_scraper")

# ═══════════════════════════════════════════════════════════════════════════════
# TICKER SYMBOL MAPPINGS (handle renamings/mergers)
# ═══════════════════════════════════════════════════════════════════════════════

# Known ticker symbol changes (old_symbol -> new_symbol)
TICKER_CHANGES = {
    "YESBANK": "YESBANK.NS",
    "VEDL": "VEDL.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    # Add more as needed
}

# Special case mappings for companies with name changes
COMPANY_NAME_FIXES = {
    "Adani Ports and Special Economic Zone": "ADANIPORTS",
    "Adani Ports & SEZ": "ADANIPORTS",
    "Bajaj Finance": "BAJFINANCE",
    "Bajaj Finserv": "BAJAJFINSV",
    "Bharti Airtel": "BHARTIARTL",
    "HDFC Bank": "HDFCBANK",
    "ICICI Bank": "ICICIBANK",
    "State Bank of India": "SBIN",
    "Tata Consultancy Services": "TCS",
    "Tata Steel": "TATASTEEL",
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Stock:
    symbol: str
    company: str
    index: str  # nifty50, nifty100, or nifty200

@dataclass
class DownloadStats:
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float = field(default_factory=time.time)

    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        m, s = divmod(int(elapsed), 60)
        return (
            f"[{m:02d}:{s:02d}] total={self.total} "
            f"success={self.success} failed={self.failed} skipped={self.skipped}"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# NIFTY CONSTITUENT FETCHERS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_nifty50_constituents() -> list[Stock]:
    """Fetch current Nifty 50 constituent list from NSE India."""
    log.info("Fetching Nifty 50 constituents from NSE...")
    
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        # NSE requires a session with cookies
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        time.sleep(1)
        
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        stocks = []
        for item in data.get("data", []):
            symbol = item.get("symbol", "").strip()
            company = item.get("meta", {}).get("companyName", "") or item.get("symbol", "")
            
            if symbol and symbol not in ["NIFTY 50", "Nifty 50"]:
                # Add .NS suffix for Yahoo Finance
                yf_symbol = f"{symbol}.NS"
                stocks.append(Stock(yf_symbol, company, "nifty50"))
        
        log.info(f"✓ Fetched {len(stocks)} Nifty 50 stocks")
        return stocks
    
    except Exception as e:
        log.error(f"Failed to fetch Nifty 50 from NSE API: {e}")
        log.info("Falling back to hardcoded Nifty 50 list...")
        return get_fallback_nifty50()

def fetch_nifty100_constituents() -> list[Stock]:
    """Fetch current Nifty 100 constituent list."""
    log.info("Fetching Nifty 100 constituents from NSE...")
    
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        time.sleep(1)
        
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        stocks = []
        for item in data.get("data", []):
            symbol = item.get("symbol", "").strip()
            company = item.get("meta", {}).get("companyName", "") or item.get("symbol", "")
            
            if symbol and symbol not in ["NIFTY 100", "Nifty 100"]:
                yf_symbol = f"{symbol}.NS"
                stocks.append(Stock(yf_symbol, company, "nifty100"))
        
        log.info(f"✓ Fetched {len(stocks)} Nifty 100 stocks")
        return stocks
    
    except Exception as e:
        log.error(f"Failed to fetch Nifty 100: {e}")
        return []

def fetch_nifty200_constituents() -> list[Stock]:
    """Fetch current Nifty 200 constituent list."""
    log.info("Fetching Nifty 200 constituents from NSE...")
    
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20200"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        time.sleep(1)
        
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        stocks = []
        for item in data.get("data", []):
            symbol = item.get("symbol", "").strip()
            company = item.get("meta", {}).get("companyName", "") or item.get("symbol", "")
            
            if symbol and symbol not in ["NIFTY 200", "Nifty 200"]:
                yf_symbol = f"{symbol}.NS"
                stocks.append(Stock(yf_symbol, company, "nifty200"))
        
        log.info(f"✓ Fetched {len(stocks)} Nifty 200 stocks")
        return stocks
    
    except Exception as e:
        log.error(f"Failed to fetch Nifty 200: {e}")
        return []

def get_fallback_nifty50() -> list[Stock]:
    """Hardcoded Nifty 50 list as fallback (current as of Apr 2024)."""
    symbols = [
        "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE",
        "BAJAJFINSV", "BHARTIARTL", "BPCL", "BRITANNIA", "CIPLA",
        "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM",
        "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
        "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC",
        "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI",
        "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
        "SBILIFE", "SBIN", "SUNPHARMA", "TATACONSUM", "TATAMOTORS",
        "TATASTEEL", "TCS", "TECHM", "TITAN", "ULTRACEMCO",
        "WIPRO", "APOLLOHOSP", "ADANIENT", "LTIM", "TRENT"
    ]
    return [Stock(f"{s}.NS", s, "nifty50") for s in symbols]

# ═══════════════════════════════════════════════════════════════════════════════
# TICKER SYMBOL VALIDATION & CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_and_fix_ticker(stock: Stock) -> Optional[Stock]:
    """Validate ticker exists on Yahoo Finance, handle renamings."""
    
    # Try original symbol first
    try:
        ticker = yf.Ticker(stock.symbol)
        info = ticker.info
        
        # Check if data exists
        if info and "symbol" in info:
            return stock
    except Exception:
        pass
    
    # Check if there's a known mapping
    base_symbol = stock.symbol.replace(".NS", "")
    if base_symbol in TICKER_CHANGES:
        new_symbol = TICKER_CHANGES[base_symbol]
        if not new_symbol.endswith(".NS"):
            new_symbol += ".NS"
        
        log.info(f"  ↻ Ticker renamed: {stock.symbol} → {new_symbol}")
        return Stock(new_symbol, stock.company, stock.index)
    
    # Try alternative suffixes
    alternatives = [
        stock.symbol,
        base_symbol + ".NS",
        base_symbol + ".BO",  # Bombay Stock Exchange
    ]
    
    for alt_symbol in alternatives:
        try:
            ticker = yf.Ticker(alt_symbol)
            info = ticker.info
            if info and "symbol" in info:
                if alt_symbol != stock.symbol:
                    log.info(f"  ↻ Using alternative: {stock.symbol} → {alt_symbol}")
                return Stock(alt_symbol, stock.company, stock.index)
        except Exception:
            continue
    
    log.warning(f"  ✗ Could not validate ticker: {stock.symbol} ({stock.company})")
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def download_stock_data(stock: Stock, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download historical data for a single stock with retry logic."""
    
    for attempt in range(MAX_RETRIES):
        try:
            ticker = yf.Ticker(stock.symbol)
            df = ticker.history(start=start, end=end, auto_adjust=False)
            
            if df.empty:
                log.warning(f"  ⚠ No data for {stock.symbol} ({stock.company})")
                return None
            
            # Add metadata columns
            df["Symbol"] = stock.symbol
            df["Company"] = stock.company
            df["Index"] = stock.index
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Reorder columns (include Adj Close)
            cols = ["Date", "Symbol", "Company", "Index", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
            df = df[cols]
            
            time.sleep(RATE_LIMIT_DELAY)
            return df
        
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                log.warning(f"  ⚠ Attempt {attempt + 1} failed for {stock.symbol}: {e}")
                time.sleep(RETRY_DELAY)
            else:
                log.error(f"  ✗ Failed to download {stock.symbol} after {MAX_RETRIES} attempts: {e}")
                return None
    
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(index_name: str) -> set[str]:
    """Load list of already-downloaded symbols from checkpoint."""
    checkpoint_file = CHECKPOINT_DIR / f"{index_name}_checkpoint.json"
    
    if not checkpoint_file.exists():
        return set()
    
    try:
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
            return set(data.get("completed", []))
    except Exception as e:
        log.warning(f"Could not load checkpoint: {e}")
        return set()

def save_checkpoint(index_name: str, completed_symbols: set[str]):
    """Save checkpoint of completed downloads."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"{index_name}_checkpoint.json"
    
    try:
        with open(checkpoint_file, "w") as f:
            json.dump({"completed": list(completed_symbols)}, f, indent=2)
    except Exception as e:
        log.error(f"Could not save checkpoint: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCRAPER
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_index(index_name: str, stocks: list[Stock], start_date: str, end_date: str):
    """Download historical data for all stocks in an index."""
    
    if not stocks:
        log.error(f"No stocks found for {index_name}")
        return
    
    log.info(f"═══ Starting {index_name.upper()} Download ═══")
    log.info(f"  Stocks: {len(stocks)}")
    log.info(f"  Date Range: {start_date} to {end_date}")
    log.info(f"  Workers: {MAX_WORKERS}")
    
    # Load checkpoint
    completed = load_checkpoint(index_name)
    if completed:
        log.info(f"  ↻ Resuming: {len(completed)} stocks already downloaded")
    
    # Validate tickers
    log.info("Validating ticker symbols...")
    valid_stocks = []
    for stock in stocks:
        if stock.symbol in completed:
            continue
        
        validated = validate_and_fix_ticker(stock)
        if validated:
            valid_stocks.append(validated)
    
    log.info(f"  ✓ {len(valid_stocks)} stocks to download")
    
    if not valid_stocks:
        log.info("  All stocks already downloaded!")
        return
    
    # Prepare output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{index_name}_ticker.csv"
    
    # Download in parallel
    stats = DownloadStats(total=len(valid_stocks))
    all_data = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all download tasks
        future_to_stock = {
            executor.submit(download_stock_data, stock, start_date, end_date): stock
            for stock in valid_stocks
        }
        
        # Process completed downloads
        for future in as_completed(future_to_stock):
            stock = future_to_stock[future]
            
            try:
                df = future.result()
                
                if df is not None and not df.empty:
                    all_data.append(df)
                    completed.add(stock.symbol)
                    stats.success += 1
                    log.info(f"  ✓ [{stats.success}/{stats.total}] {stock.symbol} ({len(df)} rows)")
                else:
                    stats.failed += 1
                    log.warning(f"  ✗ [{stats.success + stats.failed}/{stats.total}] {stock.symbol} - No data")
                
                # Save checkpoint every 10 stocks
                if stats.success % 10 == 0:
                    save_checkpoint(index_name, completed)
            
            except Exception as e:
                stats.failed += 1
                log.error(f"  ✗ [{stats.success + stats.failed}/{stats.total}] {stock.symbol} - {e}")
    
    # Save all data to CSV
    if all_data:
        log.info(f"Saving data to {output_file}...")
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        log.info(f"  ✓ Saved {len(combined_df)} rows to {output_file}")
    
    # Save final checkpoint
    save_checkpoint(index_name, completed)
    
    log.info(f"═══ DONE ═══  {stats.summary()}")

# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MENU
# ═══════════════════════════════════════════════════════════════════════════════

def print_menu():
    """Display the interactive menu."""
    print("\n" + "═" * 70)
    print("  NIFTY STOCK PRICE SCRAPER - Yahoo Finance")
    print("═" * 70)
    print(f"\n📅 Date Range: {START_DATE} to {END_DATE}")
    print(f"💾 Output Directory: {OUTPUT_DIR}")
    print(f"⚡ Parallel Workers: {MAX_WORKERS}")
    print("\n" + "─" * 70)
    print("  SELECT INDEX TO DOWNLOAD:")
    print("─" * 70)
    print("  [0] Nifty 50   (50 stocks)")
    print("  [1] Nifty 100  (100 stocks)")
    print("  [2] Nifty 200  (200 stocks)")
    print("  [3] All Indices (all three)")
    print("  [4] Exit")
    print("─" * 70)

def get_user_choice() -> str:
    """Get and validate user input."""
    while True:
        try:
            choice = input("\n👉 Enter your choice (0-4): ").strip()
            
            if choice in ["0", "1", "2", "3", "4"]:
                return choice
            else:
                print("❌ Invalid choice! Please enter 0, 1, 2, 3, or 4.")
        except (EOFError, KeyboardInterrupt):
            print("\n\n⚠️  Interrupted by user. Exiting...")
            exit(0)

def confirm_download(index_name: str) -> bool:
    """Ask user to confirm the download."""
    index_map = {
        "nifty50": "Nifty 50",
        "nifty100": "Nifty 100",
        "nifty200": "Nifty 200",
        "all": "All Indices (Nifty 50, 100, 200)"
    }
    
    print(f"\n{'─' * 70}")
    print(f"  You selected: {index_map[index_name]}")
    print(f"  Date Range: {START_DATE} to {END_DATE}")
    print(f"{'─' * 70}")
    
    while True:
        confirm = input("\n✓ Start download? (y/n): ").strip().lower()
        if confirm in ["y", "yes"]:
            return True
        elif confirm in ["n", "no"]:
            return False
        else:
            print("❌ Please enter 'y' or 'n'")

def interactive_mode():
    """Run the scraper in interactive menu mode."""
    while True:
        print_menu()
        choice = get_user_choice()
        
        # Map choice to index
        choice_map = {
            "0": "nifty50",
            "1": "nifty100",
            "2": "nifty200",
            "3": "all",
            "4": "exit"
        }
        
        index_name = choice_map[choice]
        
        if index_name == "exit":
            print("\n👋 Thank you for using Nifty Stock Scraper. Goodbye!\n")
            break
        
        # Confirm before starting download
        if not confirm_download(index_name):
            print("\n❌ Download cancelled. Returning to menu...\n")
            continue
        
        # Start download
        print("\n" + "═" * 70)
        print("  STARTING DOWNLOAD...")
        print("═" * 70 + "\n")
        
        # Fetch constituent lists
        indices_to_scrape = []
        
        if index_name in ["nifty50", "all"]:
            stocks = fetch_nifty50_constituents()
            if stocks:
                indices_to_scrape.append(("nifty50", stocks))
        
        if index_name in ["nifty100", "all"]:
            stocks = fetch_nifty100_constituents()
            if stocks:
                indices_to_scrape.append(("nifty100", stocks))
        
        if index_name in ["nifty200", "all"]:
            stocks = fetch_nifty200_constituents()
            if stocks:
                indices_to_scrape.append(("nifty200", stocks))
        
        # Scrape each index
        for idx_name, stocks in indices_to_scrape:
            scrape_index(idx_name, stocks, START_DATE, END_DATE)
            print()  # blank line between indices
        
        print("\n" + "═" * 70)
        print("  ✅ DOWNLOAD COMPLETED!")
        print("═" * 70)
        print(f"  📂 Files saved in: {OUTPUT_DIR}")
        print("═" * 70 + "\n")
        
        # Ask if user wants to continue
        while True:
            again = input("📥 Download another index? (y/n): ").strip().lower()
            if again in ["y", "yes"]:
                break
            elif again in ["n", "no"]:
                print("\n👋 Thank you for using Nifty Stock Scraper. Goodbye!\n")
                return
            else:
                print("❌ Please enter 'y' or 'n'")

# ═══════════════════════════════════════════════════════════════════════════════
# CLI MODE (for advanced users)
# ═══════════════════════════════════════════════════════════════════════════════

def cli_mode():
    """Run the scraper in command-line mode (with arguments)."""
    parser = argparse.ArgumentParser(
        description="Download Nifty stock historical prices from Yahoo Finance"
    )
    parser.add_argument(
        "--index",
        choices=["nifty50", "nifty100", "nifty200", "all"],
        required=True,
        help="Which index to download (nifty50, nifty100, nifty200, or all)"
    )
    parser.add_argument(
        "--start",
        default=START_DATE,
        help=f"Start date (YYYY-MM-DD, default: {START_DATE})"
    )
    parser.add_argument(
        "--end",
        default=END_DATE,
        help=f"End date (YYYY-MM-DD, default: {END_DATE})"
    )
    
    args = parser.parse_args()
    
    log.info("Nifty Stock Price Scraper")
    log.info(f"  Date Range: {args.start} to {args.end}")
    log.info(f"  Output Dir: {OUTPUT_DIR}")
    
    # Fetch constituent lists
    indices_to_scrape = []
    
    if args.index in ["nifty50", "all"]:
        stocks = fetch_nifty50_constituents()
        if stocks:
            indices_to_scrape.append(("nifty50", stocks))
    
    if args.index in ["nifty100", "all"]:
        stocks = fetch_nifty100_constituents()
        if stocks:
            indices_to_scrape.append(("nifty100", stocks))
    
    if args.index in ["nifty200", "all"]:
        stocks = fetch_nifty200_constituents()
        if stocks:
            indices_to_scrape.append(("nifty200", stocks))
    
    # Scrape each index
    for index_name, stocks in indices_to_scrape:
        scrape_index(index_name, stocks, args.start, args.end)
        print()  # blank line between indices

def main():
    """Main entry point - choose between interactive or CLI mode."""
    import sys
    
    # If arguments are provided, use CLI mode
    if len(sys.argv) > 1:
        cli_mode()
    else:
        # No arguments - use interactive menu mode
        try:
            interactive_mode()
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Exiting...")
            exit(0)

if __name__ == "__main__":
    main()
