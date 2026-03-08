"""
Nifty 50 Historical Data Extractor using Yahoo Finance
Downloads daily open, high, low, close (OHLC), adj close, and volume
for all Nifty 50 stocks from Sep 1, 2025 to Feb 28, 2026.

Environment requirements:
- Run inside the 'ml' conda environment.
- Install dependencies: `conda install -c conda-forge yfinance pandas`

Usage:
1. conda activate ml
2. python nifty50_yfinance_extractor.py
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

# The list of Nifty 50 companies (with .NS appended for Yahoo Finance NSE tickers)
# Updated to reflect the Nifty 50 composition as of Dec 2025:
#   Sep 2024 rejig: DIVISLAB→TRENT, LTIM→BEL
#   Mar 2025 rejig: BPCL→ETERNAL, BRITANNIA→JIOFIN
#   Sep 2025 rejig: HEROMOTOCO→MAXHEALTH, INDUSINDBK→INDIGO
#   Oct 2025: TATAMOTORS demerged → TMPV (Tata Motors Passenger Vehicles)
NIFTY_50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDIGO.NS", "INFY.NS", "ITC.NS",
    "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "MAXHEALTH.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATASTEEL.NS", "TECHM.NS",
    "TITAN.NS", "TMPV.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]

# yfinance end_date is exclusive, so we use Mar 1 to include Feb 28.
START_DATE = "2023-01-01"
END_DATE = "2026-03-01"
CSV_FILE = "nifty50_historical_prices.csv"

def extract_nifty50_data():
    print("=" * 70)
    print("Nifty 50 Historical Data Extractor (yfinance)")
    print(f"Date Range: {START_DATE} to {datetime.strptime(END_DATE, '%Y-%m-%d').strftime('%Y-%b-%d')} (Exclusive)")
    print("=" * 70)

    print(f"\nDownloading data for {len(NIFTY_50_TICKERS)} Nifty 50 tickers...")
    print("This may take 10-20 seconds...\n")
    
    # yfinance download efficiently grabs all tickers at once.
    # group_by='ticker' groups the columns by the stock symbol.
    data = yf.download(
        tickers=" ".join(NIFTY_50_TICKERS),
        start=START_DATE,
        end=END_DATE,
        group_by='ticker',
        threads=True, 
        auto_adjust=False # Keep explicit Adj Close column
    )

    if data.empty:
        print("Error: No data was downloaded. Please check your internet connection or date range.")
        return

    print("Formatting and flattening data for CSV...")
    
    # The downloaded dataframe is multi-indexed in columns: (Ticker, Price Type)
    # E.g., ('RELIANCE.NS', 'Close')
    # We want to flatten it to rows: Date, Ticker, Open, High, Low, Close, Adj Close, Volume
    
    flattened_rows = []
    
    for ticker in NIFTY_50_TICKERS:
        if ticker not in data.columns.levels[0]:
            print(f"  Warning: No data found for {ticker}")
            continue
            
        ticker_data = data[ticker].dropna(how='all') # Drop days where the market was closed/no data
        
        for date, row in ticker_data.iterrows():
            flattened_rows.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Ticker': ticker.replace('.NS', ''), # Remove .NS suffix to match exact Nifty 50 name
                'Open': round(row.get('Open', 0), 2),
                'High': round(row.get('High', 0), 2),
                'Low': round(row.get('Low', 0), 2),
                'Close': round(row.get('Close', 0), 2),
                'Adj Close': round(row.get('Adj Close', 0), 2),
                'Volume': int(row.get('Volume', 0))
            })

    # Create a clean flat pandas DataFrame
    df = pd.DataFrame(flattened_rows)
    
    # Sort logically by Date then by Ticker
    df = df.sort_values(by=['Date', 'Ticker'])
    
    # Save to CSV
    df.to_csv(CSV_FILE, index=False)
    
    print("\n" + "=" * 70)
    print(f"Extraction complete! Saved {len(df)} rows across {len(df['Ticker'].unique())} tickers.")
    print(f"Output saved to: {CSV_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    extract_nifty50_data()
