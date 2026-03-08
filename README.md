# Stock Price Prediction Using NLP (FinBERT + LSTM)

A deep learning pipeline that predicts **Nifty 50 closing prices** by combining historical OHLCV data with **news sentiment** extracted via **FinBERT**. The project scrapes financial news, computes sentiment scores, engineers time-series features, and trains a stacked LSTM model with Bayesian-optimised hyperparameters.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Model Details](#model-details)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## Overview

| Aspect | Details |
|---|---|
| **Market** | NSE Nifty 50 (50 stocks) |
| **Date Range** | January 2023 – February 2026 |
| **News Source** | Financial Express (Business & Market sections) |
| **Sentiment Model** | [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) |
| **Prediction Model** | Stacked 2-layer LSTM (TensorFlow/Keras) |
| **HP Tuning** | Bayesian Optimisation via Keras Tuner |
| **Prediction Target** | Next-day closing price per ticker |

---

## Architecture

```
Financial Express  ──►  Web Scraper  ──►  Cleaned News CSV
                                              │
                                         FinBERT Sentiment
                                         (pos / neg / neu)
                                              │
Yahoo Finance  ──►  OHLCV Extractor  ──►  Merged Dataset  ──►  LSTM  ──►  Predicted Close
(Nifty 50)          (yfinance)            (prices + sentiment)
```

1. **Scrape** financial news articles (multi-threaded).
2. **Clean & preprocess** text and price data.
3. **Score sentiment** per day using FinBERT (GPU-accelerated, sliding-window batched inference).
4. **Merge** daily sentiment scores with per-ticker OHLCV data.
5. **Engineer features**: 10-day lagged features for Close, Open, High, Low, Volume, and sentiment scores.
6. **Train** a 2-layer LSTM with tuned hyperparameters; evaluate on a chronological test split.

---

## Project Structure

```
├── financial_express_scraper.py       # Multi-threaded news scraper (Financial Express)
├── nifty50_yfinance_extractor.py      # Downloads Nifty 50 OHLCV data via yfinance
├── preprocessor.py                    # 4-phase preprocessing pipeline (merge, clean, features, calendar)
├── finBERT_Sen_Final.ipynb            # FinBERT sentiment analysis → merged dataset
├── pred_LSTM_final.ipynb              # Final LSTM training, evaluation & visualisation
├── HPT_10days.ipynb                   # Hyperparameter tuning (10-day window, Bayesian Opt.)
├── HPT_7-15-30Days.ipynb              # Hyperparameter tuning across 7/15/30-day windows
├── cleaned_news.csv                   # Pre-cleaned news articles
├── nifty50_historical_prices.csv      # Raw Nifty 50 OHLCV data
├── merged_financial_data_news.csv     # Final merged dataset (prices + sentiment)
├── lstm_stock_predictor_norm_2lstm_tsip.keras  # Trained LSTM model weights
├── checkpoints/
│   └── best_lstm.keras                # Best checkpoint from training
└── Old/                               # Earlier experimental notebooks
```

---

## Data Pipeline

### 1. News Scraping (`financial_express_scraper.py`)

- Scrapes the **Business** and **Market** sections of Financial Express.
- Uses 20 concurrent threads with retry logic.
- Extracts article date, title, body, author, and URL.
- Date range: **Sep 1, 2025 – Feb 28, 2026**.

### 2. Price Extraction (`nifty50_yfinance_extractor.py`)

- Downloads daily OHLCV + Adj Close for all **50 Nifty 50 tickers** via `yfinance`.
- Covers **Jan 1, 2023 – Feb 28, 2026**.
- Outputs a flat CSV with columns: `Date, Ticker, Open, High, Low, Close, Adj Close, Volume`.

### 3. Preprocessing (`preprocessor.py`)

A 4-phase crash-resilient pipeline:

| Phase | Description | Output |
|---|---|---|
| **Phase 1** | Merge raw `*_news.csv` files, normalize columns, deduplicate by URL | `combined_market_news.csv` |
| **Phase 2** | HTML decode, remove boilerplate, filter short articles, enforce date range | `cleaned_news.csv` |
| **Phase 3** | Holiday/stale-row removal, sector tagging, 13 technical indicators per ticker | `cleaned_prices.csv`, `price_features.csv` |
| **Phase 4** | Trading calendar, daily news volume computation | `trading_calendar.csv`, `daily_news_volume.csv` |

### 4. Sentiment Analysis (`finBERT_Sen_Final.ipynb`)

- Loads `cleaned_news.csv`; cleans text (lowercase, remove punctuation, lemmatize, drop stopwords) using multiprocessing.
- Concatenates all articles per day into a single document.
- Runs **ProsusAI/FinBERT** with a sliding-window strategy (window=512, stride=256) and batched GPU inference (FP16).
- Produces per-day `pos_score`, `neg_score`, `neu_score`.
- Left-joins sentiment scores with `nifty50_historical_prices.csv` → outputs `merged_financial_data_news.csv`.

---

## Model Details

### Features

| Feature Type | Columns |
|---|---|
| **Price** | Open, High, Low, Close, Volume |
| **Sentiment** | pos_score, neg_score, neu_score |
| **Ticker** | ticker_scaled (MinMax-encoded ticker ID) |
| **Lagged** | 10-day lags for each of the 8 features above |

Total input per sample: **11 timesteps × 9 features** (reshaped for LSTM).

### LSTM Architecture

```
Input (11 × 9)
    │
LSTM (64 units, return_sequences=True)
    │
Dropout (0.3)
    │
LSTM (32 units)
    │
Dropout (0.3)
    │
Dense (64, ReLU)
    │
Dense (1) → Predicted Close Price
```

### Hyperparameter Tuning

Bayesian Optimisation (Keras Tuner, 20 trials) explored:

| Hyperparameter | Search Space | Best Value |
|---|---|---|
| `lstm_units_1` | {64, 128, 256} | 64 |
| `lstm_units_2` | {32, 64, 128} | 32 |
| `dropout_rate` | 0.1 – 0.5 | 0.3 |
| `dense_units` | {16, 32, 64} | 64 |
| `learning_rate` | 1e-4 – 1e-2 (log) | 0.0021 |

### Training Configuration

- **Loss**: MSE
- **Optimizer**: Adam (lr = 0.0021)
- **Batch size**: 64
- **Max epochs**: 120 (EarlyStopping patience=15)
- **LR scheduler**: ReduceLROnPlateau (factor=0.5, patience=7)
- **Data split**: 80/10/10 (train/val/test, chronological)
- **Seed**: 32 (reproducible)

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- NVIDIA GPU with CUDA (optional, for faster training & inference)

### Environment Setup

```bash
# Create and activate conda environment
conda create -n ml python=3.10 -y
conda activate ml

# Core dependencies
conda install pandas numpy scikit-learn matplotlib seaborn -c conda-forge
conda install -c conda-forge yfinance
conda install requests beautifulsoup4

# Deep learning
pip install tensorflow[and-cuda]    # or: conda install tensorflow-gpu
pip install transformers
pip install keras-tuner

# NLP
pip install nltk torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Usage

### 1. Scrape News

```bash
conda activate ml
python financial_express_scraper.py
```

### 2. Download Price Data

```bash
python nifty50_yfinance_extractor.py
```

### 3. Run Preprocessing

```bash
python preprocessor.py              # full pipeline
python preprocessor.py --skip-merge  # skip Phase 1 if already merged
python preprocessor.py --force       # re-run all steps
```

### 4. Generate Sentiment Scores

Open and run all cells in **`finBERT_Sen_Final.ipynb`**. This produces `merged_financial_data_news.csv`.

### 5. Train the LSTM Model

Open and run all cells in **`pred_LSTM_final.ipynb`**. The trained model is saved to `lstm_stock_predictor_norm_2lstm_tsip.keras`.

### 6. Hyperparameter Tuning

- **10-day window**: `HPT_10days.ipynb`[Got best hyperparameter from this]
- **7/15/30-day windows**: `HPT_7-15-30Days.ipynb`

---

## Results

The final model is evaluated on a chronological test set with the following metrics:

- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error) — computed per ticker

The notebooks produce:
- Actual vs. Predicted close price plots (full test set)
- Per-ticker MAPE rankings
- Top-5 lowest-MAPE stock visualisations with individual subplots

