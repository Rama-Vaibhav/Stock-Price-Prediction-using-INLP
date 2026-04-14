# INLP Stock Price Prediction Project

This repository contains a modular pipeline for Nifty stock forecasting using news + market data, plus multiple model tracks (TFT, GRU, LSTM, and an ensemble).

## What We Are Trying To Do

The core goal of this project is to predict short-term stock movement for Nifty-listed companies by combining:
- market price history (OHLCV-based signals),
- financial news context (direct, sectoral, and global impact), and
- sentiment intelligence from language models.

In practice, we transform raw scraped data into a unified modeling dataset (`tft_ready.csv`), train multiple forecasting models, compare their performance, and then use an ensemble to improve robustness.

In exact terms, the forecasting setup uses various past windows (`7`, `10`, `15`, `30` days) to predict the **next-day Adjusted Close price**.

## Root-Level Items

| Item | Purpose |
|---|---|
| `README.md` | Main project documentation (this file). |
| `dataset/` | Active root datasets used by feature engineering and model training. |
| `models/` | Model-specific code, notebooks, outputs, and model READMEs. |
| `src/` | Main data pipeline scripts and web/data scrapers. |
| `archive/` | Deprecated legacy assets retained only for historical traceability; not part of the active workflow. |

## Current Directory Structure

```text
.
├── dataset/
│   ├── news_sentiment.csv
│   ├── nifty50_ticker.csv
│   └── tft_ready.csv
├── models/
│   ├── ENSEMBLED MODEL/
│   │   ├── ensemble_learner_final.ipynb
│   │   └── ensemble_outputs/
│   ├── GRU/
│   │   ├── gru_stock_prediction.ipynb
│   │   └── GRU_outputs/
│   ├── LSTM/
│   │   ├── lstm_stock_prediction.ipynb
│   │   ├── checkpoints/
│   │   └── lstm_outputs/
│   └── TFT/
│       ├── src/
│       │   ├── 4_tft_hpt_train_test.py
│       │   └── 5_tft_visualize.py
│       ├── dataset/
│       ├── artifacts/
│       └── README.md
├── src/
│   ├── 0_preprocess_news.py
│   ├── 1_qwen_news_segregation.py
│   ├── 2_finbert_sentiment.py
│   ├── 3_feature_engineering.py
│   └── scrapers/
│       ├── businessstandard_scraper.py
│       ├── economictimes_scraper.py
│       ├── financialexpress_scraper.py
│       ├── moneycontrol_scraper.py
│       └── nifty_yfinance_scraper.py
└── README.md
```

## Dataset Brief

This snapshot includes processed datasets needed for modeling, while most raw/intermediate files from the original full pipeline are not included.

Available now:
- `dataset/news_sentiment.csv` (125,511 rows)
- `dataset/nifty50_ticker.csv` (74,439 rows)
- `dataset/tft_ready.csv` (71,939 rows)

Column highlights:
- `news_sentiment.csv`: date/symbol + sentiment probabilities and article counts for direct, sectoral, global news.
- `nifty50_ticker.csv`: OHLCV and corporate-action columns per symbol/date.
- `tft_ready.csv`: engineered market + sentiment features, `target_pct_change`, and `time_idx`.

If you need the full raw-to-final flow context (including files not present in this snapshot), use `README copy.md` as the reference.

## Environment and Dependency Installation

Per project policy, use the conda environment `inlp_project`.

```bash
conda activate inlp_project
```

Install in priority order.

1. Conda defaults first:

```bash
conda install -y numpy pandas scipy scikit-learn matplotlib tqdm requests aiohttp beautifulsoup4 lxml selenium python-dotenv
```

2. Then conda-forge for missing packages:

```bash
conda install -y -c conda-forge yfinance ta transformers lightning pytorch-forecasting optuna optuna-integration curl_cffi jupyterlab
```

3. PyTorch (CUDA stack):

```bash
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8
```

4. Pip only if still unavailable via conda channels:

```bash
pip install ollama
```

## `src/` Files: Purpose and Commands

### Core pipeline scripts

| File | What it does | Main output | Run command |
|---|---|---|---|
| `src/0_preprocess_news.py` | Merges all `dataset/raw_dataset/*_raw.csv`, cleans/deduplicates text, filters date range, sorts chronologically. | `dataset/processed_news_dataset.csv` | `python src/0_preprocess_news.py` |
| `src/1_qwen_news_segregation.py` | Uses Ollama Qwen to split news into direct/sectoral/global impact with checkpoint/retry support. | `dataset/tier_segregated_news.csv` | `python src/1_qwen_news_segregation.py` |
| `src/2_finbert_sentiment.py` | Runs FinBERT sentiment scoring (with unique-text caching and sliding windows). | `dataset/news_sentiment.csv` | `python src/2_finbert_sentiment.py` |
| `src/3_feature_engineering.py` | Merges ticker + sentiment data, adds technical features, scales per symbol, builds modeling table. | `dataset/tft_ready.csv` | `python src/3_feature_engineering.py` |

Notes:
- `1_qwen_news_segregation.py` is menu-driven: `1` NER only, `2` CSV only, `3` full pipeline.
- For Qwen segregation, run Ollama first:
  - `ollama serve`
  - `ollama pull qwen2.5:3b`
- `2_finbert_sentiment.py` currently expects `dataset/tier_segragated_news.csv` (filename typo in script constant). Ensure the expected input filename exists before running.

### Scraper files

| File | What it does | Main output | Run command |
|---|---|---|---|
| `src/scrapers/moneycontrol_scraper.py` | Async scraper (curl_cffi TLS impersonation) for Moneycontrol news pages with checkpoint resume. | `dataset/raw_dataset/moneycontrol_raw.csv` | `python src/scrapers/moneycontrol_scraper.py` |
| `src/scrapers/financialexpress_scraper.py` | Async + process-pool scraper for Financial Express sections with robust checkpointing. | `dataset/raw_dataset/financialexpress_raw.csv` | `python src/scrapers/financialexpress_scraper.py` |
| `src/scrapers/economictimes_scraper.py` | Sitemap-driven historical Economic Times scraper (`/markets/stocks/news/`) with dedup checkpoints. | `dataset/raw_dataset/economictimes_raw.csv` | `python src/scrapers/economictimes_scraper.py` |
| `src/scrapers/businessstandard_scraper.py` | Hybrid scraper: curl_cffi for listing pages + Selenium for JS-rendered article pages. | `dataset/raw_dataset/businessstandard_raw.csv` | `python src/scrapers/businessstandard_scraper.py` |
| `src/scrapers/nifty_yfinance_scraper.py` | Downloads Nifty constituent OHLCV data (Nifty50/100/200/all) from Yahoo Finance with resume support. | `dataset/stock_dataset/<index>_ticker.csv` | `python src/scrapers/nifty_yfinance_scraper.py --index nifty50 --start 2020-01-01 --end 2026-03-31` |

`nifty_yfinance_scraper.py` also supports interactive mode if run without CLI arguments.

## Models Overview

- `models/TFT/`: Temporal Fusion Transformer pipeline (tuning, training/testing, inference, visualizations).
- `models/GRU/`: GRU quantile forecasting notebook with multi-window experiments and evaluation artifacts.
- `models/LSTM/`: LSTM quantile forecasting notebook with multi-window experiments and evaluation artifacts.
- `models/ENSEMBLED MODEL/`: Stacking ensemble notebook combining LSTM + GRU + TFT outputs using meta-learners.

Detailed setup and usage for each model are documented inside each model folder:
- `models/TFT/README.md`
- `models/GRU/README.md`
- `models/LSTM/README.md`
- `models/ENSEMBLED MODEL/README.md`
