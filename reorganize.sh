#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# reorganize.sh — Restructure repository into a modular layout
#
# Rules:
#   • No files or directories are renamed or deleted.
#   • Only mkdir -p and mv are used.
#
# Target layout:
#   models/        — LSTM, GRU, ENSEMBLED MODEL, TFT, tft_codebase, checkpoints
#   notebooks/     — all .ipynb files from root
#   src/           — all .py scripts from root
#   data/          — all .csv data files from root
#   archive/       — Old/
# ──────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "📂 Creating target directories..."
mkdir -p models
mkdir -p notebooks
mkdir -p src
mkdir -p data
mkdir -p archive

# ── Models ────────────────────────────────────────────────────
echo "🤖 Moving model folders → models/"
mv "LSTM"             models/
mv "GRU"              models/
mv "ENSEMBLED MODEL"  models/
mv "TFT"              models/
mv "tft_codebase"     models/
mv "checkpoints"      models/

# ── Notebooks ─────────────────────────────────────────────────
echo "📓 Moving notebooks → notebooks/"
mv "HPT_10days.ipynb"                    notebooks/
mv "HPT_7-15-30Days.ipynb"              notebooks/
mv "ensemble_learner_(1) (5).ipynb"     notebooks/
mv "finBERT_Sen_Final.ipynb"            notebooks/
mv "gru_stock_prediction.ipynb"         notebooks/
mv "lstm_stock_prediction.ipynb"        notebooks/
mv "pred_LSTM_final.ipynb"              notebooks/

# ── Scripts ───────────────────────────────────────────────────
echo "🐍 Moving scripts → src/"
mv "financial_express_scraper.py"       src/
mv "nifty50_yfinance_extractor.py"      src/
mv "preprocessor.py"                    src/

# ── Data ──────────────────────────────────────────────────────
echo "📊 Moving data files → data/"
mv "cleaned_news.csv"                   data/
mv "nifty50_historical_prices.csv"      data/

# ── Archive ───────────────────────────────────────────────────
echo "🗄️  Moving archive → archive/"
mv "Old"                                archive/

echo ""
echo "✅ Reorganization complete! New layout:"
echo ""
ls -1F "$REPO_ROOT"
