# TFT Model

## Approach

`src/4_tft_hpt_train_test.py` provides a unified TFT pipeline for:
- Hyperparameter tuning (Optuna)
- Training and testing
- Inference-only runs

`src/5_tft_visualize.py` generates evaluation plots from TFT prediction artifacts.

Default window set: `7,10,15,30`

## Folder Layout

```text
models/TFT/
├── src/
│   ├── 4_tft_hpt_train_test.py
│   └── 5_tft_visualize.py
├── dataset/
│   ├── news_sentiment.csv
│   ├── tft_ready.csv
│   └── stock_dataset/nifty50_ticker.csv
├── artifacts/
│   ├── tft/
│   ├── tft_tune/
│   └── visualizations/
└── README.md
```

## Install Dependencies (`ml` env)

```bash
conda activate ml
conda install -y numpy pandas scipy scikit-learn matplotlib tqdm
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8
conda install -y -c conda-forge lightning pytorch-forecasting optuna optuna-integration
```

## Run Usage

From project root:

```bash
cd models/TFT
```

### 1) Unified TFT runner (interactive menu)

```bash
python src/4_tft_hpt_train_test.py --data-path dataset/tft_ready.csv --windows 7,10,15,30
```

Menu modes:
- `0`: tune -> train/test
- `1`: train/test only
- `2`: inference only
- `3`: exit

### 2) Unified TFT runner (non-interactive)

Tune then train/test:

```bash
python src/4_tft_hpt_train_test.py --mode 0 --no-window-menu --data-path dataset/tft_ready.csv --windows 7,10,15,30
```

Train/test only:

```bash
python src/4_tft_hpt_train_test.py --mode 1 --no-window-menu --data-path dataset/tft_ready.csv --windows 7,10,15,30
```

Inference only:

```bash
python src/4_tft_hpt_train_test.py --mode 2 --no-window-menu --data-path dataset/tft_ready.csv --windows 7,10,15,30
```

### 3) Visualization

```bash
python src/5_tft_visualize.py \
  --artifact-root artifacts/tft \
  --ticker-path dataset/stock_dataset/nifty50_ticker.csv \
  --windows 7,10,15,30 \
  --output-dir artifacts/visualizations \
  --dpi 300
```

## Outputs

- Training/test artifacts: `artifacts/tft/window_<N>/`
- Tuning artifacts: `artifacts/tft_tune/window_<N>/`
- Summary files:
  - `artifacts/tft/metrics_summary.csv`
  - `artifacts/tft_tune/tuning_summary.csv`
- Plots:
  - `artifacts/visualizations/best_window_actual_vs_predicted.png`
  - `artifacts/visualizations/per_window_metrics.png`
  - `artifacts/visualizations/best_window_stock_mape.png`
