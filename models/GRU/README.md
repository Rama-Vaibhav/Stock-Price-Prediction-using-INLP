# GRU Model

## Approach

This notebook trains a GRU-based quantile forecaster for next-day `target_pct_change`.

- Multi-window setup: `7, 10, 15, 30`
- Model sizes per window: `small`, `medium`, `large`
- Quantile output: 7 quantiles (pinball loss)
- Split: train `<= 2024-12-31`, val `2025-01-01` to `2025-06-30`, test `>= 2025-07-01`

Main notebook:
- `gru_stock_prediction.ipynb`

Stored outputs in this repo:
- `GRU_outputs/gru_outputs/results/`
- `GRU_outputs/gru_outputs/plots/`

## Install Dependencies

```bash
conda activate inlp_project
conda install -y numpy pandas scipy scikit-learn matplotlib seaborn tqdm jupyterlab
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8
```

## Run Usage

From project root:

```bash
cd models/GRU
cp ../../dataset/tft_ready.csv dataset.csv
cp ../../dataset/nifty50_ticker.csv nifty50_ticker.csv
jupyter lab gru_stock_prediction.ipynb
```

Optional non-interactive execution:

```bash
jupyter nbconvert --to notebook --execute --inplace gru_stock_prediction.ipynb
```

