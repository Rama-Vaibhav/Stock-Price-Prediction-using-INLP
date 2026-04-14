# LSTM Model

## Approach

This notebook trains an LSTM-based quantile forecaster for next-day `target_pct_change`.

- Multi-window setup: `7, 10, 15, 30`
- Model sizes per window: `small`, `medium`, `large`
- Quantile output: 7 quantiles (pinball loss)
- Split: train `<= 2024-12-31`, val `2025-01-01` to `2025-06-30`, test `>= 2025-07-01`

Main notebook:
- `lstm_stock_prediction.ipynb`

Stored outputs in this repo:
- `lstm_outputs/lstm_outputs-2/results/`
- `lstm_outputs/lstm_outputs-2/plots/`
- `checkpoints/`

## Install Dependencies

```bash
conda activate inlp_project
conda install -y numpy pandas scipy scikit-learn matplotlib seaborn tqdm jupyterlab
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8
```

## Run Usage

From project root:

```bash
cd models/LSTM
cp ../../dataset/tft_ready.csv dataset.csv
cp ../../dataset/nifty50_ticker.csv nifty50_ticker.csv
jupyter lab lstm_stock_prediction.ipynb
```

Optional non-interactive execution:

```bash
jupyter nbconvert --to notebook --execute --inplace lstm_stock_prediction.ipynb
```

