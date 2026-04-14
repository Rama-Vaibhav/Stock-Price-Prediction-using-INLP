# Ensemble Model

## Approach

This notebook builds a stacking ensemble over three base models:

- LSTM (window 30)
- GRU (window 30)
- TFT (window 15)

How it works:
- Each base model provides 7 quantile predictions.
- Quantiles are concatenated into 21 stacked features.
- Multiple meta-learners are trained (Ridge, ElasticNet, GBR, RF, SVR, MLP variants).
- Best meta-learner is selected by validation MSE (no test leakage).

Main notebook:
- `ensemble_learner_final.ipynb`

Stored outputs in this repo:
- `ensemble_outputs/results/`
- `ensemble_outputs/plots/`

## Install Dependencies

```bash
conda activate inlp_project
conda install -y numpy pandas scipy scikit-learn matplotlib seaborn tqdm jupyterlab
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8
conda install -y -c conda-forge lightning pytorch-forecasting
```

## Run Usage

From project root:

```bash
cd 'models/ENSEMBLED MODEL'
cp ../../dataset/tft_ready.csv dataset.csv
jupyter lab ensemble_learner_final.ipynb
```

Optional non-interactive execution:

```bash
jupyter nbconvert --to notebook --execute --inplace ensemble_learner_final.ipynb
```
