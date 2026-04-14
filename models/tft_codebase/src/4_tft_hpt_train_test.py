#!/usr/bin/env python3
"""
TFT pipeline for Nifty50 multi-series forecasting.


Sections
--------
1. Imports  (stdlib → third-party)

2. SHARED CONSTANTS & UTILITIES
   Date/column names, split boundaries, batch-size map, metric column list.
   configure_logging, configure_warnings, set_seed, ensure_dir, parse_windows,
   choose_windows_menu, now_utc_iso, read_json, write_json,
   find_latest_checkpoint, to_numpy, safe_div, compute_regression_metrics,
   compute_mape, compute_direction_metrics, load_and_prepare_dataframe,
   get_split_masks, choose_model_features, filter_eligible_symbols,
   build_datasets_for_window, unpack_prediction_output, find_symbol_col,
   find_time_col, build_truth_and_price_matrices.

3. TRAINING & TESTING
   WindowStateCallback, RunConfig, compute_window_metrics,
   save_prediction_audit, evaluate_window_and_write_outputs,
   run_window_training.

4. HYPERPARAMETER TUNING
   TuneConfig, pick_precision_sequence, pick_loader_workers, is_probable_oom,
   is_probable_bf16_issue, compute_per_symbol_equal_direction_metrics,
   WindowObjective, export_trials_csv, print_window_summary, tune_window.

5. UNIFIED CONFIG
   UnifiedConfig dataclass merging all RunConfig and TuneConfig fields.
   make_run_config (builds RunConfig with optional per-window HP overrides),
   make_tune_config, load_tuned_hps_for_window.

6. CLI & MAIN
   build_arg_parser, make_unified_config, show_main_menu,
   run_tuning_mode, run_train_test_mode, run_inference_only_mode, main.
   if __name__ == "__main__": main()

Interactive top-level menu
--------------------------
  [0] Hyperparameter tuning  → then training/testing
  [1] Training/testing only  (loads tuned HPs when available)
  [2] Inference only
  [3] Exit
"""

from __future__ import annotations

# ── stdlib ──────────────────────────────────────────────────────────────────
import argparse
import gc
import json
import logging
import os
import random
import signal
import sys
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        Callback,
        EarlyStopping,
        ModelCheckpoint,
        TQDMProgressBar,
    )
    from lightning.pytorch.loggers import CSVLogger
except ImportError as exc:
    raise ImportError(
        "Missing dependency: lightning. "
        "Install in your conda env (ml/ml2), e.g. `pip install lightning==2.6.1`."
    ) from exc

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
except ImportError as exc:
    raise ImportError(
        "Missing dependency: pytorch-forecasting. "
        "Install in your conda env, e.g. `pip install pytorch-forecasting==1.7.0`."
    ) from exc

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "Missing dependency: optuna. "
        "Install in your conda env, e.g. `pip install optuna`."
    ) from exc

try:
    from optuna.integration import PyTorchLightningPruningCallback
except Exception:
    try:
        # Newer packaging may require the separate optuna-integration package.
        from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
    except Exception as exc:
        raise ImportError(
            "Missing dependency for Optuna pruning callback. "
            "Install `optuna-integration` (or compatible optuna extras)."
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SHARED CONSTANTS & UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

DATE_COL = "Date"
SYMBOL_COL = "Symbol"
TIME_IDX_COL = "time_idx"
TARGET_COL = "target_pct_change"
PRICE_COL = "Adj Close"

TRAIN_END = "2024-12-31"
VAL_START = "2025-01-01"
VAL_END = "2025-06-30"
TEST_START = "2025-07-01"

DEFAULT_WINDOWS = (7, 10, 15, 30)
DEFAULT_PREDICTION_LENGTH = 1

DEFAULT_DATA_PATH = Path("dataset/tft_ready.csv")
DEFAULT_TRAIN_ARTIFACT_ROOT = Path("artifacts/tft")
DEFAULT_TUNE_ARTIFACT_ROOT = Path("artifacts/tft_tune")

# Drop raw OHLCV from model features (keep Adj Close in the original dataframe
# for price-space evaluation).
RAW_FEATURE_DROP = {
    "Open", "High", "Low", "Close", "Adj Close", "Volume", "symbol_base", DATE_COL
}

# Calendar known covariates
CALENDAR_KNOWN_CATEGORICALS = ["dow", "dom", "month", "is_month_start", "is_month_end"]
CALENDAR_KNOWN_REALS = [TIME_IDX_COL]

# Per-window base batch sizes (effective batch doubles with accumulate_grad_batches=2).
WINDOW_BATCH_SIZE = {7: 128, 10: 128, 15: 96, 30: 64}

FINAL_METRIC_COLUMNS = [
    "window",
    "mape",
    "mae",
    "rmse",
    "mse",
    "directional_accuracy",
    "precision_up",
    "recall_up",
    "f1_up",
]


# ── Logging / warnings / seeding ─────────────────────────────────────────────

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy Lightning info lines (hardware availability, LOCAL_RANK, …)
    for noisy_logger in ("lightning", "lightning.pytorch", "pytorch_lightning"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def configure_warnings() -> None:
    # Non-actionable Lightning / PyTorch-Forecasting warnings that add terminal noise.
    warnings.filterwarnings(
        "ignore",
        message=r"Attribute 'loss' is an instance of `nn.Module` and is already saved during checkpointing.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Attribute 'logging_metrics' is an instance of `nn.Module` and is already saved during checkpointing.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Checkpoint directory .* exists and is not empty\.",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Starting from v1\.9\.0, `tensorboardX` has been removed as a dependency of the `lightning\.pytorch` package.*",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


# ── Path / JSON helpers ───────────────────────────────────────────────────────

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_windows(text: str) -> List[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def choose_windows_menu(default_windows: Sequence[int]) -> List[int]:
    """
    Interactive selector for encoder windows.
    Menu mapping:
      0 -> [7]
      1 -> [10]
      2 -> [15]
      3 -> [30]
      4 -> [7, 10, 15, 30]
      5 -> Exit
    Empty input defaults to option 4 (all windows).
    """
    menu = {
        "0": [7],
        "1": [10],
        "2": [15],
        "3": [30],
        "4": list(DEFAULT_WINDOWS),
    }
    lines = [
        "",
        "Select TFT encoder window(s):",
        "  [0] 7 days",
        "  [1] 10 days",
        "  [2] 15 days",
        "  [3] 30 days",
        "  [4] All windows (7,10,15,30)",
        "  [5] Exit",
    ]
    print("\n".join(lines))

    while True:
        try:
            choice = input("Enter choice (0-5) [default: 4]: ").strip()
        except EOFError:
            logging.info(
                "No interactive input detected. Using default windows: %s",
                list(default_windows),
            )
            return list(default_windows)

        if choice == "":
            choice = "4"

        if choice in menu:
            selected = menu[choice]
            logging.info("Window menu selection -> %s", selected)
            return selected

        if choice == "5":
            logging.info("Window menu selection -> exit")
            raise SystemExit(0)

        print("Invalid choice. Please select one of: 0, 1, 2, 3, 4, 5.")


def now_utc_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None
    candidates = [p for p in checkpoint_dir.glob("*.ckpt") if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# ── Array helpers ─────────────────────────────────────────────────────────────

def to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return a / np.where(np.abs(b) < eps, eps, b)


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    err = y_pred - y_true
    mse_val = float(np.mean(np.square(err)))
    mae_val = float(np.mean(np.abs(err)))
    rmse_val = float(np.sqrt(mse_val))
    return {"mse": mse_val, "mae": mae_val, "rmse": rmse_val}


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_pred - y_true
    return float(np.mean(np.abs(safe_div(err, y_true))) * 100.0)


def compute_direction_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    true_up = (y_true > 0.0).astype(np.int32)
    pred_up = (y_pred > 0.0).astype(np.int32)
    directional_accuracy = float(np.mean(true_up == pred_up))
    precision_up = float(precision_score(true_up, pred_up, zero_division=0))
    recall_up = float(recall_score(true_up, pred_up, zero_division=0))
    f1_up = float(f1_score(true_up, pred_up, zero_division=0))
    return {
        "directional_accuracy": directional_accuracy,
        "precision_up": precision_up,
        "recall_up": recall_up,
        "f1_up": f1_up,
    }


# ── Data preparation ──────────────────────────────────────────────────────────

def load_and_prepare_dataframe(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    required = {DATE_COL, SYMBOL_COL, TIME_IDX_COL, TARGET_COL, PRICE_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if df[DATE_COL].isna().any():
        raise ValueError("Invalid Date values found in dataset.")

    if df[[SYMBOL_COL, TIME_IDX_COL, TARGET_COL]].isna().any().any():
        raise ValueError(
            "Found NaNs in one of required columns: Symbol, time_idx, target_pct_change."
        )

    df[TIME_IDX_COL] = df[TIME_IDX_COL].astype(np.int64)
    df = df.sort_values([SYMBOL_COL, DATE_COL], kind="mergesort").reset_index(drop=True)

    # Add known calendar features as categoricals for TFT embeddings.
    df["dow"] = df[DATE_COL].dt.weekday.astype(np.int16).astype(str)
    df["dom"] = df[DATE_COL].dt.day.astype(np.int16).astype(str)
    df["month"] = df[DATE_COL].dt.month.astype(np.int16).astype(str)
    df["is_month_start"] = df[DATE_COL].dt.is_month_start.astype(np.int8).astype(str)
    df["is_month_end"] = df[DATE_COL].dt.is_month_end.astype(np.int8).astype(str)

    dup_count = int(df.duplicated([SYMBOL_COL, DATE_COL]).sum())
    if dup_count > 0:
        raise ValueError(f"Found duplicated (Symbol, Date) rows: {dup_count}")

    return df


def get_split_masks(
    df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    train_end = pd.Timestamp(TRAIN_END)
    val_start = pd.Timestamp(VAL_START)
    val_end = pd.Timestamp(VAL_END)
    test_start = pd.Timestamp(TEST_START)

    train_mask = df[DATE_COL] <= train_end
    val_mask = (df[DATE_COL] >= val_start) & (df[DATE_COL] <= val_end)
    test_mask = df[DATE_COL] >= test_start

    return train_mask, val_mask, test_mask


def choose_model_features(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str]]:
    known_categoricals = [c for c in CALENDAR_KNOWN_CATEGORICALS if c in df.columns]
    # Known reals are deterministic numeric features available in the future.
    known_reals = [c for c in CALENDAR_KNOWN_REALS if c in df.columns]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    excluded = set(known_reals + [TARGET_COL, TIME_IDX_COL])
    unknown_covariates = [
        c for c in numeric_cols if c not in excluded and c not in RAW_FEATURE_DROP
    ]

    # Target must be included among unknown reals for autoregressive decoding.
    unknown_reals = [TARGET_COL] + unknown_covariates

    # Keep order stable and unique.
    seen: set = set()
    unknown_reals = [c for c in unknown_reals if not (c in seen or seen.add(c))]
    return known_categoricals, known_reals, unknown_reals


def filter_eligible_symbols(
    df: pd.DataFrame,
    encoder_length: int,
    prediction_length: int,
) -> List[str]:
    train_mask, val_mask, test_mask = get_split_masks(df)

    min_train_rows = encoder_length + prediction_length + 1

    train_counts = df.loc[train_mask].groupby(SYMBOL_COL).size()
    val_counts = df.loc[val_mask].groupby(SYMBOL_COL).size()
    test_counts = df.loc[test_mask].groupby(SYMBOL_COL).size()

    eligible = []
    for symbol in sorted(df[SYMBOL_COL].unique()):
        if int(train_counts.get(symbol, 0)) < min_train_rows:
            continue
        if int(val_counts.get(symbol, 0)) < prediction_length:
            continue
        if int(test_counts.get(symbol, 0)) < prediction_length:
            continue
        eligible.append(symbol)

    return eligible


def build_datasets_for_window(
    df: pd.DataFrame,
    encoder_length: int,
    prediction_length: int,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame]:
    eligible_symbols = filter_eligible_symbols(
        df=df,
        encoder_length=encoder_length,
        prediction_length=prediction_length,
    )
    if not eligible_symbols:
        raise ValueError(
            f"No eligible symbols for window={encoder_length}. "
            "Check split dates and minimum history."
        )

    work_df = df[df[SYMBOL_COL].isin(eligible_symbols)].copy()
    known_categoricals, known_reals, unknown_reals = choose_model_features(work_df)

    train_mask, _, _ = get_split_masks(work_df)
    train_df = work_df.loc[train_mask].copy()

    val_start_idx = int(
        work_df.loc[work_df[DATE_COL] >= pd.Timestamp(VAL_START), TIME_IDX_COL].min()
    )
    test_start_idx = int(
        work_df.loc[work_df[DATE_COL] >= pd.Timestamp(TEST_START), TIME_IDX_COL].min()
    )
    val_df = work_df.loc[work_df[DATE_COL] <= pd.Timestamp(VAL_END)].copy()
    test_df = work_df.copy()

    # Drop raw OHLCV columns from model covariates (but keep DATE_COL in work_df).
    drop_cols = [
        c for c in RAW_FEATURE_DROP if c in train_df.columns and c != DATE_COL
    ]
    train_model = train_df.drop(columns=drop_cols, errors="ignore")
    val_model = val_df.drop(columns=drop_cols, errors="ignore")
    test_model = test_df.drop(columns=drop_cols, errors="ignore")

    training = TimeSeriesDataSet(
        train_model,
        time_idx=TIME_IDX_COL,
        target=TARGET_COL,
        group_ids=[SYMBOL_COL],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=prediction_length,
        max_prediction_length=prediction_length,
        static_categoricals=[SYMBOL_COL],
        time_varying_known_categoricals=known_categoricals,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=GroupNormalizer(groups=[SYMBOL_COL], method="standard"),
        add_relative_time_idx=False,
        add_target_scales=True,
        add_encoder_length=True,
        # Some symbols have legitimate gaps in the trading timeline (listing changes,
        # corporate actions, data holes).  Enable this to let TFT build sequences
        # without hard-failing on non-unit time_idx jumps.
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_model,
        min_prediction_idx=val_start_idx,
        stop_randomization=True,
    )
    test = TimeSeriesDataSet.from_dataset(
        training,
        test_model,
        min_prediction_idx=test_start_idx,
        stop_randomization=True,
    )

    return training, validation, test, work_df


# ── Prediction helpers ────────────────────────────────────────────────────────

def unpack_prediction_output(
    pred_output,
) -> Tuple[np.ndarray, pd.DataFrame]:
    preds = None
    index_df = None

    if hasattr(pred_output, "prediction"):
        preds = pred_output.prediction
        index_df = getattr(pred_output, "index", None)
    elif hasattr(pred_output, "output"):
        preds = pred_output.output
        index_df = getattr(pred_output, "index", None)
    elif isinstance(pred_output, tuple):
        for item in pred_output:
            if isinstance(item, pd.DataFrame):
                index_df = item
            elif torch.is_tensor(item) or isinstance(item, np.ndarray):
                preds = item
            elif hasattr(item, "prediction"):
                preds = item.prediction
                index_df = getattr(item, "index", index_df)
            elif hasattr(item, "output"):
                preds = item.output
                index_df = getattr(item, "index", index_df)
    elif torch.is_tensor(pred_output) or isinstance(pred_output, np.ndarray):
        preds = pred_output

    if preds is None:
        raise RuntimeError(
            "Could not extract predictions from model.predict(...) output."
        )
    if index_df is None:
        raise RuntimeError(
            "Could not extract index dataframe from model.predict(...) output."
        )

    pred_np = to_numpy(preds)

    # Quantile output fallback: choose median quantile.
    if pred_np.ndim == 3:
        pred_np = pred_np[:, :, pred_np.shape[-1] // 2]
    elif pred_np.ndim == 1:
        pred_np = pred_np[:, None]

    return pred_np, index_df.copy()


def find_symbol_col(index_df: pd.DataFrame) -> str:
    candidates = [SYMBOL_COL, "__group_id__Symbol", "symbol", "__group_id__symbol"]
    for c in candidates:
        if c in index_df.columns:
            return c
    obj_cols = [c for c in index_df.columns if index_df[c].dtype == "object"]
    if obj_cols:
        return obj_cols[0]
    raise RuntimeError(
        f"Could not identify symbol column in prediction index columns: "
        f"{index_df.columns.tolist()}"
    )


def find_time_col(index_df: pd.DataFrame) -> str:
    candidates = [TIME_IDX_COL, "decoder_time_idx", "__time_idx__"]
    for c in candidates:
        if c in index_df.columns:
            return c
    fuzzy = [c for c in index_df.columns if "time_idx" in c]
    if fuzzy:
        return fuzzy[0]
    raise RuntimeError(
        f"Could not identify time_idx column in prediction index columns: "
        f"{index_df.columns.tolist()}"
    )


def build_truth_and_price_matrices(
    pred_returns: np.ndarray,
    pred_index: pd.DataFrame,
    base_df: pd.DataFrame,
    prediction_length: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if pred_returns.ndim == 1:
        pred_returns = pred_returns[:, None]

    symbol_col = find_symbol_col(pred_index)
    time_col = find_time_col(pred_index)

    symbols = pred_index[symbol_col].astype(str).to_numpy()
    start_idx = pred_index[time_col].astype(np.int64).to_numpy()

    n = pred_returns.shape[0]
    if len(symbols) != n:
        m = min(len(symbols), n)
        symbols = symbols[:m]
        start_idx = start_idx[:m]
        pred_returns = pred_returns[:m]
        n = m

    lookup_target = base_df.set_index([SYMBOL_COL, TIME_IDX_COL])[TARGET_COL].to_dict()
    lookup_price = base_df.set_index([SYMBOL_COL, TIME_IDX_COL])[PRICE_COL].to_dict()
    lookup_date = (
        base_df.drop_duplicates(TIME_IDX_COL)
        .set_index(TIME_IDX_COL)[DATE_COL]
        .to_dict()
    )

    true_returns = np.full((n, prediction_length), np.nan, dtype=np.float64)
    for h in range(prediction_length):
        ti = start_idx + h
        vals = [
            lookup_target.get((sym, int(t)), np.nan)
            for sym, t in zip(symbols, ti)
        ]
        true_returns[:, h] = np.asarray(vals, dtype=np.float64)

    # Use the encoder end price (time_idx - 1) as base for strict no-lookahead
    # price reconstruction.
    base_prices = np.asarray(
        [
            lookup_price.get((sym, int(t) - 1), np.nan)
            for sym, t in zip(symbols, start_idx)
        ],
        dtype=np.float64,
    )
    pred_prices = base_prices[:, None] * np.cumprod(1.0 + pred_returns, axis=1)
    true_prices = base_prices[:, None] * np.cumprod(1.0 + true_returns, axis=1)

    start_dates = np.asarray(
        [lookup_date.get(int(t), pd.NaT) for t in start_idx]
    )
    return symbols, start_idx, start_dates, true_returns, pred_prices, true_prices


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRAINING & TESTING
# ══════════════════════════════════════════════════════════════════════════════

class WindowStateCallback(Callback):
    """Lightning callback that writes crash-safe state JSON after every epoch."""

    def __init__(self, state_path: Path, checkpoint_dir: Path, window: int) -> None:
        super().__init__()
        self.state_path = state_path
        self.checkpoint_dir = checkpoint_dir
        self.window = window

    def _write(
        self, trainer: pl.Trainer, status: str, message: str = ""
    ) -> None:
        latest_ckpt = find_latest_checkpoint(self.checkpoint_dir)
        payload = read_json(self.state_path)
        payload.update(
            {
                "window": self.window,
                "status": status,
                "last_epoch_completed": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                "last_ckpt": (
                    str(latest_ckpt)
                    if latest_ckpt
                    else payload.get("last_ckpt", "")
                ),
                "message": message,
                "updated_at": now_utc_iso(),
            }
        )
        write_json(self.state_path, payload)

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._write(trainer, status="training")

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._write(trainer, status="training")

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        self._write(trainer, status="interrupted", message=str(exception))


@dataclass
class RunConfig:
    data_path: Path
    artifact_root: Path
    windows: List[int]
    prediction_length: int
    max_epochs: int
    patience: int
    num_workers: int
    seed: int
    learning_rate: float
    hidden_size: int
    hidden_continuous_size: int
    attention_head_size: int
    lstm_layers: int
    dropout: float
    gradient_clip_val: float
    force_retrain: bool
    limit_val_batches: int
    accumulate_grad_batches: int


def compute_window_metrics(
    window: int,
    true_returns: np.ndarray,
    pred_returns: np.ndarray,
    true_prices: np.ndarray,
    pred_prices: np.ndarray,
) -> Dict[str, float]:
    if true_returns.ndim == 1:
        true_returns = true_returns[:, None]
    if pred_returns.ndim == 1:
        pred_returns = pred_returns[:, None]
    if true_prices.ndim == 1:
        true_prices = true_prices[:, None]
    if pred_prices.ndim == 1:
        pred_prices = pred_prices[:, None]

    y_true_ret = true_returns.reshape(-1)
    y_pred_ret = pred_returns.reshape(-1)
    valid_ret = np.isfinite(y_true_ret) & np.isfinite(y_pred_ret)
    if not np.any(valid_ret):
        raise RuntimeError(
            f"window={window}: no valid return samples for pooled metric computation."
        )

    y_true_ret = y_true_ret[valid_ret]
    y_pred_ret = y_pred_ret[valid_ret]
    reg = compute_regression_metrics(y_true_ret, y_pred_ret)
    cls = compute_direction_metrics(y_true_ret, y_pred_ret)

    y_true_price = true_prices.reshape(-1)
    y_pred_price = pred_prices.reshape(-1)
    valid_price = np.isfinite(y_true_price) & np.isfinite(y_pred_price)
    if not np.any(valid_price):
        raise RuntimeError(
            f"window={window}: no valid price samples for MAPE computation."
        )

    mape = compute_mape(y_true_price[valid_price], y_pred_price[valid_price])
    return {
        "window": int(window),
        "mse": reg["mse"],
        "mae": reg["mae"],
        "rmse": reg["rmse"],
        "mape": mape,
        "directional_accuracy": cls["directional_accuracy"],
        "precision_up": cls["precision_up"],
        "recall_up": cls["recall_up"],
        "f1_up": cls["f1_up"],
    }


def save_prediction_audit(
    window_dir: Path,
    symbols: np.ndarray,
    start_idx: np.ndarray,
    start_dates: np.ndarray,
    pred_returns: np.ndarray,
    true_returns: np.ndarray,
    pred_prices: np.ndarray,
    true_prices: np.ndarray,
) -> None:
    if pred_returns.ndim == 2:
        pred_returns = pred_returns[:, 0]
    if true_returns.ndim == 2:
        true_returns = true_returns[:, 0]
    if pred_prices.ndim == 2:
        pred_prices = pred_prices[:, 0]
    if true_prices.ndim == 2:
        true_prices = true_prices[:, 0]

    pred_df = pd.DataFrame(
        {
            "Symbol": symbols,
            "decoder_start_time_idx": start_idx,
            "decoder_start_date": start_dates,
            "pred_return": pred_returns,
            "true_return": true_returns,
            "pred_price": pred_prices,
            "true_price": true_prices,
        }
    )
    pred_df.to_csv(window_dir / "test_predictions.csv", index=False)


def evaluate_window_and_write_outputs(
    window: int,
    prediction_length: int,
    work_df: pd.DataFrame,
    test_loader,
    best_ckpt: str,
    window_dir: Path,
    metrics_path: Path,
    metrics_rows: List[Dict],
    state_path: Path,
) -> None:
    logging.info("window=%s test prediction start", window)
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    pred_output = best_model.predict(
        test_loader,
        mode="prediction",
        return_index=True,
        trainer_kwargs={"logger": False, "enable_checkpointing": False},
    )
    pred_returns, pred_index = unpack_prediction_output(pred_output)
    if pred_returns.shape[1] < prediction_length:
        raise RuntimeError(
            f"window={window}: predicted horizon {pred_returns.shape[1]} "
            f"< expected {prediction_length}"
        )
    pred_returns = pred_returns[:, :prediction_length].astype(np.float64)

    symbols, start_idx, start_dates, true_returns, pred_prices, true_prices = (
        build_truth_and_price_matrices(
            pred_returns=pred_returns,
            pred_index=pred_index,
            base_df=work_df[
                [SYMBOL_COL, TIME_IDX_COL, TARGET_COL, PRICE_COL, DATE_COL]
            ].copy(),
            prediction_length=prediction_length,
        )
    )

    window_metric = compute_window_metrics(
        window=window,
        true_returns=true_returns,
        pred_returns=pred_returns,
        true_prices=true_prices,
        pred_prices=pred_prices,
    )
    metrics_df = pd.DataFrame([window_metric], columns=FINAL_METRIC_COLUMNS)
    metrics_df.to_csv(metrics_path, index=False)
    metrics_rows.extend(metrics_df.to_dict(orient="records"))

    save_prediction_audit(
        window_dir=window_dir,
        symbols=symbols,
        start_idx=start_idx,
        start_dates=start_dates,
        pred_returns=pred_returns,
        true_returns=true_returns,
        pred_prices=pred_prices,
        true_prices=true_prices,
    )

    write_json(
        state_path,
        {
            "window": window,
            "status": "completed",
            "updated_at": now_utc_iso(),
            "completed_at": now_utc_iso(),
            "best_ckpt": best_ckpt,
            "metrics_path": str(metrics_path),
            "prediction_audit_path": str(window_dir / "test_predictions.csv"),
        },
    )
    logging.info(
        "window=%s completed (metrics rows=%s).", window, len(metrics_df)
    )


def run_window_training(
    cfg: RunConfig,
    base_df: pd.DataFrame,
    window: int,
    metrics_rows: List[Dict],
) -> None:
    log = logging.getLogger(__name__)

    window_dir = cfg.artifact_root / f"window_{window}"
    checkpoints_dir = window_dir / "checkpoints"
    logs_dir = window_dir / "logs"
    state_path = window_dir / "state.json"
    metrics_path = window_dir / "metrics.csv"

    ensure_dir(window_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(logs_dir)

    state = read_json(state_path)
    metrics_only_recompute = False
    if (
        state.get("status") == "completed"
        and metrics_path.exists()
        and not cfg.force_retrain
    ):
        existing = pd.read_csv(metrics_path)
        exact_schema = list(existing.columns) == FINAL_METRIC_COLUMNS
        expected_rows = len(existing) == 1
        no_missing_values = (
            exact_schema
            and expected_rows
            and not existing[FINAL_METRIC_COLUMNS].isna().any().any()
        )
        if no_missing_values:
            log.info(
                "window=%s already completed with up-to-date metrics, skipping.",
                window,
            )
            metrics_rows.extend(
                existing[FINAL_METRIC_COLUMNS].to_dict(orient="records")
            )
            return
        else:
            log.warning(
                "window=%s existing metrics file is outdated "
                "(schema_ok=%s rows=%s has_nan=%s). Recomputing this window.",
                window,
                exact_schema,
                len(existing),
                bool(existing[FINAL_METRIC_COLUMNS].isna().any().any())
                if exact_schema and expected_rows
                else True,
            )
            metrics_only_recompute = True
    elif state.get("status") == "completed" and not cfg.force_retrain:
        log.warning(
            "window=%s marked completed but metrics file missing. "
            "Recomputing this window.",
            window,
        )
        metrics_only_recompute = True

    training_ds, val_ds, test_ds, work_df = build_datasets_for_window(
        df=base_df,
        encoder_length=window,
        prediction_length=cfg.prediction_length,
    )
    log.info(
        "window=%s eligible_symbols=%s train_rows=%s val_rows=%s test_rows=%s",
        window,
        work_df[SYMBOL_COL].nunique(),
        int((work_df[DATE_COL] <= pd.Timestamp(TRAIN_END)).sum()),
        int(
            (
                (work_df[DATE_COL] >= pd.Timestamp(VAL_START))
                & (work_df[DATE_COL] <= pd.Timestamp(VAL_END))
            ).sum()
        ),
        int((work_df[DATE_COL] >= pd.Timestamp(TEST_START)).sum()),
    )

    batch_size = WINDOW_BATCH_SIZE.get(window, 64)
    test_loader = test_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    if metrics_only_recompute and not cfg.force_retrain:
        eval_ckpt = None
        for ckpt_key in ("best_ckpt", "last_ckpt"):
            ck = state.get(ckpt_key)
            if ck and Path(ck).exists():
                eval_ckpt = str(Path(ck))
                break
        if eval_ckpt is None:
            latest = find_latest_checkpoint(checkpoints_dir)
            if latest is not None:
                eval_ckpt = str(latest)

        if eval_ckpt is not None:
            log.info(
                "window=%s metrics-only recompute using checkpoint: %s",
                window,
                eval_ckpt,
            )
            evaluate_window_and_write_outputs(
                window=window,
                prediction_length=cfg.prediction_length,
                work_df=work_df,
                test_loader=test_loader,
                best_ckpt=eval_ckpt,
                window_dir=window_dir,
                metrics_path=metrics_path,
                metrics_rows=metrics_rows,
                state_path=state_path,
            )
            return

        log.warning(
            "window=%s requested metrics-only recompute but no checkpoint found. "
            "Falling back to training.",
            window,
        )

    train_loader = training_ds.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=cfg.patience,
        min_delta=1e-4,
        verbose=False,
    )
    best_ckpt_cb = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="best-epoch{epoch:03d}-valloss{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        auto_insert_metric_name=False,
    )
    step_ckpt_cb = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="step-{step:09d}",
        monitor=None,
        save_top_k=-1,
        every_n_train_steps=1000,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )
    state_cb = WindowStateCallback(
        state_path=state_path, checkpoint_dir=checkpoints_dir, window=window
    )
    progress_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(logs_dir), name="lightning")

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        max_epochs=cfg.max_epochs,
        logger=csv_logger,
        callbacks=[early_stop, best_ckpt_cb, step_ckpt_cb, state_cb, progress_cb],
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        limit_val_batches=cfg.limit_val_batches,
        deterministic=False,
        benchmark=True,
        enable_model_summary=False,
        log_every_n_steps=50,
        num_sanity_val_steps=0,
    )

    model = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        attention_head_size=cfg.attention_head_size,
        hidden_continuous_size=cfg.hidden_continuous_size,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
        loss=QuantileLoss(),
        output_size=7,
        # FP16-safe attention mask bias; default can overflow in 16-mixed on some GPUs.
        mask_bias=-1e4,
        log_interval=-1,
        log_val_interval=-1,
        reduce_on_plateau_patience=4,
    )

    resume_ckpt = None
    if not cfg.force_retrain:
        state_ckpt = state.get("last_ckpt")
        if state_ckpt and Path(state_ckpt).exists():
            resume_ckpt = str(Path(state_ckpt))
        else:
            latest = find_latest_checkpoint(checkpoints_dir)
            if latest is not None:
                resume_ckpt = str(latest)

    write_json(
        state_path,
        {
            "window": window,
            "status": "training",
            "started_at": now_utc_iso(),
            "last_ckpt": resume_ckpt or "",
            "max_epochs": cfg.max_epochs,
        },
    )

    try:
        logging.info(
            "window=%s training start "
            "(max_epochs=%s, batch=%s, accumulate_grad_batches=%s, limit_val_batches=%s)",
            window,
            cfg.max_epochs,
            batch_size,
            cfg.accumulate_grad_batches,
            cfg.limit_val_batches,
        )
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_ckpt,
        )
    except KeyboardInterrupt:
        latest = find_latest_checkpoint(checkpoints_dir)
        write_json(
            state_path,
            {
                "window": window,
                "status": "interrupted",
                "updated_at": now_utc_iso(),
                "last_ckpt": str(latest) if latest else "",
                "last_epoch_completed": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                "message": "KeyboardInterrupt",
            },
        )
        raise
    except Exception as exc:
        latest = find_latest_checkpoint(checkpoints_dir)
        write_json(
            state_path,
            {
                "window": window,
                "status": "failed",
                "updated_at": now_utc_iso(),
                "last_ckpt": str(latest) if latest else "",
                "last_epoch_completed": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise

    best_ckpt = best_ckpt_cb.best_model_path
    if not best_ckpt:
        latest = find_latest_checkpoint(checkpoints_dir)
        if latest is None:
            raise RuntimeError(
                f"window={window}: no checkpoint found after training."
            )
        best_ckpt = str(latest)

    write_json(
        state_path,
        {
            "window": window,
            "status": "trained",
            "updated_at": now_utc_iso(),
            "best_ckpt": best_ckpt,
            "last_epoch_completed": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "best_score": float(best_ckpt_cb.best_model_score.item())
            if best_ckpt_cb.best_model_score is not None
            else None,
        },
    )

    evaluate_window_and_write_outputs(
        window=window,
        prediction_length=cfg.prediction_length,
        work_df=work_df,
        test_loader=test_loader,
        best_ckpt=best_ckpt,
        window_dir=window_dir,
        metrics_path=metrics_path,
        metrics_rows=metrics_rows,
        state_path=state_path,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TuneConfig:
    data_path: Path
    artifact_root: Path
    windows: List[int]
    prediction_length: int
    n_trials: int
    max_total_trials: Optional[int]
    max_epochs: int
    patience: int
    num_workers: int
    seed: int
    study_prefix: str
    timeout_seconds: Optional[int]
    accumulate_grad_batches: int
    limit_val_batches: int
    eval_test_metrics: bool


def pick_precision_sequence() -> List[str]:
    """
    Prefer 16-mixed on CUDA for RTX 3050-class GPUs.
    Keep a deterministic order so effective precision is recorded per trial.
    """
    if torch.cuda.is_available():
        return ["16-mixed"]
    return ["32-true"]


def pick_loader_workers(requested_workers: int) -> Tuple[int, int]:
    """
    Choose worker counts for train and eval dataloaders.
    If requested_workers < 0, auto-tune based on host CPU.
    """
    if requested_workers < 0:
        train_workers = 4
    else:
        train_workers = max(0, int(requested_workers))
    eval_workers = max(0, train_workers // 2)
    return train_workers, eval_workers


def is_probable_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def is_probable_bf16_issue(exc: BaseException) -> bool:
    msg = str(exc).lower()
    keywords = ["bf16", "bfloat16", "bfloat", "unsupported", "amp"]
    return any(k in msg for k in keywords)


def compute_per_symbol_equal_direction_metrics(
    true_returns: np.ndarray,
    pred_returns: np.ndarray,
    symbols: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute per-symbol-equal-weight Directional Accuracy and F1_Up on return space.
    For 1-step predictions, we use the first horizon column.
    """
    if true_returns.ndim == 2:
        y_true = true_returns[:, 0]
    else:
        y_true = true_returns.reshape(-1)

    if pred_returns.ndim == 2:
        y_pred = pred_returns[:, 0]
    else:
        y_pred = pred_returns.reshape(-1)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return float("nan"), float("nan")

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    sym_valid = symbols[valid]

    per_symbol_da: List[float] = []
    per_symbol_f1: List[float] = []
    for sym in np.unique(sym_valid):
        m = sym_valid == sym
        if not np.any(m):
            continue
        d = compute_direction_metrics(y_true[m], y_pred[m])
        per_symbol_da.append(float(d["directional_accuracy"]))
        per_symbol_f1.append(float(d["f1_up"]))

    if not per_symbol_da:
        return float("nan"), float("nan")

    return float(np.mean(per_symbol_da)), float(np.mean(per_symbol_f1))


class WindowObjective:
    """Optuna objective for a single encoder window."""

    def __init__(
        self,
        cfg: TuneConfig,
        window: int,
        train_ds: TimeSeriesDataSet,
        val_ds: TimeSeriesDataSet,
        test_ds: TimeSeriesDataSet,
        work_df: pd.DataFrame,
        window_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.window = window
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.work_df = work_df
        self.window_dir = window_dir

    def __call__(self, trial: optuna.trial.Trial) -> float:
        batch_size = WINDOW_BATCH_SIZE.get(self.window, 64)
        train_workers, eval_workers = pick_loader_workers(self.cfg.num_workers)
        train_loader_kwargs: Dict = {
            "train": True,
            "batch_size": batch_size,
            "num_workers": train_workers,
            "persistent_workers": train_workers > 0,
            "pin_memory": torch.cuda.is_available(),
        }
        eval_loader_kwargs: Dict = {
            "train": False,
            "batch_size": batch_size,
            "num_workers": eval_workers,
            "persistent_workers": eval_workers > 0,
            "pin_memory": torch.cuda.is_available(),
        }
        if train_workers > 0:
            train_loader_kwargs["prefetch_factor"] = 2
            train_loader_kwargs["multiprocessing_context"] = "fork"
        if eval_workers > 0:
            eval_loader_kwargs["prefetch_factor"] = 2
            eval_loader_kwargs["multiprocessing_context"] = "fork"

        # ── Search space ────────────────────────────────────────────────────
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_categorical("hidden_size_v2", [16, 32, 48])
        hidden_continuous_size_raw = trial.suggest_categorical(
            "hidden_continuous_size_v2", [8, 16, 24]
        )
        attention_head_size_raw = trial.suggest_categorical(
            "attention_head_size_v2", [1, 2, 4]
        )

        # Keep Optuna distributions static across trials, then map raw samples to
        # values compatible with this trial's hidden_size.
        hc_candidates = [x for x in [8, 16, 24] if x <= hidden_size]
        hidden_continuous_size = max(
            [x for x in hc_candidates if x <= hidden_continuous_size_raw],
            default=hc_candidates[0],
        )

        head_candidates = [h for h in [1, 2, 4] if hidden_size % h == 0]
        attention_head_size = max(
            [h for h in head_candidates if h <= attention_head_size_raw],
            default=head_candidates[0],
        )

        if hidden_continuous_size != hidden_continuous_size_raw:
            trial.set_user_attr(
                "hidden_continuous_size_adjusted_from", int(hidden_continuous_size_raw)
            )
        if attention_head_size != attention_head_size_raw:
            trial.set_user_attr(
                "attention_head_size_adjusted_from", int(attention_head_size_raw)
            )
        trial.set_user_attr("effective_hidden_size", int(hidden_size))
        trial.set_user_attr(
            "effective_hidden_continuous_size", int(hidden_continuous_size)
        )
        trial.set_user_attr(
            "effective_attention_head_size", int(attention_head_size)
        )

        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.1, 1.0)

        trial_dir = self.window_dir / "trials" / f"trial_{trial.number:05d}"
        checkpoints_dir = trial_dir / "checkpoints"
        ensure_dir(checkpoints_dir)

        train_loader = self.train_ds.to_dataloader(**train_loader_kwargs)
        val_loader = self.val_ds.to_dataloader(**eval_loader_kwargs)
        test_loader = (
            self.test_ds.to_dataloader(**eval_loader_kwargs)
            if self.cfg.eval_test_metrics
            else None
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=self.cfg.patience,
            min_delta=1e-4,
            verbose=False,
        )
        best_ckpt_cb = ModelCheckpoint(
            dirpath=str(checkpoints_dir),
            filename="best-epoch{epoch:03d}-valloss{val_loss:.6f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
            auto_insert_metric_name=False,
        )
        prune_cb = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        progress_cb = TQDMProgressBar(refresh_rate=50)

        precision_attempts = pick_precision_sequence()
        last_exc: Optional[BaseException] = None
        best_score: Optional[float] = None
        best_ckpt_path: Optional[str] = None
        effective_precision: Optional[str] = None

        for idx, precision in enumerate(precision_attempts):
            model = None
            trainer = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                model = TemporalFusionTransformer.from_dataset(
                    self.train_ds,
                    learning_rate=learning_rate,
                    hidden_size=hidden_size,
                    attention_head_size=attention_head_size,
                    hidden_continuous_size=hidden_continuous_size,
                    lstm_layers=1 if self.window <= 10 else 2,
                    dropout=dropout,
                    loss=QuantileLoss(),
                    output_size=7,
                    mask_bias=-1e4,
                    log_interval=-1,
                    log_val_interval=-1,
                    reduce_on_plateau_patience=2,
                )

                trainer = pl.Trainer(
                    accelerator="auto",
                    devices=1,
                    precision=precision,
                    max_epochs=self.cfg.max_epochs,
                    logger=False,
                    callbacks=[early_stop, best_ckpt_cb, prune_cb, progress_cb],
                    gradient_clip_val=gradient_clip_val,
                    accumulate_grad_batches=self.cfg.accumulate_grad_batches,
                    limit_val_batches=self.cfg.limit_val_batches,
                    deterministic=False,
                    benchmark=True,
                    enable_model_summary=False,
                    log_every_n_steps=100,
                    num_sanity_val_steps=0,
                    enable_checkpointing=True,
                )

                trainer.fit(
                    model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                )

                if best_ckpt_cb.best_model_score is not None:
                    best_score = float(best_ckpt_cb.best_model_score.item())
                else:
                    v = trainer.callback_metrics.get("val_loss")
                    best_score = (
                        float(v.item()) if v is not None else float("inf")
                    )

                if best_ckpt_cb.best_model_path:
                    best_ckpt_path = best_ckpt_cb.best_model_path
                else:
                    # Fallback to last checkpoint in trial directory.
                    ckpts = sorted(
                        checkpoints_dir.glob("*.ckpt"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    best_ckpt_path = str(ckpts[0]) if ckpts else None

                effective_precision = precision
                break

            except optuna.TrialPruned:
                # Keep pruning behaviour strict.
                raise
            except RuntimeError as exc:
                last_exc = exc
                if is_probable_oom(exc):
                    trial.set_user_attr("oom", True)
                    trial.set_user_attr("failed_precision", precision)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned("Trial pruned due to OOM")

                # Only fall back to next precision attempt if this looks
                # precision-related and more attempts remain.
                if idx < len(precision_attempts) - 1 and is_probable_bf16_issue(exc):
                    trial.set_user_attr("bf16_fallback", True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                raise
            finally:
                if trainer is not None:
                    del trainer
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if best_score is None:
            if last_exc is not None:
                raise RuntimeError(
                    f"Trial failed before producing val_loss: {last_exc}"
                )
            raise RuntimeError("Trial failed before producing val_loss")

        trial.set_user_attr("effective_precision", effective_precision)
        if best_ckpt_path:
            trial.set_user_attr("best_checkpoint", best_ckpt_path)

        # Optional extra metrics (expensive). Disabled by default for faster tuning.
        if (
            self.cfg.eval_test_metrics
            and best_ckpt_path
            and test_loader is not None
        ):
            best_model = TemporalFusionTransformer.load_from_checkpoint(
                best_ckpt_path
            )
            pred_output = best_model.predict(
                test_loader,
                mode="prediction",
                return_index=True,
                trainer_kwargs={"logger": False, "enable_checkpointing": False},
            )
            pred_returns, pred_index = unpack_prediction_output(pred_output)
            pred_returns = pred_returns[
                :, : self.cfg.prediction_length
            ].astype(np.float64)

            symbols, _, _, true_returns, _, _ = build_truth_and_price_matrices(
                pred_returns=pred_returns,
                pred_index=pred_index,
                base_df=self.work_df[
                    [SYMBOL_COL, TIME_IDX_COL, TARGET_COL, PRICE_COL, DATE_COL]
                ].copy(),
                prediction_length=self.cfg.prediction_length,
            )
            da_eq, f1_eq = compute_per_symbol_equal_direction_metrics(
                true_returns=true_returns,
                pred_returns=pred_returns,
                symbols=symbols,
            )
            trial.set_user_attr("Directional_Accuracy", da_eq)
            trial.set_user_attr("F1_Up", f1_eq)
            del best_model

        del train_loader, val_loader, test_loader
        gc.collect()
        return best_score


def export_trials_csv(study: optuna.Study, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    df = study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "state",
            "params",
            "user_attrs",
            "datetime_start",
            "datetime_complete",
        )
    )
    df.to_csv(out_path, index=False)


def print_window_summary(study: optuna.Study, window: int) -> None:
    log = logging.getLogger(__name__)
    if study.best_trial is None:
        log.warning("window=%s: no best trial available.", window)
        return

    best = study.best_trial
    log.info(
        "window=%s best trial=%s val_loss=%.8f",
        window,
        best.number,
        float(best.value),
    )
    log.info("window=%s best params=%s", window, best.params)
    log.info(
        "window=%s best attrs: F1_Up=%s Directional_Accuracy=%s effective_precision=%s",
        window,
        best.user_attrs.get("F1_Up"),
        best.user_attrs.get("Directional_Accuracy"),
        best.user_attrs.get("effective_precision"),
    )

    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    def top_trials_by_attr(attr: str, n: int = 5) -> list:
        scored = [
            t
            for t in completed
            if attr in t.user_attrs and t.user_attrs.get(attr) is not None
        ]
        scored = [
            t
            for t in scored
            if isinstance(t.user_attrs.get(attr), (int, float))
            and np.isfinite(t.user_attrs.get(attr))
        ]
        scored.sort(key=lambda t: float(t.user_attrs[attr]), reverse=True)
        return scored[:n]

    top_f1 = top_trials_by_attr("F1_Up")
    top_da = top_trials_by_attr("Directional_Accuracy")

    if top_f1:
        log.info("window=%s top trials by F1_Up:", window)
        for t in top_f1:
            log.info(
                "  trial=%s F1_Up=%.6f val_loss=%.8f",
                t.number,
                float(t.user_attrs["F1_Up"]),
                float(t.value),
            )

    if top_da:
        log.info("window=%s top trials by Directional_Accuracy:", window)
        for t in top_da:
            log.info(
                "  trial=%s Directional_Accuracy=%.6f val_loss=%.8f",
                t.number,
                float(t.user_attrs["Directional_Accuracy"]),
                float(t.value),
            )


def tune_window(cfg: TuneConfig, base_df: pd.DataFrame, window: int) -> Dict:
    log = logging.getLogger(__name__)

    window_dir = cfg.artifact_root / f"window_{window}"
    ensure_dir(window_dir)

    train_ds, val_ds, test_ds, work_df = build_datasets_for_window(
        df=base_df,
        encoder_length=window,
        prediction_length=cfg.prediction_length,
    )

    study_name = f"{cfg.study_prefix}{window}"
    study_db_path = window_dir / "study.db"
    storage_url = (
        f"sqlite:///{study_db_path.resolve()}?timeout=30&_journal_mode=WAL"
    )

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=3, interval_steps=1
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = WindowObjective(
        cfg=cfg,
        window=window,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        work_df=work_df,
        window_dir=window_dir,
    )

    existing_trials = len(study.trials)
    n_trials_to_run = cfg.n_trials
    if cfg.max_total_trials is not None:
        remaining = max(0, int(cfg.max_total_trials) - existing_trials)
        n_trials_to_run = min(n_trials_to_run, remaining)

    log.info(
        "window=%s tuning start "
        "(existing_trials=%s, n_trials_this_run=%s, max_total_trials=%s, "
        "timeout_seconds=%s)",
        window,
        existing_trials,
        n_trials_to_run,
        cfg.max_total_trials,
        cfg.timeout_seconds,
    )

    if n_trials_to_run <= 0:
        log.info(
            "window=%s already reached max_total_trials=%s "
            "(existing_trials=%s). Skipping optimize.",
            window,
            cfg.max_total_trials,
            existing_trials,
        )
        export_trials_csv(study, window_dir / "trials.csv")
        print_window_summary(study, window)
        return {
            "window": window,
            "study_name": study_name,
            "study_db": str(study_db_path),
            "n_trials_total": len(study.trials),
            "best_val_loss": float(study.best_trial.value)
            if study.best_trial is not None
            else None,
            "best_trial_number": int(study.best_trial.number)
            if study.best_trial is not None
            else None,
        }

    stop_state: Dict = {"requested": False, "sigint_count": 0}
    prev_sigint_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame) -> None:
        stop_state["sigint_count"] += 1
        if stop_state["sigint_count"] == 1:
            stop_state["requested"] = True
            log.warning(
                "window=%s SIGINT received: will stop after current trial "
                "completes (press Ctrl+C again to force stop).",
                window,
            )
            return
        # Second Ctrl+C: force immediate interruption.
        raise KeyboardInterrupt

    def _stop_after_trial_cb(
        study_: optuna.Study, trial_: optuna.trial.FrozenTrial
    ) -> None:
        if stop_state["requested"]:
            study_.stop()

    try:
        signal.signal(signal.SIGINT, _sigint_handler)
        study.optimize(
            objective,
            n_trials=n_trials_to_run,
            timeout=cfg.timeout_seconds,
            gc_after_trial=True,
            show_progress_bar=True,
            callbacks=[_stop_after_trial_cb],
        )
    except KeyboardInterrupt:
        log.warning(
            "window=%s tuning interrupted by user. "
            "Study is safely persisted in %s",
            window,
            study_db_path,
        )
    finally:
        signal.signal(signal.SIGINT, prev_sigint_handler)

    export_trials_csv(study, window_dir / "trials.csv")
    print_window_summary(study, window)

    best_payload: Dict = {}
    if study.best_trial is not None:
        best_payload = {
            "window": window,
            "best_trial_number": int(study.best_trial.number),
            "best_val_loss": float(study.best_trial.value),
            "best_params": dict(study.best_trial.params),
            "best_user_attrs": dict(study.best_trial.user_attrs),
            "updated_at": now_utc_iso(),
            "study_db": str(study_db_path),
        }
        write_json(window_dir / "best_trial.json", best_payload)

    return {
        "window": window,
        "study_name": study_name,
        "study_db": str(study_db_path),
        "n_trials_total": len(study.trials),
        "best_val_loss": float(study.best_trial.value)
        if study.best_trial is not None
        else None,
        "best_trial_number": int(study.best_trial.number)
        if study.best_trial is not None
        else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — UNIFIED CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedConfig:
    """Single config that encompasses all fields from RunConfig and TuneConfig.

    Fields that have different defaults between training and tuning are given
    distinct names (e.g. ``train_max_epochs`` vs ``tune_max_epochs``) so that
    both defaults can be expressed independently in the CLI.
    """

    # ── shared ───────────────────────────────────────────────────────────────
    data_path: Path
    windows: List[int]
    prediction_length: int
    num_workers: int
    seed: int
    no_window_menu: bool

    # ── training-specific ────────────────────────────────────────────────────
    train_artifact_root: Path
    train_max_epochs: int          # default 50
    train_patience: int            # default 8
    learning_rate: float
    hidden_size: int
    hidden_continuous_size: int
    attention_head_size: int
    lstm_layers: int
    dropout: float
    gradient_clip_val: float
    force_retrain: bool
    train_limit_val_batches: int   # default 200
    train_accumulate_grad_batches: int  # default 2

    # ── tuning-specific ──────────────────────────────────────────────────────
    tune_artifact_root: Path
    tune_max_epochs: int           # default 8
    tune_patience: int             # default 3
    n_trials: int
    max_total_trials: Optional[int]
    study_prefix: str
    timeout_seconds: Optional[int]
    tune_accumulate_grad_batches: int   # default 1
    tune_limit_val_batches: int         # default 50
    eval_test_metrics: bool


def make_run_config(
    cfg: UnifiedConfig,
    windows: Optional[List[int]] = None,
    hp_overrides: Optional[Dict] = None,
) -> RunConfig:
    """
    Construct a RunConfig from UnifiedConfig.

    ``windows`` overrides cfg.windows (used to pass a single-window list per
    loop iteration while keeping the unified config intact).

    ``hp_overrides`` is a dict of model HP values sourced from a completed
    Optuna study (see ``load_tuned_hps_for_window``).  When provided, the
    corresponding RunConfig fields are replaced; all other fields use the
    UnifiedConfig values unchanged.
    """
    ovr: Dict = hp_overrides or {}
    return RunConfig(
        data_path=cfg.data_path,
        artifact_root=cfg.train_artifact_root,
        windows=windows if windows is not None else cfg.windows,
        prediction_length=cfg.prediction_length,
        max_epochs=cfg.train_max_epochs,
        patience=cfg.train_patience,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        learning_rate=float(ovr.get("learning_rate", cfg.learning_rate)),
        hidden_size=int(ovr.get("hidden_size", cfg.hidden_size)),
        hidden_continuous_size=int(
            ovr.get("hidden_continuous_size", cfg.hidden_continuous_size)
        ),
        attention_head_size=int(
            ovr.get("attention_head_size", cfg.attention_head_size)
        ),
        lstm_layers=int(ovr.get("lstm_layers", cfg.lstm_layers)),
        dropout=float(ovr.get("dropout", cfg.dropout)),
        gradient_clip_val=float(
            ovr.get("gradient_clip_val", cfg.gradient_clip_val)
        ),
        force_retrain=cfg.force_retrain,
        limit_val_batches=cfg.train_limit_val_batches,
        accumulate_grad_batches=cfg.train_accumulate_grad_batches,
    )


def make_tune_config(cfg: UnifiedConfig) -> TuneConfig:
    """Construct a TuneConfig from UnifiedConfig."""
    return TuneConfig(
        data_path=cfg.data_path,
        artifact_root=cfg.tune_artifact_root,
        windows=cfg.windows,
        prediction_length=cfg.prediction_length,
        n_trials=cfg.n_trials,
        max_total_trials=cfg.max_total_trials,
        max_epochs=cfg.tune_max_epochs,
        patience=cfg.tune_patience,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        study_prefix=cfg.study_prefix,
        timeout_seconds=cfg.timeout_seconds,
        accumulate_grad_batches=cfg.tune_accumulate_grad_batches,
        limit_val_batches=cfg.tune_limit_val_batches,
        eval_test_metrics=cfg.eval_test_metrics,
    )


def load_tuned_hps_for_window(
    window: int, tune_artifact_root: Path
) -> Optional[Dict]:
    """
    Load best hyperparameters from a completed Optuna tuning run for one window.

    Reads ``<tune_artifact_root>/window_<window>/best_trial.json`` which is
    written by ``tune_window`` after each study completes.

    Returns a dict with keys matching RunConfig model-HP fields, or None if no
    best_trial.json exists for this window.
    """
    best_trial_path = tune_artifact_root / f"window_{window}" / "best_trial.json"
    if not best_trial_path.exists():
        return None

    data = read_json(best_trial_path)
    params = data.get("best_params", {})
    user_attrs = data.get("best_user_attrs", {})

    # Effective values (post-constraint) are stored in user_attrs.
    # Fall back to raw param values if user_attrs are missing.
    hps = {
        "learning_rate": float(params.get("learning_rate", 1e-3)),
        "hidden_size": int(
            user_attrs.get(
                "effective_hidden_size", params.get("hidden_size_v2", 32)
            )
        ),
        "hidden_continuous_size": int(
            user_attrs.get(
                "effective_hidden_continuous_size",
                params.get("hidden_continuous_size_v2", 16),
            )
        ),
        "attention_head_size": int(
            user_attrs.get(
                "effective_attention_head_size",
                params.get("attention_head_size_v2", 4),
            )
        ),
        # lstm_layers was not part of the Optuna search space; replicate the
        # tuner's fixed rule (1 for short windows, 2 for longer ones).
        "lstm_layers": 1 if window <= 10 else 2,
        "dropout": float(params.get("dropout", 0.2)),
        "gradient_clip_val": float(params.get("gradient_clip_val", 0.5)),
    }
    logging.info(
        "window=%s loaded tuned HPs from %s: %s", window, best_trial_path, hps
    )
    return hps


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CLI & MAIN
# ══════════════════════════════════════════════════════════════════════════════

def show_main_menu() -> int:
    """Display the top-level interactive mode menu and return the chosen mode."""
    lines = [
        "",
        "TFT Unified Pipeline — Select Mode:",
        "  [0] Hyperparameter tuning  → then training/testing",
        "  [1] Training/testing only  (loads tuned HPs when available)",
        "  [2] Inference only",
        "  [3] Exit",
    ]
    print("\n".join(lines))

    while True:
        try:
            choice = input("Enter choice (0-3): ").strip()
        except EOFError:
            logging.info(
                "No interactive input detected. Defaulting to mode 1 "
                "(training/testing only)."
            )
            return 1

        if choice in ("0", "1", "2", "3"):
            return int(choice)

        print("Invalid choice. Please enter 0, 1, 2, or 3.")


def run_tuning_mode(cfg: UnifiedConfig, base_df: pd.DataFrame) -> None:
    """
    Run Optuna hyperparameter tuning for all selected windows, then optionally
    proceed to training/testing with the best discovered hyperparameters.
    """
    tune_cfg = make_tune_config(cfg)
    ensure_dir(tune_cfg.artifact_root)

    # Reduce file descriptor pressure from DataLoader shared-memory handles.
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    summary_rows: List[Dict] = []
    for window in tune_cfg.windows:
        try:
            row = tune_window(cfg=tune_cfg, base_df=base_df, window=window)
            summary_rows.append(row)
        except Exception as exc:
            logging.error("window=%s tuning failed: %s", window, exc)
            logging.debug("Traceback:\n%s", traceback.format_exc())
            summary_rows.append(
                {
                    "window": window,
                    "study_name": f"{tune_cfg.study_prefix}{window}",
                    "study_db": str(
                        tune_cfg.artifact_root / f"window_{window}" / "study.db"
                    ),
                    "n_trials_total": None,
                    "best_val_loss": None,
                    "best_trial_number": None,
                    "error": str(exc),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = tune_cfg.artifact_root / "tuning_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info("Saved tuning summary -> %s", summary_path)

    # Prompt whether to proceed to training with the tuned HPs.
    try:
        proceed = (
            input(
                "\nTuning complete. Proceed to training/testing with tuned HPs? [Y/n]: "
            )
            .strip()
            .lower()
        )
    except EOFError:
        proceed = "y"

    if proceed in ("", "y", "yes"):
        run_train_test_mode(cfg=cfg, base_df=base_df)
    else:
        logging.info("Skipping training/testing.")


def run_train_test_mode(cfg: UnifiedConfig, base_df: pd.DataFrame) -> None:
    """
    Run full training and testing for all selected windows.

    For each window, attempts to load previously tuned HPs from
    ``cfg.tune_artifact_root``.  If found, they override the default model
    architecture HPs for that window; otherwise the CLI defaults are used.
    """
    ensure_dir(cfg.train_artifact_root)

    # Resolve window list (apply interactive menu unless suppressed).
    windows = cfg.windows
    if not cfg.no_window_menu:
        windows = choose_windows_menu(default_windows=windows)

    all_metrics: List[Dict] = []
    for window in tqdm(windows, desc="Training windows", unit="window"):
        hp_overrides = load_tuned_hps_for_window(
            window=window, tune_artifact_root=cfg.tune_artifact_root
        )
        if hp_overrides:
            logging.info("window=%s using tuned HPs from Optuna study.", window)
        else:
            logging.info(
                "window=%s no tuned HPs found — using CLI/default HPs.", window
            )

        run_cfg = make_run_config(cfg, windows=[window], hp_overrides=hp_overrides)
        run_window_training(
            cfg=run_cfg, base_df=base_df, window=window, metrics_rows=all_metrics
        )

    if not all_metrics:
        logging.warning(
            "No metrics produced. Check state files under %s",
            cfg.train_artifact_root,
        )
        return

    summary_df = pd.DataFrame(all_metrics)
    summary_df = summary_df[FINAL_METRIC_COLUMNS].sort_values(
        ["window"], kind="mergesort"
    )
    summary_path = cfg.train_artifact_root / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info("Saved global metrics summary -> %s", summary_path)


def run_inference_only_mode(cfg: UnifiedConfig, base_df: pd.DataFrame) -> None:
    """
    Run inference only: for each selected window, locate the best checkpoint
    under ``cfg.train_artifact_root`` and write test predictions + metrics.

    No model training is performed.  Skips windows that have no checkpoint.
    """
    # Resolve window list.
    windows = cfg.windows
    if not cfg.no_window_menu:
        windows = choose_windows_menu(default_windows=windows)

    all_metrics: List[Dict] = []
    for window in tqdm(windows, desc="Inference windows", unit="window"):
        window_dir = cfg.train_artifact_root / f"window_{window}"
        checkpoints_dir = window_dir / "checkpoints"
        state_path = window_dir / "state.json"
        metrics_path = window_dir / "metrics.csv"

        # Locate best checkpoint (prefer state.json metadata, then filesystem scan).
        state = read_json(state_path) if state_path.exists() else {}
        best_ckpt = state.get("best_ckpt", "")
        if not best_ckpt or not Path(best_ckpt).exists():
            latest = find_latest_checkpoint(checkpoints_dir)
            if latest is None:
                logging.warning(
                    "window=%s no checkpoint found under %s — skipping.",
                    window,
                    checkpoints_dir,
                )
                continue
            best_ckpt = str(latest)

        logging.info(
            "window=%s inference using checkpoint: %s", window, best_ckpt
        )

        _, _, test_ds, work_df = build_datasets_for_window(
            df=base_df,
            encoder_length=window,
            prediction_length=cfg.prediction_length,
        )

        batch_size = WINDOW_BATCH_SIZE.get(window, 64)
        test_loader = test_ds.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

        ensure_dir(window_dir)
        evaluate_window_and_write_outputs(
            window=window,
            prediction_length=cfg.prediction_length,
            work_df=work_df,
            test_loader=test_loader,
            best_ckpt=best_ckpt,
            window_dir=window_dir,
            metrics_path=metrics_path,
            metrics_rows=all_metrics,
            state_path=state_path,
        )

    if not all_metrics:
        logging.warning("No inference outputs produced.")
        return

    summary_df = pd.DataFrame(all_metrics)
    summary_df = summary_df[FINAL_METRIC_COLUMNS].sort_values(
        ["window"], kind="mergesort"
    )
    summary_path = cfg.train_artifact_root / "inference_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info("Saved inference summary -> %s", summary_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "TFT Unified Pipeline — hyperparameter tuning, training/testing, "
            "or inference on dataset/tft_ready.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── mode (optional bypass of interactive menu) ───────────────────────────
    parser.add_argument(
        "--mode",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help=(
            "Run mode: 0=tune→train, 1=train-only, 2=inference-only, 3=exit. "
            "Omit to show the interactive menu."
        ),
    )

    # ── shared args ──────────────────────────────────────────────────────────
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--windows", default="7,10,15,30")
    parser.add_argument(
        "--no-window-menu",
        action="store_true",
        help="Skip the interactive encoder-window selector and use --windows directly.",
    )
    parser.add_argument(
        "--prediction-length", type=int, default=DEFAULT_PREDICTION_LENGTH
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # ── training args ────────────────────────────────────────────────────────
    train_grp = parser.add_argument_group("training")
    train_grp.add_argument(
        "--train-artifact-root", default=str(DEFAULT_TRAIN_ARTIFACT_ROOT)
    )
    train_grp.add_argument("--train-max-epochs", type=int, default=50)
    train_grp.add_argument("--train-patience", type=int, default=8)
    train_grp.add_argument("--learning-rate", type=float, default=1e-3)
    train_grp.add_argument("--hidden-size", type=int, default=32)
    train_grp.add_argument("--hidden-continuous-size", type=int, default=16)
    train_grp.add_argument("--attention-head-size", type=int, default=4)
    train_grp.add_argument("--lstm-layers", type=int, default=2)
    train_grp.add_argument("--dropout", type=float, default=0.2)
    train_grp.add_argument("--gradient-clip-val", type=float, default=0.5)
    train_grp.add_argument("--force-retrain", action="store_true")
    train_grp.add_argument("--train-limit-val-batches", type=int, default=200)
    train_grp.add_argument("--train-accumulate-grad-batches", type=int, default=2)

    # ── tuning args ──────────────────────────────────────────────────────────
    tune_grp = parser.add_argument_group("tuning")
    tune_grp.add_argument(
        "--tune-artifact-root", default=str(DEFAULT_TUNE_ARTIFACT_ROOT)
    )
    tune_grp.add_argument("--tune-max-epochs", type=int, default=8)
    tune_grp.add_argument("--tune-patience", type=int, default=3)
    tune_grp.add_argument("--n-trials", type=int, default=30)
    tune_grp.add_argument("--max-total-trials", type=int, default=None)
    tune_grp.add_argument("--study-prefix", default="tft_tune_w")
    tune_grp.add_argument("--timeout-seconds", type=int, default=None)
    tune_grp.add_argument("--tune-accumulate-grad-batches", type=int, default=1)
    tune_grp.add_argument("--tune-limit-val-batches", type=int, default=50)
    tune_grp.add_argument("--eval-test-metrics", action="store_true")

    return parser


def make_unified_config(args: argparse.Namespace) -> UnifiedConfig:
    return UnifiedConfig(
        # shared
        data_path=Path(args.data_path),
        windows=parse_windows(args.windows),
        prediction_length=int(args.prediction_length),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        no_window_menu=bool(args.no_window_menu),
        # training
        train_artifact_root=Path(args.train_artifact_root),
        train_max_epochs=int(args.train_max_epochs),
        train_patience=int(args.train_patience),
        learning_rate=float(args.learning_rate),
        hidden_size=int(args.hidden_size),
        hidden_continuous_size=int(args.hidden_continuous_size),
        attention_head_size=int(args.attention_head_size),
        lstm_layers=int(args.lstm_layers),
        dropout=float(args.dropout),
        gradient_clip_val=float(args.gradient_clip_val),
        force_retrain=bool(args.force_retrain),
        train_limit_val_batches=int(args.train_limit_val_batches),
        train_accumulate_grad_batches=int(args.train_accumulate_grad_batches),
        # tuning
        tune_artifact_root=Path(args.tune_artifact_root),
        tune_max_epochs=int(args.tune_max_epochs),
        tune_patience=int(args.tune_patience),
        n_trials=int(args.n_trials),
        max_total_trials=(
            int(args.max_total_trials)
            if args.max_total_trials is not None
            else None
        ),
        study_prefix=str(args.study_prefix),
        timeout_seconds=(
            int(args.timeout_seconds)
            if args.timeout_seconds is not None
            else None
        ),
        tune_accumulate_grad_batches=int(args.tune_accumulate_grad_batches),
        tune_limit_val_batches=int(args.tune_limit_val_batches),
        eval_test_metrics=bool(args.eval_test_metrics),
    )


def main() -> None:
    configure_logging()
    configure_warnings()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "100000")

    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = make_unified_config(args)

    torch.set_float32_matmul_precision("medium")
    set_seed(cfg.seed)

    logging.info("Loading dataset from %s", cfg.data_path)
    base_df = load_and_prepare_dataframe(cfg.data_path)
    logging.info(
        "Dataset shape=%s symbols=%s date_min=%s date_max=%s",
        base_df.shape,
        base_df[SYMBOL_COL].nunique(),
        base_df[DATE_COL].min().date(),
        base_df[DATE_COL].max().date(),
    )

    # Mode selection: use --mode flag when provided, otherwise show menu.
    if args.mode is not None:
        mode = args.mode
    else:
        mode = show_main_menu()

    if mode == 0:
        run_tuning_mode(cfg=cfg, base_df=base_df)
    elif mode == 1:
        run_train_test_mode(cfg=cfg, base_df=base_df)
    elif mode == 2:
        run_inference_only_mode(cfg=cfg, base_df=base_df)
    elif mode == 3:
        logging.info("Exiting.")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
