#!/usr/bin/env python3
"""
Build high-quality TFT visualizations from per-window test predictions.

Run manually in conda env `ml`, for example:
    conda activate ml
    python src/6_tft_visualize.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


DEFAULT_ARTIFACT_ROOT = Path("artifacts/tft")
DEFAULT_TICKER_PATH = Path("dataset/stock_dataset/nifty50_ticker.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/visualizations")
DEFAULT_WINDOWS = (7, 10, 15, 30)
PREDICTION_REQUIRED_COLS = {"Symbol", "decoder_start_date", "pred_return", "true_return"}
TICKER_REQUIRED_COLS = {"Date", "Symbol", "Adj Close"}


def parse_windows(text: str) -> List[int]:
    out: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("No windows provided. Use --windows like '7,10,15,30'.")
    return out


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def compute_mape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        raise ValueError("No valid samples for MAPE computation.")
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    denom = np.where(np.abs(y_true) < eps, np.nan, y_true)
    ape = np.abs((y_pred - y_true) / denom)
    ape = ape[np.isfinite(ape)]
    if ape.size == 0:
        raise ValueError("MAPE undefined: all denominators are near zero or invalid.")
    return float(np.mean(ape) * 100.0)


def compute_return_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        raise ValueError("No valid return samples for metric computation.")

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    err = y_pred - y_true

    mse = float(np.mean(np.square(err)))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))

    true_up = y_true > 0.0
    pred_up = y_pred > 0.0
    directional_accuracy = float(np.mean(true_up == pred_up))

    tp = int(np.sum(true_up & pred_up))
    fp = int(np.sum(~true_up & pred_up))
    fn = int(np.sum(true_up & ~pred_up))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_up = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "directional_accuracy": directional_accuracy,
        "f1_up": float(f1_up),
    }


def load_ticker_prices(ticker_path: Path) -> pd.DataFrame:
    if not ticker_path.exists():
        raise FileNotFoundError(f"Ticker file not found: {ticker_path}")

    ticker = pd.read_csv(ticker_path)
    ensure_required_columns(ticker, TICKER_REQUIRED_COLS, ticker_path)

    ticker = ticker[["Date", "Symbol", "Adj Close"]].copy()
    ticker["Date"] = (
        pd.to_datetime(ticker["Date"], errors="coerce", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    ticker["Adj Close"] = pd.to_numeric(ticker["Adj Close"], errors="coerce")
    ticker = ticker.dropna(subset=["Date", "Symbol", "Adj Close"]).copy()
    ticker = ticker.sort_values(["Symbol", "Date"], kind="mergesort").reset_index(drop=True)
    return ticker


def load_window_predictions(window: int, artifact_root: Path) -> pd.DataFrame:
    pred_path = artifact_root / f"window_{window}" / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    pred = pd.read_csv(pred_path)
    ensure_required_columns(pred, PREDICTION_REQUIRED_COLS, pred_path)

    pred["Symbol"] = pred["Symbol"].astype(str)
    pred["decoder_start_date"] = pd.to_datetime(pred["decoder_start_date"], errors="coerce").dt.normalize()
    pred["pred_return"] = pd.to_numeric(pred["pred_return"], errors="coerce")
    pred["true_return"] = pd.to_numeric(pred["true_return"], errors="coerce")

    pred = pred.dropna(subset=["Symbol", "decoder_start_date", "pred_return", "true_return"]).copy()
    pred = pred.sort_values(["Symbol", "decoder_start_date"], kind="mergesort").reset_index(drop=True)
    pred["window"] = int(window)
    return pred


def reconstruct_prices_inr(pred_df: pd.DataFrame, ticker_df: pd.DataFrame, window: int) -> pd.DataFrame:
    merged_parts: List[pd.DataFrame] = []

    for symbol, sym_pred in pred_df.groupby("Symbol", sort=True):
        sym_price = ticker_df.loc[ticker_df["Symbol"] == symbol, ["Date", "Adj Close"]].copy()
        if sym_price.empty:
            missing = sym_pred[["Symbol", "decoder_start_date"]].copy()
            missing["base_price_inr"] = np.nan
            missing["base_date"] = pd.NaT
            merged_parts.append(missing)
            continue

        sym_pred = sym_pred.sort_values("decoder_start_date", kind="mergesort").copy()
        sym_price = sym_price.sort_values("Date", kind="mergesort").copy()
        sym_price = sym_price.rename(columns={"Date": "base_date", "Adj Close": "base_price_inr"})

        merged = pd.merge_asof(
            sym_pred,
            sym_price,
            left_on="decoder_start_date",
            right_on="base_date",
            direction="backward",
            allow_exact_matches=False,
        )
        merged_parts.append(merged)

    out = pd.concat(merged_parts, axis=0, ignore_index=True)
    missing_base = out["base_price_inr"].isna()
    if bool(missing_base.any()):
        sample = (
            out.loc[missing_base, ["Symbol", "decoder_start_date"]]
            .drop_duplicates()
            .head(8)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"window={window}: missing previous-trading-day base Adj Close for "
            f"{int(missing_base.sum())} prediction rows. Sample: {sample}"
        )

    out["actual_price_inr"] = out["base_price_inr"] * (1.0 + out["true_return"])
    out["pred_price_inr"] = out["base_price_inr"] * (1.0 + out["pred_return"])
    out = out.sort_values(["Symbol", "decoder_start_date"], kind="mergesort").reset_index(drop=True)
    return out


def compute_window_metrics(window_df: pd.DataFrame, window: int) -> Dict[str, float]:
    y_true_return = window_df["true_return"].to_numpy(dtype=np.float64)
    y_pred_return = window_df["pred_return"].to_numpy(dtype=np.float64)
    y_true_price = window_df["actual_price_inr"].to_numpy(dtype=np.float64)
    y_pred_price = window_df["pred_price_inr"].to_numpy(dtype=np.float64)

    ret_metrics = compute_return_metrics(y_true_return, y_pred_return)
    mape = compute_mape_percent(y_true_price, y_pred_price)
    return {
        "window": int(window),
        "mape": mape,
        "mae": ret_metrics["mae"],
        "rmse": ret_metrics["rmse"],
        "mse": ret_metrics["mse"],
        "directional_accuracy": ret_metrics["directional_accuracy"],
        "f1_up": ret_metrics["f1_up"],
    }


def compute_stock_mape(window_df: pd.DataFrame, window: int) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for symbol, sym_df in window_df.groupby("Symbol", sort=True):
        y_true = sym_df["actual_price_inr"].to_numpy(dtype=np.float64)
        y_pred = sym_df["pred_price_inr"].to_numpy(dtype=np.float64)
        mape = compute_mape_percent(y_true, y_pred)
        rows.append(
            {
                "window": int(window),
                "Symbol": str(symbol),
                "mape": mape,
                "n_samples": int(len(sym_df)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["window", "mape", "Symbol"], kind="mergesort").reset_index(drop=True)


def validate_metric_ranges(metrics_df: pd.DataFrame) -> None:
    if (metrics_df["mape"] < 0).any():
        raise ValueError("Invalid metrics: MAPE must be >= 0.")
    if (metrics_df["mse"] < 0).any():
        raise ValueError("Invalid metrics: MSE must be >= 0.")
    if ((metrics_df["directional_accuracy"] < 0) | (metrics_df["directional_accuracy"] > 1)).any():
        raise ValueError("Invalid metrics: directional_accuracy must be in [0, 1].")
    if ((metrics_df["f1_up"] < 0) | (metrics_df["f1_up"] > 1)).any():
        raise ValueError("Invalid metrics: f1_up must be in [0, 1].")


def plot_best_window_actual_vs_predicted(best_df: pd.DataFrame, output_path: Path, dpi: int, best_window: int) -> None:
    symbols = sorted(best_df["Symbol"].unique().tolist())
    if not symbols:
        raise ValueError("No symbols found for best-window plotting.")

    ncols = 5
    nrows = math.ceil(len(symbols) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.6 * ncols, 2.3 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()

    for i, symbol in enumerate(symbols):
        ax = axes_arr[i]
        sym = (
            best_df.loc[best_df["Symbol"] == symbol]
            .sort_values("decoder_start_date", kind="mergesort")
            .copy()
        )
        ax.plot(
            sym["decoder_start_date"],
            sym["actual_price_inr"],
            color="#0D47A1",
            linewidth=0.95,
            linestyle="-",
            label="Actual Price",
        )
        ax.plot(
            sym["decoder_start_date"],
            sym["pred_price_inr"],
            color="#42A5F5",
            linewidth=0.95,
            linestyle="-",
            label="Predicted Price",
        )
        ax.set_title(symbol, fontsize=8, pad=2.5)
        ax.grid(alpha=0.22, linewidth=0.45)
        ax.tick_params(axis="both", labelsize=6, length=2)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")

    for j in range(len(symbols), len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle(
        f"Best Window (W{best_window}) - Actual vs Predicted Price (INR)",
        fontsize=14,
        weight="bold",
        y=0.996,
    )
    fig.text(0.5, 0.004, "Decoder Start Date", ha="center", fontsize=10)
    fig.text(0.004, 0.5, "Price (INR)", va="center", rotation=90, fontsize=10)

    legend_handles = [
        plt.Line2D([0], [0], color="#0D47A1", lw=1.0, linestyle="-", label="Actual Price"),
        plt.Line2D([0], [0], color="#42A5F5", lw=1.0, linestyle="-", label="Predicted Price"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.982))

    fig.tight_layout(rect=[0.015, 0.02, 1.0, 0.972])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_per_window_metrics(metrics_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    metrics_df = metrics_df.sort_values("window", kind="mergesort").reset_index(drop=True)
    x_labels = [f"W{int(w)}" for w in metrics_df["window"].tolist()]
    x = np.arange(len(x_labels))

    panels = [
        ("mape", "MAPE on Price (%)"),
        ("mae", "MAE (Return)"),
        ("rmse", "RMSE (Return)"),
        ("mse", "MSE (Return)"),
        ("directional_accuracy", "Directional Accuracy"),
        ("f1_up", "F1_up"),
    ]
    colors = ["#1565C0", "#1E88E5", "#42A5F5", "#64B5F6"]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes_arr = axes.ravel()

    for ax, (col, title) in zip(axes_arr, panels):
        vals = metrics_df[col].to_numpy(dtype=np.float64)
        bars = ax.bar(x, vals, color=[colors[i % len(colors)] for i in range(len(vals))], width=0.64)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

        y_max = float(np.nanmax(vals)) if len(vals) else 0.0
        if col in {"directional_accuracy", "f1_up"}:
            upper = max(1.0, y_max * 1.12 + 0.02)
            ax.set_ylim(0.0, upper)
        elif y_max > 0:
            ax.set_ylim(0.0, y_max * 1.16)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + (ax.get_ylim()[1] * 0.012),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                weight="bold",
            )

    fig.suptitle("Per-Window TFT Model - Recomputed Test Metrics", fontsize=16, weight="bold", y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_best_window_stock_mape(
    best_stock_mape_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
    best_window: int,
) -> None:
    part = best_stock_mape_df.sort_values(["mape", "Symbol"], kind="mergesort").reset_index(drop=True)
    if part.empty:
        raise ValueError(f"No stock-MAPE data found for best window W{best_window}.")

    colors = np.where(
        part["mape"].to_numpy(dtype=np.float64) < 2.0,
        "#2CA02C",
        np.where(part["mape"].to_numpy(dtype=np.float64) <= 5.0, "#F1C232", "#D62728"),
    )

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(part["Symbol"], part["mape"], color=colors, width=0.82, edgecolor="none")
    ax.set_title(
        f"Per-Stock Real MAPE Ranking (Ascending) - Best Window (W{best_window})",
        fontsize=24,
        weight="bold",
    )
    ax.set_xlabel("Symbol", fontsize=26)
    ax.set_ylabel("MAPE (%)", fontsize=26)
    ax.tick_params(axis="x", labelsize=11, rotation=75)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.25)
    ax.set_axisbelow(True)

    ax.axhline(2.0, color="#66BB6A", linestyle="--", linewidth=1.4, alpha=0.9)
    ax.axhline(5.0, color="#E57373", linestyle="--", linewidth=1.4, alpha=0.9)
    y_max = float(np.nanmax(part["mape"].to_numpy(dtype=np.float64)))
    ax.set_ylim(0.0, max(5.25, y_max * 1.15))

    legend_handles = [
        Patch(facecolor="#2CA02C", label="MAPE < 2%"),
        Patch(facecolor="#F1C232", label="2% <= MAPE <= 5%"),
        Patch(facecolor="#D62728", label="MAPE > 5%"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=16, frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TFT evaluation visualizations (non-notebook).")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--ticker-path", type=Path, default=DEFAULT_TICKER_PATH)
    parser.add_argument("--windows", type=str, default="7,10,15,30")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    args = build_arg_parser().parse_args()

    artifact_root = Path(args.artifact_root)
    ticker_path = Path(args.ticker_path)
    output_dir = Path(args.output_dir)
    windows = parse_windows(args.windows)
    dpi = int(args.dpi)

    output_dir.mkdir(parents=True, exist_ok=True)

    ticker_df = load_ticker_prices(ticker_path)
    all_windows_data: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, float]] = []
    stock_rows: List[pd.DataFrame] = []

    for window in windows:
        pred_df = load_window_predictions(window=window, artifact_root=artifact_root)
        enriched_df = reconstruct_prices_inr(pred_df=pred_df, ticker_df=ticker_df, window=window)
        all_windows_data.append(enriched_df)
        metric_rows.append(compute_window_metrics(enriched_df, window=window))
        stock_rows.append(compute_stock_mape(enriched_df, window=window))

    metrics_df = pd.DataFrame(metric_rows).sort_values("window", kind="mergesort").reset_index(drop=True)
    stock_mape_df = pd.concat(stock_rows, axis=0, ignore_index=True)
    stock_mape_df = stock_mape_df.sort_values(["window", "mape", "Symbol"], kind="mergesort").reset_index(drop=True)
    all_data_df = pd.concat(all_windows_data, axis=0, ignore_index=True)

    validate_metric_ranges(metrics_df)
    best_window = int(metrics_df.loc[metrics_df["mape"].idxmin(), "window"])
    best_df = (
        all_data_df.loc[all_data_df["window"] == best_window]
        .sort_values(["Symbol", "decoder_start_date"], kind="mergesort")
        .reset_index(drop=True)
    )

    best_plot = output_dir / "best_window_actual_vs_predicted.png"
    metrics_plot = output_dir / "per_window_metrics.png"
    stock_plot = output_dir / "best_window_stock_mape.png"
    best_stock_mape_df = stock_mape_df.loc[stock_mape_df["window"] == best_window].copy()

    plot_best_window_actual_vs_predicted(best_df=best_df, output_path=best_plot, dpi=dpi, best_window=best_window)
    plot_per_window_metrics(metrics_df=metrics_df, output_path=metrics_plot, dpi=dpi)
    plot_best_window_stock_mape(
        best_stock_mape_df=best_stock_mape_df,
        output_path=stock_plot,
        dpi=dpi,
        best_window=best_window,
    )

    # Remove legacy CSV artifacts from earlier versions of this script.
    legacy_csvs = [
        output_dir / "recomputed_window_metrics.csv",
        output_dir / "stock_mape_by_window.csv",
        output_dir / "best_window_price_series.csv",
    ]
    for csv_path in legacy_csvs:
        if csv_path.exists():
            csv_path.unlink()

    print(f"[INFO] Best window by recomputed MAPE: W{best_window}")
    print(f"[INFO] Saved: {best_plot}")
    print(f"[INFO] Saved: {metrics_plot}")
    print(f"[INFO] Saved: {stock_plot}")


if __name__ == "__main__":
    main()
