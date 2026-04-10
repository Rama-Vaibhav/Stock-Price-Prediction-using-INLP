#!/usr/bin/env python3
"""
Unified TFT pipeline: Hyperparameter tuning + training/testing + inference.

This script combines the existing optimized flows from:
- src/4_tft_train_test.py
- src/5_tft_tune.py

Design intent:
- Preserve model/data/evaluation logic from existing scripts.
- Keep artifact locations unchanged:
  - tuning artifacts: artifacts/tft_tune/...
  - training/inference artifacts: artifacts/tft/...
- Provide one top-level interactive mode menu.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import traceback
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm


# --------------------------------------------------------------------------------------
# Dynamic module loading (filenames start with digits)
# --------------------------------------------------------------------------------------


def load_module_from_path(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Required for decorators like @dataclass that inspect sys.modules during class creation.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Embedded source payloads for fully self-contained runtime.
# Maintenance note: if train/tune scripts evolve, regenerate these payloads from
# the latest `src/4_tft_train_test.py` and `src/5_tft_tune.py`.
# After regeneration, run `python src/4_tft_hpt_static_audit.py` for static parity checks.
EMBEDDED_TRAIN_SRC_B64_ZLIB = (

    "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"
    "y/EbEMINlPyMkgTZYJJ4M3l/DIl98r8BTQyplw=="
)

EMBEDDED_TUNE_SRC_B64_ZLIB = (

    "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"
)


def _decode_embedded_source(payload: str) -> str:
    return zlib.decompress(base64.b64decode(payload.encode("ascii"))).decode("utf-8")


def _materialize_embedded_runtime_dir() -> Path:
    runtime_dir = Path(tempfile.mkdtemp(prefix="tft_unified_embedded_"))

    train_path = runtime_dir / "4_tft_train_test.py"
    tune_path = runtime_dir / "5_tft_tune.py"

    train_path.write_text(_decode_embedded_source(EMBEDDED_TRAIN_SRC_B64_ZLIB), encoding="utf-8")
    tune_path.write_text(_decode_embedded_source(EMBEDDED_TUNE_SRC_B64_ZLIB), encoding="utf-8")

    def _cleanup() -> None:
        shutil.rmtree(runtime_dir, ignore_errors=True)

    atexit.register(_cleanup)
    return runtime_dir


_RUNTIME_DIR = _materialize_embedded_runtime_dir()
TRAIN_SCRIPT_PATH = _RUNTIME_DIR / "4_tft_train_test.py"
TUNE_SCRIPT_PATH = _RUNTIME_DIR / "5_tft_tune.py"

TRAIN = load_module_from_path(TRAIN_SCRIPT_PATH, "tft_train_test_unified_base")
_TUNE = None


def get_tune_module():
    global _TUNE
    if _TUNE is None:
        _TUNE = load_module_from_path(TUNE_SCRIPT_PATH, "tft_tune_unified_base")
    return _TUNE


# Reused constants for parity with existing scripts.
DEFAULT_WINDOWS = TRAIN.DEFAULT_WINDOWS
DEFAULT_PREDICTION_LENGTH = TRAIN.DEFAULT_PREDICTION_LENGTH
DEFAULT_DATA_PATH = TRAIN.DEFAULT_DATA_PATH


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass
class UnifiedConfig:
    data_path: Path
    train_artifact_root: Path
    tune_artifact_root: Path
    windows: List[int]
    prediction_length: int
    seed: int
    allow_cpu_fallback: bool

    # Train knobs (from 4_tft_train_test.py)
    train_max_epochs: int
    train_patience: int
    train_num_workers: int
    train_learning_rate: float
    train_hidden_size: int
    train_hidden_continuous_size: int
    train_attention_head_size: int
    train_lstm_layers: int
    train_dropout: float
    train_gradient_clip_val: float
    train_limit_val_batches: int
    train_accumulate_grad_batches: int

    # Tune knobs (from 5_tft_tune.py)
    tune_n_trials: int
    tune_max_total_trials: Optional[int]
    tune_max_epochs: int
    tune_patience: int
    tune_num_workers: int
    tune_study_prefix: str
    tune_timeout_seconds: Optional[int]
    tune_accumulate_grad_batches: int
    tune_limit_val_batches: int
    tune_eval_test_metrics: bool


# --------------------------------------------------------------------------------------
# Menu / interaction
# --------------------------------------------------------------------------------------


def choose_mode_menu() -> str:
    """
    Mode menu:
      0 -> tune + train + test
      1 -> train + test
      2 -> inference only
      3 -> exit

    Empty input defaults to option 2.
    """
    lines = [
        "",
        "Select TFT operation mode:",
        "  [0] hyperparameter Tunning, Train and test on all the windows,",
        "  [1] Only trainig and testing",
        "  [2] only Inferenece",
        "  [3] Exit",
    ]
    print("\n".join(lines))

    while True:
        try:
            choice = input("Enter choice (0-3) [default: 2]: ").strip()
        except EOFError:
            logging.info("No interactive input detected. Using default mode: [2] inference.")
            return "2"

        if choice == "":
            choice = "2"

        if choice in {"0", "1", "2", "3"}:
            return choice

        print("Invalid choice. Please select one of: 0, 1, 2, 3.")


def confirm_mode(choice: str) -> bool:
    if choice == "3":
        return True

    descriptions = {
        "0": "Hyperparameter Tuning + Train + Test",
        "1": "Training + Testing",
        "2": "Inference only",
    }
    label = descriptions.get(choice, "selected mode")
    prompt = (
        f"Confirm {label}? This may update saved artifacts "
        f"(resume checkpoints for mode 0/1, outputs for mode 2) [y/N]: "
    )

    try:
        text = input(prompt).strip().lower()
    except EOFError:
        logging.warning("Confirmation input not available. Cancelling.")
        return False

    return text in {"y", "yes"}


# --------------------------------------------------------------------------------------
# Hyperparameter loading
# --------------------------------------------------------------------------------------


def _first_present(dct: Dict, keys: Sequence[str]):
    for key in keys:
        if key in dct and dct[key] is not None:
            return dct[key]
    return None


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def adaptive_lstm_layers_for_window(window: int) -> int:
    """
    Mirror the architecture rule used during Optuna trials in src/5_tft_tune.py.
    """
    return 1 if int(window) <= 10 else 2


def pick_loader_workers(requested_workers: int) -> Tuple[int, int]:
    """
    Match src/5_tft_tune.py behavior:
    - train workers: requested value (or auto=4 when negative)
    - eval workers: half of train workers
    """
    if requested_workers < 0:
        train_workers = 4
    else:
        train_workers = max(0, int(requested_workers))
    eval_workers = max(0, train_workers // 2)
    return train_workers, eval_workers


def load_tuned_hparams_for_window(tune_artifact_root: Path, window: int) -> Optional[Tuple[Dict[str, float], Path]]:
    """
    Reads artifacts/tft_tune/window_{w}/best_trial.json and supports both old/new schemas.

    Contract:
    - learning_rate, dropout, gradient_clip_val from best_params
    - hidden_size, hidden_continuous_size, attention_head_size:
      prefer best_user_attrs.effective_*, fallback to best_params keys
      with or without _v2 suffix.
    """
    best_path = tune_artifact_root / f"window_{window}" / "best_trial.json"
    if not best_path.exists():
        return None

    payload = TRAIN.read_json(best_path)
    best_params = payload.get("best_params", {}) if isinstance(payload, dict) else {}
    best_user_attrs = payload.get("best_user_attrs", {}) if isinstance(payload, dict) else {}

    if not isinstance(best_params, dict):
        best_params = {}
    if not isinstance(best_user_attrs, dict):
        best_user_attrs = {}

    tuned = {
        "learning_rate": _as_float(_first_present(best_params, ["learning_rate"])),
        "dropout": _as_float(_first_present(best_params, ["dropout"])),
        "gradient_clip_val": _as_float(_first_present(best_params, ["gradient_clip_val"])),
        "hidden_size": _as_int(
            _first_present(best_user_attrs, ["effective_hidden_size"])
            or _first_present(best_params, ["hidden_size", "hidden_size_v2"])
        ),
        "hidden_continuous_size": _as_int(
            _first_present(best_user_attrs, ["effective_hidden_continuous_size"])
            or _first_present(best_params, ["hidden_continuous_size", "hidden_continuous_size_v2"])
        ),
        "attention_head_size": _as_int(
            _first_present(best_user_attrs, ["effective_attention_head_size"])
            or _first_present(best_params, ["attention_head_size", "attention_head_size_v2"])
        ),
        "lstm_layers": _as_int(
            _first_present(best_user_attrs, ["effective_lstm_layers"])
            or _first_present(best_params, ["lstm_layers"])
            or adaptive_lstm_layers_for_window(window)
        ),
    }

    return tuned, best_path


def resolve_effective_hparams(
    cfg: UnifiedConfig,
    window: int,
    require_tuned: bool,
) -> Tuple[Dict[str, float], str, Optional[Path]]:
    defaults = {
        "learning_rate": float(cfg.train_learning_rate),
        "dropout": float(cfg.train_dropout),
        "gradient_clip_val": float(cfg.train_gradient_clip_val),
        "hidden_size": int(cfg.train_hidden_size),
        "hidden_continuous_size": int(cfg.train_hidden_continuous_size),
        "attention_head_size": int(cfg.train_attention_head_size),
        "lstm_layers": int(cfg.train_lstm_layers),
    }

    tuned_payload = load_tuned_hparams_for_window(cfg.tune_artifact_root, window)
    if tuned_payload is None:
        if require_tuned:
            raise FileNotFoundError(
                f"window={window}: expected tuned hyperparameters at "
                f"{cfg.tune_artifact_root / f'window_{window}' / 'best_trial.json'}"
            )
        return defaults, "default", None

    tuned, path = tuned_payload
    effective = defaults.copy()
    for key in effective:
        if tuned.get(key) is not None:
            effective[key] = tuned[key]

    return effective, "tuned", path


def backfill_tune_lstm_metadata(tune_artifact_root: Path, window: int) -> None:
    """
    Ensure best_trial.json carries lstm layer info so future train-only runs can read
    architecture directly from persisted tuning metadata.
    """
    best_path = tune_artifact_root / f"window_{window}" / "best_trial.json"
    if not best_path.exists():
        return

    payload = TRAIN.read_json(best_path)
    if not isinstance(payload, dict):
        return

    best_params = payload.get("best_params")
    if not isinstance(best_params, dict):
        best_params = {}
    best_user_attrs = payload.get("best_user_attrs")
    if not isinstance(best_user_attrs, dict):
        best_user_attrs = {}

    lstm_layers = adaptive_lstm_layers_for_window(window)
    changed = False
    if best_params.get("lstm_layers") != int(lstm_layers):
        best_params["lstm_layers"] = int(lstm_layers)
        changed = True
    if best_user_attrs.get("effective_lstm_layers") != int(lstm_layers):
        best_user_attrs["effective_lstm_layers"] = int(lstm_layers)
        changed = True

    if changed:
        payload["best_params"] = best_params
        payload["best_user_attrs"] = best_user_attrs
        TRAIN.write_json(best_path, payload)


# --------------------------------------------------------------------------------------
# Runtime setup
# --------------------------------------------------------------------------------------


def configure_runtime(seed: int) -> None:
    TRAIN.configure_logging()
    TRAIN.configure_warnings()

    # Optimizations used in tune script.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "100000")

    try:
        # Reduce file descriptor pressure from DataLoader shared-memory handles.
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    torch.set_float32_matmul_precision("medium")
    TRAIN.set_seed(seed)


# --------------------------------------------------------------------------------------
# Device checks
# --------------------------------------------------------------------------------------


def get_cuda_runtime_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }
    if info["cuda_available"]:
        try:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            info["cuda_device_name"] = "<unknown>"
    else:
        info["cuda_device_name"] = None
    return info


def ensure_cuda_for_training_or_raise(allow_cpu_fallback: bool, mode_label: str) -> bool:
    """
    Returns True when CUDA is available (GPU training), otherwise either:
    - raises RuntimeError (default), or
    - returns False if --allow-cpu-fallback is enabled.
    """
    info = get_cuda_runtime_info()
    if info["cuda_available"]:
        logging.info(
            "CUDA ready for %s | torch=%s cuda_build=%s device_count=%s device_0=%s visible_devices=%s",
            mode_label,
            info["torch_version"],
            info["torch_cuda_build"],
            info["cuda_device_count"],
            info["cuda_device_name"],
            info["cuda_visible_devices"],
        )
        return True

    msg = (
        f"GPU training was requested for {mode_label}, but torch.cuda.is_available() is False. "
        f"Detected: torch={info['torch_version']} cuda_build={info['torch_cuda_build']} "
        f"device_count={info['cuda_device_count']} CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']}. "
        "This usually means a CPU-only PyTorch build or CUDA/driver mismatch in the current env. "
        "Install CUDA-enabled PyTorch in your `ml` conda env. "
        "If you intentionally want CPU, rerun with --allow-cpu-fallback."
    )
    if allow_cpu_fallback:
        logging.warning(msg)
        logging.warning("Continuing with CPU because --allow-cpu-fallback is enabled.")
        return False
    raise RuntimeError(msg)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def remove_existing_train_window_artifacts(train_artifact_root: Path, windows: Sequence[int]) -> None:
    """Force overwrite behavior by clearing window directories before training."""
    for window in windows:
        window_dir = train_artifact_root / f"window_{window}"
        if window_dir.exists():
            shutil.rmtree(window_dir)


def build_test_loader_for_window(cfg: UnifiedConfig, base_df: pd.DataFrame, window: int):
    _, _, test_ds, work_df = TRAIN.build_datasets_for_window(
        df=base_df,
        encoder_length=window,
        prediction_length=cfg.prediction_length,
    )
    batch_size = TRAIN.WINDOW_BATCH_SIZE.get(window, 64)
    test_loader = test_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=cfg.train_num_workers,
        persistent_workers=cfg.train_num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, work_df


def resolve_checkpoint_for_inference(window_dir: Path) -> Optional[str]:
    state_path = window_dir / "state.json"
    checkpoints_dir = window_dir / "checkpoints"

    state = TRAIN.read_json(state_path)
    for key in ("best_ckpt", "last_ckpt"):
        ckpt = state.get(key)
        if ckpt and Path(ckpt).exists():
            return str(Path(ckpt))

    latest = TRAIN.find_latest_checkpoint(checkpoints_dir)
    if latest is not None:
        return str(latest)
    return None


# --------------------------------------------------------------------------------------
# Mode implementations
# --------------------------------------------------------------------------------------


def run_tuning_mode(cfg: UnifiedConfig, base_df: pd.DataFrame) -> None:
    tune = get_tune_module()
    TRAIN.ensure_dir(cfg.tune_artifact_root)

    tune_cfg = tune.TuneConfig(
        data_path=cfg.data_path,
        artifact_root=cfg.tune_artifact_root,
        windows=list(cfg.windows),
        prediction_length=int(cfg.prediction_length),
        n_trials=int(cfg.tune_n_trials),
        max_total_trials=int(cfg.tune_max_total_trials) if cfg.tune_max_total_trials is not None else None,
        max_epochs=int(cfg.tune_max_epochs),
        patience=int(cfg.tune_patience),
        num_workers=int(cfg.tune_num_workers),
        seed=int(cfg.seed),
        study_prefix=str(cfg.tune_study_prefix),
        timeout_seconds=int(cfg.tune_timeout_seconds) if cfg.tune_timeout_seconds is not None else None,
        accumulate_grad_batches=int(cfg.tune_accumulate_grad_batches),
        limit_val_batches=int(cfg.tune_limit_val_batches),
        eval_test_metrics=bool(cfg.tune_eval_test_metrics),
    )

    summary_rows: List[Dict] = []
    for window in cfg.windows:
        try:
            row = tune.tune_window(cfg=tune_cfg, base_df=base_df, window=window)
            backfill_tune_lstm_metadata(cfg.tune_artifact_root, window)
            summary_rows.append(row)
        except Exception as exc:
            logging.error("window=%s tuning failed: %s", window, exc)
            logging.debug("Traceback:\n%s", traceback.format_exc())
            summary_rows.append(
                {
                    "window": window,
                    "study_name": f"{cfg.tune_study_prefix}{window}",
                    "study_db": str((cfg.tune_artifact_root / f"window_{window}" / "study.db")),
                    "n_trials_total": None,
                    "best_val_loss": None,
                    "best_trial_number": None,
                    "error": str(exc),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = cfg.tune_artifact_root / "tuning_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info("Saved tuning summary -> %s", summary_path)


def run_window_training_optimized(
    cfg: TRAIN.RunConfig,
    base_df: pd.DataFrame,
    window: int,
    metrics_rows: List[Dict],
    use_gpu: bool,
) -> None:
    """
    Same training/eval flow as TRAIN.run_window_training with train/eval dataloader
    worker tuning borrowed from src/5_tft_tune.py.
    """
    log = logging.getLogger(__name__)

    window_dir = cfg.artifact_root / f"window_{window}"
    checkpoints_dir = window_dir / "checkpoints"
    logs_dir = window_dir / "logs"
    state_path = window_dir / "state.json"
    metrics_path = window_dir / "metrics.csv"

    TRAIN.ensure_dir(window_dir)
    TRAIN.ensure_dir(checkpoints_dir)
    TRAIN.ensure_dir(logs_dir)

    state = TRAIN.read_json(state_path)
    metrics_only_recompute = False
    if state.get("status") == "completed" and metrics_path.exists() and not cfg.force_retrain:
        existing = pd.read_csv(metrics_path)
        exact_schema = list(existing.columns) == TRAIN.FINAL_METRIC_COLUMNS
        expected_rows = len(existing) == 1
        no_missing_values = exact_schema and expected_rows and not existing[TRAIN.FINAL_METRIC_COLUMNS].isna().any().any()
        if no_missing_values:
            log.info("window=%s already completed with up-to-date metrics, skipping.", window)
            metrics_rows.extend(existing[TRAIN.FINAL_METRIC_COLUMNS].to_dict(orient="records"))
            return
        else:
            log.warning(
                "window=%s existing metrics file is outdated (schema_ok=%s rows=%s has_nan=%s). Recomputing this window.",
                window,
                exact_schema,
                len(existing),
                bool(existing[TRAIN.FINAL_METRIC_COLUMNS].isna().any().any()) if exact_schema and expected_rows else True,
            )
            metrics_only_recompute = True
    elif state.get("status") == "completed" and not cfg.force_retrain:
        log.warning("window=%s marked completed but metrics file missing. Recomputing this window.", window)
        metrics_only_recompute = True

    training_ds, val_ds, test_ds, work_df = TRAIN.build_datasets_for_window(
        df=base_df,
        encoder_length=window,
        prediction_length=cfg.prediction_length,
    )
    log.info(
        "window=%s eligible_symbols=%s train_rows=%s val_rows=%s test_rows=%s",
        window,
        work_df[TRAIN.SYMBOL_COL].nunique(),
        int((work_df[TRAIN.DATE_COL] <= pd.Timestamp(TRAIN.TRAIN_END)).sum()),
        int(
            (
                (work_df[TRAIN.DATE_COL] >= pd.Timestamp(TRAIN.VAL_START))
                & (work_df[TRAIN.DATE_COL] <= pd.Timestamp(TRAIN.VAL_END))
            ).sum()
        ),
        int((work_df[TRAIN.DATE_COL] >= pd.Timestamp(TRAIN.TEST_START)).sum()),
    )

    batch_size = TRAIN.WINDOW_BATCH_SIZE.get(window, 64)
    train_workers, eval_workers = pick_loader_workers(int(cfg.num_workers))
    train_loader_kwargs = {
        "train": True,
        "batch_size": batch_size,
        "num_workers": train_workers,
        "persistent_workers": train_workers > 0,
        "pin_memory": use_gpu,
    }
    eval_loader_kwargs = {
        "train": False,
        "batch_size": batch_size,
        "num_workers": eval_workers,
        "persistent_workers": eval_workers > 0,
        "pin_memory": use_gpu,
    }
    if train_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2
        train_loader_kwargs["multiprocessing_context"] = "fork"
    if eval_workers > 0:
        eval_loader_kwargs["prefetch_factor"] = 2
        eval_loader_kwargs["multiprocessing_context"] = "fork"

    test_loader = test_ds.to_dataloader(**eval_loader_kwargs)

    if metrics_only_recompute and not cfg.force_retrain:
        eval_ckpt = None
        for ckpt_key in ("best_ckpt", "last_ckpt"):
            ck = state.get(ckpt_key)
            if ck and Path(ck).exists():
                eval_ckpt = str(Path(ck))
                break
        if eval_ckpt is None:
            latest = TRAIN.find_latest_checkpoint(checkpoints_dir)
            if latest is not None:
                eval_ckpt = str(latest)

        if eval_ckpt is not None:
            log.info("window=%s metrics-only recompute using checkpoint: %s", window, eval_ckpt)
            TRAIN.evaluate_window_and_write_outputs(
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
            "window=%s requested metrics-only recompute but no checkpoint found. Falling back to training.",
            window,
        )

    train_loader = training_ds.to_dataloader(**train_loader_kwargs)
    val_loader = val_ds.to_dataloader(**eval_loader_kwargs)

    early_stop = TRAIN.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=cfg.patience,
        min_delta=1e-4,
        verbose=False,
    )
    best_ckpt_cb = TRAIN.ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="best-epoch{epoch:03d}-valloss{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        auto_insert_metric_name=False,
    )
    step_ckpt_cb = TRAIN.ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="step-{step:09d}",
        monitor=None,
        save_top_k=-1,
        every_n_train_steps=1000,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )
    state_cb = TRAIN.WindowStateCallback(state_path=state_path, checkpoint_dir=checkpoints_dir, window=window)
    progress_cb = TRAIN.TQDMProgressBar(refresh_rate=10)

    csv_logger = TRAIN.CSVLogger(save_dir=str(logs_dir), name="lightning")

    trainer = TRAIN.pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision="16-mixed" if use_gpu else "32-true",
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

    model = TRAIN.TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        attention_head_size=cfg.attention_head_size,
        hidden_continuous_size=cfg.hidden_continuous_size,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
        loss=TRAIN.QuantileLoss(),
        output_size=7,
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
            latest = TRAIN.find_latest_checkpoint(checkpoints_dir)
            if latest is not None:
                resume_ckpt = str(latest)

    TRAIN.write_json(
        state_path,
        {
            "window": window,
            "status": "training",
            "started_at": TRAIN.now_utc_iso(),
            "last_ckpt": resume_ckpt or "",
            "max_epochs": cfg.max_epochs,
        },
    )

    try:
        logging.info(
            "window=%s training start (max_epochs=%s, batch=%s, train_workers=%s, eval_workers=%s, "
            "accumulate_grad_batches=%s, limit_val_batches=%s)",
            window,
            cfg.max_epochs,
            batch_size,
            train_workers,
            eval_workers,
            cfg.accumulate_grad_batches,
            cfg.limit_val_batches,
        )
        if resume_ckpt:
            logging.info("window=%s resuming from checkpoint: %s", window, resume_ckpt)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt)
    except KeyboardInterrupt:
        interrupted_ckpt = checkpoints_dir / "interrupt.ckpt"
        try:
            trainer.save_checkpoint(str(interrupted_ckpt))
        except Exception:
            interrupted_ckpt = None
        latest = interrupted_ckpt if interrupted_ckpt is not None and interrupted_ckpt.exists() else TRAIN.find_latest_checkpoint(checkpoints_dir)
        TRAIN.write_json(
            state_path,
            {
                "window": window,
                "status": "interrupted",
                "updated_at": TRAIN.now_utc_iso(),
                "last_ckpt": str(latest) if latest else "",
                "last_epoch_completed": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                "message": "KeyboardInterrupt",
            },
        )
        raise
    except Exception as exc:
        failed_ckpt = checkpoints_dir / "failed.ckpt"
        try:
            trainer.save_checkpoint(str(failed_ckpt))
        except Exception:
            failed_ckpt = None
        latest = failed_ckpt if failed_ckpt is not None and failed_ckpt.exists() else TRAIN.find_latest_checkpoint(checkpoints_dir)
        TRAIN.write_json(
            state_path,
            {
                "window": window,
                "status": "failed",
                "updated_at": TRAIN.now_utc_iso(),
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
        latest = TRAIN.find_latest_checkpoint(checkpoints_dir)
        if latest is None:
            raise RuntimeError(f"window={window}: no checkpoint found after training.")
        best_ckpt = str(latest)

    TRAIN.write_json(
        state_path,
        {
            "window": window,
            "status": "trained",
            "updated_at": TRAIN.now_utc_iso(),
            "best_ckpt": best_ckpt,
            "last_epoch_completed": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "best_score": float(best_ckpt_cb.best_model_score.item())
            if best_ckpt_cb.best_model_score is not None
            else None,
        },
    )

    TRAIN.evaluate_window_and_write_outputs(
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


def run_train_test_mode(
    cfg: UnifiedConfig,
    base_df: pd.DataFrame,
    require_tuned: bool,
    use_gpu: bool,
) -> None:
    TRAIN.ensure_dir(cfg.train_artifact_root)

    all_metrics: List[Dict] = []
    for window in tqdm(cfg.windows, desc="Training windows", unit="window"):
        effective_hp, source, source_path = resolve_effective_hparams(
            cfg=cfg,
            window=window,
            require_tuned=require_tuned,
        )

        logging.info(
            "window=%s hyperparameters source=%s path=%s lr=%.8f hidden_size=%s hidden_continuous_size=%s "
            "attention_head_size=%s lstm_layers=%s dropout=%.6f gradient_clip_val=%.6f",
            window,
            source,
            str(source_path) if source_path else "-",
            float(effective_hp["learning_rate"]),
            int(effective_hp["hidden_size"]),
            int(effective_hp["hidden_continuous_size"]),
            int(effective_hp["attention_head_size"]),
            int(effective_hp["lstm_layers"]),
            float(effective_hp["dropout"]),
            float(effective_hp["gradient_clip_val"]),
        )

        run_cfg = TRAIN.RunConfig(
            data_path=cfg.data_path,
            artifact_root=cfg.train_artifact_root,
            windows=[window],
            prediction_length=int(cfg.prediction_length),
            max_epochs=int(cfg.train_max_epochs),
            patience=int(cfg.train_patience),
            num_workers=int(cfg.train_num_workers),
            seed=int(cfg.seed),
            learning_rate=float(effective_hp["learning_rate"]),
            hidden_size=int(effective_hp["hidden_size"]),
            hidden_continuous_size=int(effective_hp["hidden_continuous_size"]),
            attention_head_size=int(effective_hp["attention_head_size"]),
            lstm_layers=int(effective_hp["lstm_layers"]),
            dropout=float(effective_hp["dropout"]),
            gradient_clip_val=float(effective_hp["gradient_clip_val"]),
            force_retrain=False,
            limit_val_batches=int(cfg.train_limit_val_batches),
            accumulate_grad_batches=int(cfg.train_accumulate_grad_batches),
        )

        run_window_training_optimized(
            cfg=run_cfg,
            base_df=base_df,
            window=window,
            metrics_rows=all_metrics,
            use_gpu=use_gpu,
        )

    if not all_metrics:
        logging.warning("No metrics produced. Check state files under %s", cfg.train_artifact_root)
        return

    summary_df = pd.DataFrame(all_metrics)
    summary_df = summary_df[TRAIN.FINAL_METRIC_COLUMNS].sort_values(["window"], kind="mergesort")
    summary_path = cfg.train_artifact_root / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info("Saved global metrics summary -> %s", summary_path)


def run_inference_only_mode(cfg: UnifiedConfig, base_df: pd.DataFrame) -> None:
    TRAIN.ensure_dir(cfg.train_artifact_root)

    all_metrics: List[Dict] = []
    for window in tqdm(cfg.windows, desc="Inference windows", unit="window"):
        window_dir = cfg.train_artifact_root / f"window_{window}"
        checkpoints_dir = window_dir / "checkpoints"
        state_path = window_dir / "state.json"
        metrics_path = window_dir / "metrics.csv"

        TRAIN.ensure_dir(window_dir)
        TRAIN.ensure_dir(checkpoints_dir)

        best_ckpt = resolve_checkpoint_for_inference(window_dir)
        if best_ckpt is None:
            logging.warning(
                "window=%s inference skipped: no checkpoint found in %s",
                window,
                checkpoints_dir,
            )
            continue

        logging.info("window=%s inference using checkpoint: %s", window, best_ckpt)
        test_loader, work_df = build_test_loader_for_window(cfg=cfg, base_df=base_df, window=window)

        TRAIN.evaluate_window_and_write_outputs(
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
        logging.warning("Inference mode finished with no windows processed. No checkpoints were available.")
        return

    summary_df = pd.DataFrame(all_metrics)
    summary_df = summary_df[TRAIN.FINAL_METRIC_COLUMNS].sort_values(["window"], kind="mergesort")
    summary_path = cfg.train_artifact_root / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info("Saved inference metrics summary -> %s", summary_path)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified TFT: tune + train/test + inference",
    )
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--train-artifact-root", default="artifacts/tft")
    parser.add_argument("--tune-artifact-root", default="artifacts/tft_tune")
    parser.add_argument("--windows", default="7,10,15,30")
    parser.add_argument("--prediction-length", type=int, default=DEFAULT_PREDICTION_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow CPU training/tuning when CUDA is unavailable. By default, modes 0/1 require GPU.",
    )

    # Training controls (defaults mirror src/4_tft_train_test.py)
    parser.add_argument("--train-max-epochs", type=int, default=50)
    parser.add_argument("--train-patience", type=int, default=8)
    parser.add_argument("--train-num-workers", type=int, default=4)
    parser.add_argument("--train-learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-hidden-size", type=int, default=32)
    parser.add_argument("--train-hidden-continuous-size", type=int, default=16)
    parser.add_argument("--train-attention-head-size", type=int, default=4)
    parser.add_argument("--train-lstm-layers", type=int, default=2)
    parser.add_argument("--train-dropout", type=float, default=0.2)
    parser.add_argument("--train-gradient-clip-val", type=float, default=0.5)
    parser.add_argument("--train-limit-val-batches", type=int, default=200)
    parser.add_argument("--train-accumulate-grad-batches", type=int, default=2)

    # Tuning controls (defaults mirror src/5_tft_tune.py)
    parser.add_argument("--tune-n-trials", type=int, default=30)
    parser.add_argument("--tune-max-total-trials", type=int, default=None)
    parser.add_argument("--tune-max-epochs", type=int, default=8)
    parser.add_argument("--tune-patience", type=int, default=3)
    parser.add_argument("--tune-num-workers", type=int, default=4)
    parser.add_argument("--tune-study-prefix", default="tft_tune_w")
    parser.add_argument("--tune-timeout-seconds", type=int, default=None)
    parser.add_argument("--tune-accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--tune-limit-val-batches", type=int, default=50)
    parser.add_argument("--tune-eval-test-metrics", action="store_true")

    return parser


def make_config(args: argparse.Namespace) -> UnifiedConfig:
    windows = TRAIN.parse_windows(args.windows)
    return UnifiedConfig(
        data_path=Path(args.data_path),
        train_artifact_root=Path(args.train_artifact_root),
        tune_artifact_root=Path(args.tune_artifact_root),
        windows=windows,
        prediction_length=int(args.prediction_length),
        seed=int(args.seed),
        allow_cpu_fallback=bool(args.allow_cpu_fallback),
        train_max_epochs=int(args.train_max_epochs),
        train_patience=int(args.train_patience),
        train_num_workers=int(args.train_num_workers),
        train_learning_rate=float(args.train_learning_rate),
        train_hidden_size=int(args.train_hidden_size),
        train_hidden_continuous_size=int(args.train_hidden_continuous_size),
        train_attention_head_size=int(args.train_attention_head_size),
        train_lstm_layers=int(args.train_lstm_layers),
        train_dropout=float(args.train_dropout),
        train_gradient_clip_val=float(args.train_gradient_clip_val),
        train_limit_val_batches=int(args.train_limit_val_batches),
        train_accumulate_grad_batches=int(args.train_accumulate_grad_batches),
        tune_n_trials=int(args.tune_n_trials),
        tune_max_total_trials=int(args.tune_max_total_trials) if args.tune_max_total_trials is not None else None,
        tune_max_epochs=int(args.tune_max_epochs),
        tune_patience=int(args.tune_patience),
        tune_num_workers=int(args.tune_num_workers),
        tune_study_prefix=str(args.tune_study_prefix),
        tune_timeout_seconds=int(args.tune_timeout_seconds) if args.tune_timeout_seconds is not None else None,
        tune_accumulate_grad_batches=int(args.tune_accumulate_grad_batches),
        tune_limit_val_batches=int(args.tune_limit_val_batches),
        tune_eval_test_metrics=bool(args.tune_eval_test_metrics),
    )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = make_config(args)

    configure_runtime(seed=cfg.seed)

    mode = choose_mode_menu()
    if mode == "3":
        logging.info("Exit selected by user.")
        return

    if not confirm_mode(mode):
        logging.info("Operation cancelled by user.")
        return

    TRAIN.ensure_dir(cfg.train_artifact_root)
    TRAIN.ensure_dir(cfg.tune_artifact_root)

    logging.info("Loading dataset from %s", cfg.data_path)
    base_df = TRAIN.load_and_prepare_dataframe(cfg.data_path)
    logging.info(
        "Dataset shape=%s symbols=%s date_min=%s date_max=%s",
        base_df.shape,
        base_df[TRAIN.SYMBOL_COL].nunique(),
        base_df[TRAIN.DATE_COL].min().date(),
        base_df[TRAIN.DATE_COL].max().date(),
    )

    if mode == "0":
        use_gpu = ensure_cuda_for_training_or_raise(
            allow_cpu_fallback=cfg.allow_cpu_fallback,
            mode_label="mode [0] Hyperparameter tuning + training/testing",
        )
        logging.info("Mode [0]: Hyperparameter tuning + training/testing")
        run_tuning_mode(cfg=cfg, base_df=base_df)
        run_train_test_mode(cfg=cfg, base_df=base_df, require_tuned=True, use_gpu=use_gpu)
        return

    if mode == "1":
        use_gpu = ensure_cuda_for_training_or_raise(
            allow_cpu_fallback=cfg.allow_cpu_fallback,
            mode_label="mode [1] Training/testing",
        )
        logging.info("Mode [1]: Training/testing (load tuned hyperparameters when available)")
        run_train_test_mode(cfg=cfg, base_df=base_df, require_tuned=False, use_gpu=use_gpu)
        return

    if mode == "2":
        logging.info("Mode [2]: Inference only")
        run_inference_only_mode(cfg=cfg, base_df=base_df)
        return

    raise RuntimeError(f"Unsupported mode selection: {mode}")


if __name__ == "__main__":
    main()
