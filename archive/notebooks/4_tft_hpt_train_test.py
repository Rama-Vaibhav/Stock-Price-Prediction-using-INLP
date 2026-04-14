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
import importlib.util
import logging
import os
import shutil
import sys
import traceback
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


THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT_PATH = THIS_DIR / "4_tft_train_test.py"
TUNE_SCRIPT_PATH = THIS_DIR / "5_tft_tune.py"

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
