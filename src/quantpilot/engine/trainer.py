"""LightGBM Ranker training pipeline with purged walk-forward validation."""

import json
import logging
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from quantpilot.config import LGBM_PARAMS, PREBUILT_DIR, CUSTOM_DIR

logger = logging.getLogger(__name__)


def create_labels(df: pd.DataFrame, forward_col: str = "fwd_ret_20d", n_groups: int = 5) -> pd.Series:
    """Create quantile-based ranking labels within each date cross-section.

    Returns Series with values 0..n_groups-1 (higher = better expected return).
    """
    def _quantile_rank(group):
        if len(group) < n_groups:
            return pd.Series(0, index=group.index)
        return pd.qcut(group[forward_col], q=n_groups, labels=False, duplicates="drop")

    return df.groupby("date")[forward_col].transform(_quantile_rank)


def purged_walk_forward(
    df: pd.DataFrame,
    factor_cols: list[str],
    n_splits: int = 5,
    embargo_days: int = 5,
    params: dict | None = None,
) -> tuple[lgb.LGBMRanker, dict]:
    """Train LGBMRanker with purged walk-forward validation.

    Args:
        df: DataFrame with [date, ticker, factor_cols..., fwd_ret_20d, label]
        factor_cols: List of factor column names
        n_splits: Number of time-series CV folds
        embargo_days: Gap between train/test to prevent leakage
        params: LightGBM parameters (overrides defaults)

    Returns:
        (trained_model, metrics_dict)
    """
    if params is None:
        params = LGBM_PARAMS.copy()

    dates = sorted(df["date"].unique())
    n_dates = len(dates)
    test_size = n_dates // (n_splits + 1)

    all_ic = []
    all_predictions = []

    for fold in range(n_splits):
        # Define train/test split with embargo
        test_end = n_dates - fold * test_size
        test_start = test_end - test_size
        train_end = test_start - embargo_days

        if train_end < test_size:
            continue

        train_dates = dates[:train_end]
        test_dates = dates[test_start:test_end]

        train_mask = df["date"].isin(train_dates)
        test_mask = df["date"].isin(test_dates)

        X_train = df.loc[train_mask, factor_cols].values
        X_test = df.loc[test_mask, factor_cols].values
        y_train = df.loc[train_mask, "label"].values.astype(int)
        y_test = df.loc[test_mask, "label"].values.astype(int)

        # Group sizes for LGBMRanker
        train_groups = df.loc[train_mask].groupby("date").size().values
        test_groups = df.loc[test_mask].groupby("date").size().values

        if len(train_groups) < 2 or len(test_groups) < 1:
            continue

        # Train
        model = lgb.LGBMRanker(**params)
        model.fit(
            X_train, y_train,
            group=train_groups,
            eval_set=[(X_test, y_test)],
            eval_group=[test_groups],
            callbacks=[lgb.log_evaluation(0)],
        )

        # Predict and compute IC
        preds = model.predict(X_test)
        actuals = df.loc[test_mask, "fwd_ret_20d"].values

        # IC per date
        test_df = df.loc[test_mask, ["date"]].copy()
        test_df["pred"] = preds
        test_df["actual"] = actuals

        ic_by_date = test_df.groupby("date").apply(
            lambda g: g["pred"].corr(g["actual"]), include_groups=False
        )
        all_ic.extend(ic_by_date.dropna().tolist())
        all_predictions.append(test_df)

    # Final model: train on all data
    X_all = df[factor_cols].values
    y_all = df["label"].values.astype(int)
    groups_all = df.groupby("date").size().values

    final_model = lgb.LGBMRanker(**params)
    final_model.fit(X_all, y_all, group=groups_all, callbacks=[lgb.log_evaluation(0)])

    # Metrics
    ic_array = np.array(all_ic)
    metrics = {
        "ic_mean": float(np.mean(ic_array)) if len(ic_array) > 0 else 0,
        "ic_std": float(np.std(ic_array)) if len(ic_array) > 0 else 0,
        "ir": float(np.mean(ic_array) / np.std(ic_array)) if len(ic_array) > 0 and np.std(ic_array) > 0 else 0,
        "n_folds": len(all_ic),
        "train_dates": len(dates),
    }

    return final_model, metrics


def save_model(
    model: lgb.LGBMRanker,
    sector_id: str,
    metrics: dict,
    factor_cols: list[str],
    is_custom: bool = False,
    user_id: str | None = None,
) -> Path:
    """Save trained model and metadata to disk.

    Returns path to model directory.
    """
    base = CUSTOM_DIR / user_id if is_custom and user_id else PREBUILT_DIR
    model_dir = base / sector_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "model.lgb"
    model.booster_.save_model(str(model_path))

    # Save metadata
    metadata = {
        "sector": sector_id,
        "train_date": datetime.now().isoformat(),
        "model_type": "custom" if is_custom else "pretrained",
        "factor_cols": factor_cols,
        "metrics": metrics,
        "lgbm_params": LGBM_PARAMS,
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )

    logger.info(f"Model saved: {model_dir}")
    return model_dir


def load_model(sector_id: str, is_custom: bool = False, user_id: str | None = None) -> tuple[lgb.LGBMRanker, dict]:
    """Load a trained model and its metadata.

    Returns (model, metadata_dict).
    """
    base = CUSTOM_DIR / user_id if is_custom and user_id else PREBUILT_DIR
    model_dir = base / sector_id

    model_path = model_dir / "model.lgb"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = lgb.LGBMRanker()
    model.booster_ = lgb.Booster(model_file=str(model_path))

    metadata = json.loads(metadata_path.read_text())
    return model, metadata


def model_exists(sector_id: str, is_custom: bool = False, user_id: str | None = None) -> bool:
    """Check if a model exists for the given sector."""
    base = CUSTOM_DIR / user_id if is_custom and user_id else PREBUILT_DIR
    return (base / sector_id / "model.lgb").exists()
