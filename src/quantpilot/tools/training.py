"""MCP Tool: train_sector_model — train a LightGBM ranker for a sector."""

import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quantpilot.data import resolve_sector, get_sector_stocks_cached, get_stock_daily
from quantpilot.engine import (
    extract_factors_batch, purged_walk_forward, save_model, register_model,
    explain_global,
)
from quantpilot.config import DEFAULT_LOOKBACK, DEFAULT_FORWARD, DEFAULT_N_SPLITS
from quantpilot.data.cache import DataCache

logger = logging.getLogger(__name__)
_cache = DataCache()


def train_sector_model(
    sector: str,
    start_date: str = "20240101",
    end_date: str | None = None,
    target: str = "ret_20d",
    n_splits: int = DEFAULT_N_SPLITS,
) -> str:
    """Train a LightGBM factor ranking model for a sector.

    Args:
        sector: Sector ID (cpo/pcb/ai/...) or Chinese name
        start_date: Training data start date (YYYYMMDD)
        end_date: Training data end date (default: today)
        target: Prediction target (ret_20d default)
        n_splits: Number of purged walk-forward folds

    Returns JSON string with training results.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    # Resolve sector
    sector_info = resolve_sector(sector)
    if not sector_info:
        return json.dumps({"error": f"Unknown sector: {sector}"})

    sector_id = sector_info["id"]

    # Get sector stocks
    stocks = get_sector_stocks_cached(sector_id)
    if not stocks:
        return json.dumps({"error": f"No stocks found for sector {sector_id}"})

    if len(stocks) < 10:
        return json.dumps({"error": f"Too few stocks ({len(stocks)}) for reliable training. Need at least 10."})

    # Fetch data for all stocks
    stock_data = {}
    fetch_start = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=DEFAULT_LOOKBACK + 30)).strftime("%Y%m%d")

    for s in stocks[:50]:  # Limit
        ticker = s["code"]
        df = get_stock_daily(ticker, fetch_start, end_date)
        if not df.empty and len(df) >= 60:
            stock_data[ticker] = df
            _cache.save_quotes(ticker, datetime.now().year, df)

    if len(stock_data) < 10:
        return json.dumps({"error": f"Only {len(stock_data)} stocks have sufficient data"})

    # Extract factors
    factors_df = extract_factors_batch(stock_data)
    if factors_df.empty:
        return json.dumps({"error": "Failed to extract factors"})

    # Compute forward returns
    def compute_fwd_return(group):
        group = group.sort_values("date")
        group["fwd_ret_20d"] = group["close"].pct_change(20).shift(-20)
        return group

    factors_df = factors_df.groupby("ticker", group_keys=False).apply(compute_fwd_return)
    factors_df = factors_df.dropna(subset=["fwd_ret_20d"])

    if factors_df.empty:
        return json.dumps({"error": "No valid forward return data"})

    # Filter factor columns
    factor_cols = [c for c in factors_df.columns
                   if c not in ["date", "ticker", "close", "fwd_ret_20d", "open", "high", "low",
                                "volume", "turnover", "turnover_rate"]]
    factor_cols = [c for c in factor_cols if factors_df[c].notna().sum() > len(factors_df) * 0.5]

    if not factor_cols:
        return json.dumps({"error": "No valid factor columns"})

    # Create labels
    from quantpilot.engine.trainer import create_labels
    factors_df["label"] = create_labels(factors_df, "fwd_ret_20d")

    # Train
    logger.info(f"Training model for {sector_id} with {len(factor_cols)} factors, {len(factors_df)} samples")
    model, metrics = purged_walk_forward(
        factors_df, factor_cols, n_splits=n_splits
    )

    # SHAP global explanation
    X_all = factors_df[factor_cols].fillna(0).values
    global_shap = explain_global(model, X_all, factor_cols)

    # Save model
    save_model(model, sector_id, metrics, factor_cols)
    register_model(sector_id, metrics, metadata={
        "n_stocks": len(stock_data),
        "n_factors": len(factor_cols),
        "date_range": f"{start_date}-{end_date}",
    })

    return json.dumps({
        "success": True,
        "sector": sector_id,
        "sector_name": sector_info["name"],
        "metrics": metrics,
        "n_stocks_used": len(stock_data),
        "n_factors": len(factor_cols),
        "top_factors": global_shap["top_factors"][:10],
        "train_date": datetime.now().isoformat(),
    }, ensure_ascii=False, indent=2)
