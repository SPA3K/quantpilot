"""MCP Tool: sector_ranking — query stock rankings within a sector."""

import json
import logging

import numpy as np
import pandas as pd

from quantpilot.data import resolve_sector, get_sector_stocks_cached, get_stock_daily
from quantpilot.engine import (
    extract_factors_batch, load_model, model_exists,
    explain_global, explain_stock, format_shap_summary,
)
from quantpilot.config import DEFAULT_LOOKBACK
from quantpilot.data.cache import DataCache

logger = logging.getLogger(__name__)
_cache = DataCache()


def sector_ranking(sector: str, top_n: int = 10) -> str:
    """Query stock rankings within a sector using the trained model.

    Args:
        sector: Sector ID (cpo/pcb/ai/...) or Chinese name
        top_n: Number of top stocks to return

    Returns JSON string with ranking and SHAP explanations.
    """
    # Resolve sector
    sector_info = resolve_sector(sector)
    if not sector_info:
        return json.dumps({"error": f"Unknown sector: {sector}"})

    sector_id = sector_info["id"]

    # Check model exists
    if not model_exists(sector_id):
        return json.dumps({
            "error": f"No model for sector '{sector_id}'. Train one first with train_sector_model."
        })

    # Load model
    model, metadata = load_model(sector_id)
    factor_cols = metadata.get("factor_cols", [])

    # Get sector stocks
    stocks = get_sector_stocks_cached(sector_id)
    if not stocks:
        return json.dumps({"error": f"No stocks found for sector {sector_id}"})

    # Fetch recent data and extract factors
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=DEFAULT_LOOKBACK + 30)).strftime("%Y%m%d")

    stock_data = {}
    for s in stocks[:50]:  # Limit for speed
        ticker = s["code"]
        df = _cache.get_quotes(ticker, datetime.now().year)
        if df is None or df.empty:
            df = get_stock_daily(ticker, start_date, end_date)
            if not df.empty:
                _cache.save_quotes(ticker, datetime.now().year, df)
        if not df.empty:
            stock_data[ticker] = df

    if not stock_data:
        return json.dumps({"error": "Failed to fetch stock data"})

    # Extract factors
    factors_df = extract_factors_batch(stock_data)
    if factors_df.empty:
        return json.dumps({"error": "Failed to extract factors"})

    # Get latest date's factors
    latest_date = factors_df["date"].max()
    latest = factors_df[factors_df["date"] == latest_date].copy()

    # Filter to available factor columns
    available_cols = [c for c in factor_cols if c in latest.columns]
    if not available_cols:
        return json.dumps({"error": "No matching factor columns found"})

    X = latest[available_cols].fillna(0).values
    if len(X) == 0:
        return json.dumps({"error": "No data for latest date"})

    # Predict scores
    scores = model.predict(X)
    latest = latest.copy()
    latest["score"] = scores
    latest = latest.sort_values("score", ascending=False).head(top_n)

    # Build response
    ranking = []
    stock_name_map = {s["code"]: s["name"] for s in stocks}
    X_all = factors_df[factors_df["date"] == latest_date][available_cols].fillna(0).values

    for _, row in latest.iterrows():
        ticker = row["ticker"]
        # SHAP for this stock
        stock_idx = latest.index.get_loc(row.name)
        stock_factors = X[stock_idx:stock_idx+1]

        shap_contribs = explain_stock(model, stock_factors[0], available_cols)
        shap_top = [{"factor": c["factor"], "contrib": f"{c['contribution']:+.3f}"}
                     for c in shap_contribs if c["contribution"] > 0][:3]
        shap_bottom = [{"factor": c["factor"], "contrib": f"{c['contribution']:+.3f}"}
                        for c in shap_contribs if c["contribution"] < 0][:2]

        ranking.append({
            "rank": len(ranking) + 1,
            "ticker": ticker,
            "name": stock_name_map.get(ticker, ticker),
            "score": round(float(row["score"]), 4),
            "shap_top": shap_top,
            "shap_bottom": shap_bottom,
        })

    # Global SHAP
    global_shap = explain_global(model, X_all, available_cols)

    return json.dumps({
        "success": True,
        "sector": sector_id,
        "sector_name": sector_info["name"],
        "date": str(latest_date)[:10],
        "n_stocks": len(latest),
        "ranking": ranking,
        "model_metrics": metadata.get("metrics", {}),
        "global_top_factors": global_shap["top_factors"][:5],
    }, ensure_ascii=False, indent=2)
