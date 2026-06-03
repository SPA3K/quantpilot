"""MCP Tools: stock_analysis, compare_sectors, list_sectors, get_model_info, backtest_model."""

import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quantpilot.data import (
    resolve_sector, get_sector_stocks_cached, get_stock_daily,
    get_stock_financial, list_all_sectors,
)
from quantpilot.engine import (
    extract_factors, load_model, model_exists,
    explain_stock, format_shap_summary, backtest,
)
from quantpilot.engine.registry import list_models, get_model_info as _get_model_info
from quantpilot.config import DEFAULT_LOOKBACK
from quantpilot.data.cache import DataCache

logger = logging.getLogger(__name__)
_cache = DataCache()


def stock_analysis(ticker: str) -> str:
    """Analyze a single stock using its sector's model.

    Auto-matches the stock to its sector, runs the model, and returns
    ranking position, score, SHAP decomposition, and fundamentals.
    """
    # Find which sector this stock belongs to
    from quantpilot.config import SECTORS
    matched_sector = None
    for sid in SECTORS:
        stocks = get_sector_stocks_cached(sid)
        stock_codes = [s["code"] for s in stocks]
        if ticker in stock_codes:
            matched_sector = sid
            break

    if not matched_sector:
        return json.dumps({"error": f"Stock {ticker} not found in any pre-defined sector"})

    if not model_exists(matched_sector):
        return json.dumps({"error": f"No model for sector '{matched_sector}'"})

    # Load model
    model, metadata = load_model(matched_sector)
    factor_cols = metadata.get("factor_cols", [])

    # Fetch stock data
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=DEFAULT_LOOKBACK + 60)).strftime("%Y%m%d")
    df = get_stock_daily(ticker, start_date, end_date)
    if df.empty:
        return json.dumps({"error": f"Failed to fetch data for {ticker}"})

    # Extract factors
    factors = extract_factors(df)
    if factors.empty:
        return json.dumps({"error": f"Insufficient data for factor extraction ({ticker})"})

    # Get latest
    latest = factors.iloc[-1:]
    available_cols = [c for c in factor_cols if c in latest.columns]
    X = latest[available_cols].fillna(0).values

    # Predict
    score = float(model.predict(X)[0])

    # SHAP
    shap_contribs = explain_stock(model, X[0], available_cols)

    # Fundamentals
    fin = get_stock_financial(ticker)

    # Find ranking within sector
    stocks = get_sector_stocks_cached(matched_sector)
    all_scores = []
    for s in stocks[:30]:
        sc = s["code"]
        if sc == ticker:
            all_scores.append({"ticker": sc, "name": s["name"], "score": score, "is_target": True})
            continue
        if model_exists(matched_sector):
            try:
                sd = get_stock_daily(sc, start_date, end_date)
                if not sd.empty:
                    f = extract_factors(sd)
                    if not f.empty:
                        x = f.iloc[-1:][available_cols].fillna(0).values
                        s_score = float(model.predict(x)[0])
                        all_scores.append({"ticker": sc, "name": s["name"], "score": s_score})
            except Exception:
                pass

    all_scores.sort(key=lambda x: x["score"], reverse=True)
    rank = next((i+1 for i, s in enumerate(all_scores) if s.get("is_target")), None)

    return json.dumps({
        "success": True,
        "ticker": ticker,
        "sector": matched_sector,
        "sector_name": SECTORS[matched_sector]["name"],
        "rank_in_sector": rank,
        "total_stocks": len(all_scores),
        "score": round(score, 4),
        "shap_decomposition": [
            {"factor": c["factor"], "value": c["value"], "contribution": f"{c['contribution']:+.3f}"}
            for c in shap_contribs[:8]
        ],
        "fundamentals": fin,
        "model_metrics": metadata.get("metrics", {}),
    }, ensure_ascii=False, indent=2)


def compare_sectors(sectors: list[str], metric: str = "sharpe") -> str:
    """Compare multiple sectors' factor weights and model performance."""
    from quantpilot.config import SECTORS
    results = []

    for sector_input in sectors:
        info = resolve_sector(sector_input)
        if not info:
            results.append({"sector": sector_input, "error": "not found"})
            continue

        sid = info["id"]
        if not model_exists(sid):
            results.append({"sector": sid, "name": info["name"], "error": "no model"})
            continue

        model, metadata = load_model(sid)
        metrics = metadata.get("metrics", {})
        factor_cols = metadata.get("factor_cols", [])

        results.append({
            "sector": sid,
            "name": info["name"],
            "metrics": metrics,
            "n_factors": len(factor_cols),
            "top_factors": metadata.get("top_factors", [])[:5],
        })

    # Sort by requested metric
    results.sort(key=lambda x: x.get("metrics", {}).get(metric, 0), reverse=True)

    return json.dumps({
        "success": True,
        "comparison_metric": metric,
        "sectors": results,
    }, ensure_ascii=False, indent=2)


def list_sectors_tool() -> str:
    """List all available sectors with model status."""
    sectors = list_all_sectors()
    for s in sectors:
        s["has_model"] = model_exists(s["id"])
        if s["has_model"]:
            _, meta = load_model(s["id"])
            s["model_metrics"] = meta.get("metrics", {})
    return json.dumps({"success": True, "sectors": sectors}, ensure_ascii=False, indent=2)


def get_model_info_tool(sector: str) -> str:
    """Get detailed model information for a sector."""
    info = resolve_sector(sector)
    if not info:
        return json.dumps({"error": f"Unknown sector: {sector}"})

    sid = info["id"]
    if not model_exists(sid):
        return json.dumps({"error": f"No model for sector '{sid}'"})

    model, metadata = load_model(sid)
    return json.dumps({
        "success": True,
        "sector": sid,
        "name": info["name"],
        "metadata": metadata,
    }, ensure_ascii=False, indent=2)


def backtest_model_tool(sector: str, start_date: str = "20240101", end_date: str | None = None, top_pct: float = 0.2) -> str:
    """Backtest a sector model over a date range."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    info = resolve_sector(sector)
    if not info:
        return json.dumps({"error": f"Unknown sector: {sector}"})

    sid = info["id"]
    if not model_exists(sid):
        return json.dumps({"error": f"No model for sector '{sid}'"})

    model, metadata = load_model(sid)
    factor_cols = metadata.get("factor_cols", [])

    # Fetch data
    stocks = get_sector_stocks_cached(sid)
    stock_data = {}
    for s in stocks[:30]:
        ticker = s["code"]
        df = get_stock_daily(ticker, start_date, end_date)
        if not df.empty and len(df) >= 60:
            stock_data[ticker] = df

    # Extract factors
    from quantpilot.engine import extract_factors_batch
    factors_df = extract_factors_batch(stock_data)
    if factors_df.empty:
        return json.dumps({"error": "Failed to extract factors for backtest"})

    # Compute forward returns
    def compute_fwd(group):
        group = group.sort_values("date")
        group["fwd_ret_20d"] = group["close"].pct_change(20).shift(-20)
        return group

    factors_df = factors_df.groupby("ticker", group_keys=False).apply(compute_fwd)

    # Run backtest
    result = backtest(model, factors_df, factor_cols, top_pct=top_pct)
    result["sector"] = sid
    result["sector_name"] = info["name"]

    return json.dumps(result, ensure_ascii=False, indent=2)
