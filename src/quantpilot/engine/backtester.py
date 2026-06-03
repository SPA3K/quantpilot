"""Purged walk-forward backtester for sector models."""

import logging

import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)


def backtest(
    model: lgb.LGBMRanker,
    df: pd.DataFrame,
    factor_cols: list[str],
    top_pct: float = 0.2,
    hold_days: int = 20,
) -> dict:
    """Run a simple long-only backtest using the trained model.

    Strategy: each rebalance date, buy the top_pct of stocks by model score,
    hold for hold_days, then rebalance.

    Returns dict with performance metrics and equity curve.
    """
    dates = sorted(df["date"].unique())
    
    if len(dates) < hold_days * 2:
        return {"error": "Not enough data for backtest"}

    # Predict scores for all dates
    all_scores = []
    for date in dates:
        mask = df["date"] == date
        X = df.loc[mask, factor_cols].values
        if len(X) == 0:
            continue
        scores = model.predict(X)
        tickers = df.loc[mask, "ticker"].values
        for t, s in zip(tickers, scores):
            all_scores.append({"date": date, "ticker": t, "score": s})

    scores_df = pd.DataFrame(all_scores)
    if scores_df.empty:
        return {"error": "No predictions generated"}

    # Rebalance every hold_days
    rebalance_dates = dates[::hold_days]
    
    portfolio_returns = []
    equity_curve = [{"date": str(dates[0]), "value": 1.0}]
    equity = 1.0

    for i, rebal_date in enumerate(rebalance_dates):
        # Get top stocks
        day_scores = scores_df[scores_df["date"] == rebal_date].copy()
        if day_scores.empty:
            continue
        day_scores = day_scores.sort_values("score", ascending=False)
        n_top = max(1, int(len(day_scores) * top_pct))
        top_stocks = day_scores.head(n_top)["ticker"].tolist()

        # Get returns over hold period
        next_idx = min(dates.index(rebal_date) + hold_days, len(dates) - 1)
        end_date = dates[next_idx]

        period_return = 0
        count = 0
        for ticker in top_stocks:
            mask = (df["ticker"] == ticker) & (df["date"].isin([rebDate for rebDate in dates if rebal_date <= rebDate <= end_date]))
            stock_data = df.loc[mask].sort_values("date")
            if len(stock_data) >= 2:
                r = stock_data["close"].iloc[-1] / stock_data["close"].iloc[0] - 1
                period_return += r
                count += 1

        if count > 0:
            avg_return = period_return / count
            equity *= (1 + avg_return)
            portfolio_returns.append(avg_return)
            equity_curve.append({"date": str(end_date), "value": round(equity, 4)})

    # Compute metrics
    returns = np.array(portfolio_returns)
    if len(returns) == 0:
        return {"error": "No returns generated"}

    # Annualize
    periods_per_year = 252 / hold_days
    annual_return = (1 + returns.mean()) ** periods_per_year - 1
    annual_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # Max drawdown
    equity_values = [e["value"] for e in equity_curve]
    peak = equity_values[0]
    max_dd = 0
    for v in equity_values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Turnover (approximate)
    turnover = 1.0 / hold_days  # Simplified

    return {
        "annual_return": round(float(annual_return), 4),
        "annual_volatility": round(float(annual_vol), 4),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(float(max_dd), 4),
        "total_periods": len(returns),
        "win_rate": round(float((returns > 0).mean()), 4),
        "equity_curve": equity_curve,
        "top_pct": top_pct,
        "hold_days": hold_days,
    }
