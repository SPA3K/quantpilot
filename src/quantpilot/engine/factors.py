"""Alpha158 factor extraction — simplified version for MVP.

Full Alpha158 has 158 factors. This MVP implements the most impactful ~40 factors
covering price-volume, momentum, valuation, and technical categories.
"""

import numpy as np
import pandas as pd

logger = __import__("logging").getLogger(__name__)


def extract_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Extract factors from daily OHLCV data.

    Input: DataFrame with columns [date, open, high, low, close, volume, turnover, turnover_rate]
    Output: DataFrame with same index + factor columns

    Must be called per-stock (single ticker), then concatenated across stocks.
    """
    if df.empty or len(df) < 60:
        return pd.DataFrame()

    df = df.sort_values("date").copy()
    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    v = df["volume"].astype(float)
    tr = df["turnover"].astype(float)
    ret = c.pct_change()

    factors = pd.DataFrame(index=df.index)

    # === Price-Volume Factors ===
    for w in [5, 10, 20, 60]:
        factors[f"ret_{w}d"] = c.pct_change(w)
        factors[f"std_{w}d"] = ret.rolling(w).std()
        factors[f"vol_ratio_{w}d"] = v / v.rolling(w).mean()
        factors[f"corr_cv_{w}d"] = c.rolling(w).corr(v)

    # === Momentum Factors ===
    for w in [5, 10, 20, 60]:
        factors[f"momentum_{w}d"] = c / c.shift(w) - 1
        factors[f"rank_ret_{w}d"] = ret.rolling(w).apply(lambda x: x.rank().iloc[-1] / len(x), raw=False)

    # === Reversal Factors ===
    factors["reversal_5d"] = -ret.rolling(5).sum()
    factors["reversal_20d"] = -ret.rolling(20).sum()

    # === Volume Factors ===
    factors["turnover_mean_5d"] = df["turnover_rate"].rolling(5).mean() if "turnover_rate" in df else 0
    factors["turnover_mean_20d"] = df["turnover_rate"].rolling(20).mean() if "turnover_rate" in df else 0
    factors["amplitude_5d"] = (h.rolling(5).max() - l.rolling(5).min()) / c
    factors["amplitude_20d"] = (h.rolling(20).max() - l.rolling(20).min()) / c

    # === Technical Factors ===
    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    factors["rsi_14"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    factors["macd"] = ema12 - ema26
    factors["macd_signal"] = factors["macd"].ewm(span=9).mean()
    factors["macd_hist"] = factors["macd"] - factors["macd_signal"]

    # Bollinger Band position
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    factors["bb_position"] = (c - ma20) / (2 * std20)

    # MA distances
    for w in [5, 10, 20, 60]:
        factors[f"ma_dist_{w}d"] = c / c.rolling(w).mean() - 1

    # === Price Pattern Factors ===
    factors["high_low_ratio"] = h / l
    factors["close_open_ratio"] = c / o
    factors["upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / c
    factors["lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / c

    # === Gap Factors ===
    factors["gap_up"] = (o / c.shift(1) - 1).clip(lower=0)
    factors["gap_down"] = (1 - o / c.shift(1)).clip(lower=0)

    return factors


def extract_factors_batch(stock_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Extract factors for multiple stocks and stack them.

    Args:
        stock_data: {ticker: daily_df} dict

    Returns:
        Multi-index DataFrame (date, ticker) with factor columns
    """
    all_factors = []
    for ticker, df in stock_data.items():
        if df.empty:
            continue
        factors = extract_factors(df)
        if factors.empty:
            continue
        factors["ticker"] = ticker
        factors["date"] = df["date"].values
        factors["close"] = df["close"].values  # keep for forward return calc
        all_factors.append(factors)

    if not all_factors:
        return pd.DataFrame()

    result = pd.concat(all_factors, ignore_index=True)
    return result


def get_factor_names() -> list[str]:
    """Return list of all factor column names."""
    # Generate a dummy single-row DataFrame to get column names
    dummy = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100),
        "open": np.random.randn(100).cumsum() + 100,
        "high": np.random.randn(100).cumsum() + 102,
        "low": np.random.randn(100).cumsum() + 98,
        "close": np.random.randn(100).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 100).astype(float),
        "turnover": np.random.randint(100000, 1000000, 100).astype(float),
        "turnover_rate": np.random.uniform(0.5, 5.0, 100),
    })
    factors = extract_factors(dummy)
    return list(factors.columns)
