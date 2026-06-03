"""Generate synthetic stock data for pipeline testing when akshare is unavailable."""

import numpy as np
import pandas as pd


def generate_synthetic_sector(
    sector_name: str,
    n_stocks: int = 25,
    n_days: int = 500,
    start_date: str = "2024-01-01",
) -> dict[str, pd.DataFrame]:
    """Generate realistic synthetic daily data for a sector.

    Returns {ticker: DataFrame} with realistic price/volume patterns.
    """
    np.random.seed(hash(sector_name) % 2**31)
    dates = pd.bdate_range(start_date, periods=n_days)
    
    tickers = {}
    for i in range(n_stocks):
        ticker = f"{100000 + i}"[-6:]  # 6-digit code
        
        # Base price with drift
        base_price = np.random.uniform(10, 200)
        drift = np.random.normal(0.0003, 0.001)  # slight upward bias
        vol = np.random.uniform(0.015, 0.035)  # daily volatility
        
        # Generate returns with momentum and mean-reversion
        returns = np.random.normal(drift, vol, n_days)
        
        # Add momentum (autocorrelation)
        for j in range(1, n_days):
            returns[j] += 0.05 * returns[j-1]
        
        # Add mean-reversion
        log_prices = np.cumsum(returns) + np.log(base_price)
        for j in range(20, n_days):
            ma = np.mean(log_prices[j-20:j])
            returns[j] -= 0.02 * (log_prices[j] - ma)
        
        # Generate price series
        close = np.exp(np.cumsum(returns) + np.log(base_price))
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        open_ = close * (1 + np.random.normal(0, 0.005, n_days))
        
        # Volume (correlated with abs return)
        base_vol = np.random.uniform(5e6, 5e7)
        volume = base_vol * (1 + 3 * np.abs(returns)) * np.random.lognormal(0, 0.3, n_days)
        
        # Turnover
        turnover = volume * close
        turnover_rate = volume / (base_vol * 10) * 100
        
        df = pd.DataFrame({
            "date": dates[:n_days],
            "ticker": ticker,
            "open": np.round(open_, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "volume": np.round(volume, 0).astype(int),
            "turnover": np.round(turnover, 0).astype(float),
            "turnover_rate": np.round(turnover_rate, 2),
        })
        
        tickers[ticker] = df
    
    return tickers


if __name__ == "__main__":
    # Quick test
    data = generate_synthetic_sector("test", n_stocks=5, n_days=100)
    for t, df in data.items():
        print(f"{t}: {len(df)} days, price range {df['close'].min():.2f}-{df['close'].max():.2f}")
