"""Local cache for stock data — avoids repeated API calls."""

import json
import logging
from pathlib import Path

import pandas as pd

from quantpilot.config import CACHE_DIR

logger = logging.getLogger(__name__)


class DataCache:
    """Simple file-based cache for stock data."""

    def __init__(self, base_dir: Path = CACHE_DIR):
        self.base_dir = base_dir
        self.quotes_dir = base_dir / "daily_quotes"
        self.sectors_dir = base_dir / "sector_stocks"
        self.quotes_dir.mkdir(parents=True, exist_ok=True)
        self.sectors_dir.mkdir(parents=True, exist_ok=True)

    def get_quotes(self, ticker: str, year: int) -> pd.DataFrame | None:
        """Load cached daily quotes for a ticker+year."""
        path = self.quotes_dir / f"{ticker}_{year}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def save_quotes(self, ticker: str, year: int, df: pd.DataFrame):
        """Save daily quotes to cache."""
        path = self.quotes_dir / f"{ticker}_{year}.parquet"
        df.to_parquet(path, index=False)
        logger.debug(f"Cached quotes: {path}")

    def get_sector(self, sector_id: str, quarter: str) -> list[dict] | None:
        """Load cached sector constituents."""
        path = self.sectors_dir / f"{sector_id}_{quarter}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def save_sector(self, sector_id: str, quarter: str, stocks: list[dict]):
        """Save sector constituents."""
        path = self.sectors_dir / f"{sector_id}_{quarter}.json"
        path.write_text(json.dumps(stocks, ensure_ascii=False, indent=2))
        logger.debug(f"Cached sector: {path}")

    def clear(self):
        """Clear all cached data."""
        import shutil
        shutil.rmtree(self.base_dir, ignore_errors=True)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cache cleared")
