"""Data provider interface."""

from abc import ABC, abstractmethod
import pandas as pd


class DataProvider(ABC):
    """数据提供者接口"""

    @abstractmethod
    def get_bars(
        self,
        ticker: str,
        start: str,
        end: str,
        freq: str = "daily",
    ) -> pd.DataFrame:
        """
        获取OHLCV数据。

        Args:
            ticker: 股票代码或名称
            start: 起始日期 "YYYY-MM-DD"
            end: 结束日期 "YYYY-MM-DD"
            freq: "daily" | "minute"

        Returns:
            DataFrame with columns: [date, open, high, low, close, volume, turnover]
        """
        ...

    @abstractmethod
    def resolve_ticker(self, name: str) -> str:
        """将股票名称解析为代码"""
        ...
