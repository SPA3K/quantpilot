"""Data fetcher — pulls stock data from akshare."""

import logging
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)


def get_sector_stocks(sector_id: str) -> list[dict]:
    """Get constituent stocks for an eastmoney concept sector.

    Returns list of {"code": "002475", "name": "立讯精密"}.
    """
    try:
        df = ak.stock_board_concept_cons_em(symbol=sector_id)
        return [
            {"code": row["代码"], "name": row["名称"]}
            for _, row in df.iterrows()
        ]
    except Exception as e:
        logger.error(f"Failed to get sector stocks for {sector_id}: {e}")
        return []


def get_stock_daily(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get daily OHLCV + turnover for a single stock.

    Args:
        ticker: Stock code, e.g. "002475"
        start_date: "20240101"
        end_date: "20260601"

    Returns DataFrame with columns: date, open, high, low, close, volume, turnover, turnover_rate
    """
    try:
        df = ak.stock_zh_a_hist(
            symbol=ticker,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "turnover",
            "换手率": "turnover_rate",
        })
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = ticker
        return df[["date", "ticker", "open", "high", "low", "close",
                    "volume", "turnover", "turnover_rate"]]
    except Exception as e:
        logger.error(f"Failed to get daily data for {ticker}: {e}")
        return pd.DataFrame()


def get_stock_financial(ticker: str) -> dict:
    """Get latest financial snapshot for a stock.

    Returns dict with: pe, pb, roe, revenue_growth, profit_growth, etc.
    """
    try:
        df = ak.stock_individual_info_em(symbol=ticker)
        info = dict(zip(df["item"], df["value"]))
        return {
            "ticker": ticker,
            "pe": _safe_float(info.get("市盈率(动态)")),
            "pb": _safe_float(info.get("市净率")),
            "total_mv": _safe_float(info.get("总市值")),
            "circ_mv": _safe_float(info.get("流通市值")),
        }
    except Exception as e:
        logger.error(f"Failed to get financials for {ticker}: {e}")
        return {"ticker": ticker}


def get_north_flow(ticker: str, days: int = 10) -> pd.DataFrame:
    """Get recent north-bound capital flow for a stock (approximate)."""
    try:
        df = ak.stock_individual_fund_flow(stock=ticker, market="sh" if ticker.startswith("6") else "sz")
        df = df.rename(columns={"日期": "date", "主力净流入-净额": "main_net"})
        df["date"] = pd.to_datetime(df["date"])
        return df.tail(days)[["date", "main_net"]]
    except Exception as e:
        logger.warning(f"North flow unavailable for {ticker}: {e}")
        return pd.DataFrame()


def _safe_float(val) -> float | None:
    """Safely convert to float."""
    if val is None or val == "--" or val == "":
        return None
    try:
        s = str(val).replace(",", "").replace("%", "")
        return float(s)
    except (ValueError, TypeError):
        return None
