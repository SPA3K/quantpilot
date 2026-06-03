"""QuantPilot MCP Server — exposes investment research tools via MCP protocol."""

import json
import logging
import sys

from mcp.server.fastmcp import FastMCP

from quantpilot.config import ensure_dirs

# Initialize directories
ensure_dirs()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("quantpilot")

# Create MCP server
mcp = FastMCP(
    "QuantPilot",
    instructions="AI-Powered Investment Research Workbench for A-Share Market. "
                 "Provides sector-level ML factor ranking, stock analysis, "
                 "and model training via LightGBM + SHAP.",
)


@mcp.tool()
def sector_ranking(sector: str, top_n: int = 10) -> str:
    """Query stock rankings within a sector using the trained ML model.

    Args:
        sector: Sector ID (cpo/pcb/ai/embodied/consumer/new_energy) or Chinese name
        top_n: Number of top stocks to return (default 10)

    Returns JSON with ranking, SHAP factor contributions, and model metrics.
    """
    from quantpilot.tools.ranking import sector_ranking as _rank
    return _rank(sector, top_n)


@mcp.tool()
def train_sector_model(
    sector: str,
    start_date: str = "20240101",
    end_date: str = "",
    target: str = "ret_20d",
    n_splits: int = 5,
) -> str:
    """Train a LightGBM factor ranking model for a sector.

    Args:
        sector: Sector name (Chinese) or ID. Examples: "CPO光模块", "cpo", "半导体"
        start_date: Training data start date (YYYYMMDD, default "20240101")
        end_date: Training data end date (YYYYMMDD, default today)
        target: Prediction target (default "ret_20d" = 20-day forward return)
        n_splits: Number of purged walk-forward CV folds (default 5)

    Returns training metrics (IC, IR, Sharpe), SHAP factor importance, and model info.
    Training takes 30-120 seconds depending on sector size.
    """
    from quantpilot.tools.training import train_sector_model as _train
    return _train(sector, start_date, end_date or None, target, n_splits)


@mcp.tool()
def stock_analysis(ticker: str) -> str:
    """Analyze a single stock using its sector's ML model.

    Args:
        ticker: Stock code, e.g. "002475" (立讯精密), "600519" (贵州茅台)

    Returns sector ranking position, model score, SHAP factor decomposition,
    and fundamental data (PE, PB, market cap).
    """
    from quantpilot.tools.analysis import stock_analysis as _analyze
    return _analyze(ticker)


@mcp.tool()
def compare_sectors(sectors: list[str], metric: str = "sharpe") -> str:
    """Compare multiple sectors' factor weights and model performance.

    Args:
        sectors: List of sector IDs or names, e.g. ["cpo", "pcb", "ai"]
        metric: Comparison metric: "sharpe", "ic", "ir" (default "sharpe")

    Returns side-by-side comparison of model metrics and top factors.
    """
    from quantpilot.tools.analysis import compare_sectors as _compare
    return _compare(sectors, metric)


@mcp.tool()
def list_sectors() -> str:
    """List all available sectors with model status.

    Shows which sectors have pre-trained models, stock counts,
    and model performance metrics.
    """
    from quantpilot.tools.analysis import list_sectors_tool as _list
    return _list()


@mcp.tool()
def get_model_info(sector: str) -> str:
    """Get detailed model information for a sector.

    Args:
        sector: Sector ID or name

    Returns training date, factor columns, hyperparameters, and metrics.
    """
    from quantpilot.tools.analysis import get_model_info_tool as _info
    return _info(sector)


@mcp.tool()
def backtest_model(
    sector: str,
    start_date: str = "20240101",
    end_date: str = "",
    top_pct: float = 0.2,
) -> str:
    """Backtest a sector model over a date range.

    Args:
        sector: Sector ID or name
        start_date: Backtest start date (YYYYMMDD, default "20240101")
        end_date: Backtest end date (YYYYMMDD, default today)
        top_pct: Top percentage of stocks to hold (default 0.2 = top 20%)

    Returns annual return, Sharpe ratio, max drawdown, win rate, and equity curve.
    """
    from quantpilot.tools.analysis import backtest_model_tool as _bt
    return _bt(sector, start_date, end_date or None, top_pct)


def main():
    """Entry point for the MCP server."""
    logger.info("Starting QuantPilot MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
