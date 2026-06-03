"""Sector management — resolve sector IDs to stock lists."""

import logging
from datetime import datetime

from quantpilot.config import SECTORS
from quantpilot.data.cache import DataCache
from quantpilot.data.fetcher import get_sector_stocks

logger = logging.getLogger(__name__)

_cache = DataCache()


def get_current_quarter() -> str:
    """Return current quarter string like '2026Q2'."""
    now = datetime.now()
    q = (now.month - 1) // 3 + 1
    return f"{now.year}Q{q}"


def resolve_sector(sector_input: str) -> dict | None:
    """Resolve a sector input to its config.

    Args:
        sector_input: Sector ID (e.g. "cpo"), Chinese name (e.g. "CPO光模块"),
                      or concept_id (e.g. "BK1195").

    Returns:
        Sector config dict or None if not found.
    """
    sector_input_lower = sector_input.lower().strip()

    # Match by ID
    for sid, cfg in SECTORS.items():
        if sid == sector_input_lower:
            return {"id": sid, **cfg}

    # Match by Chinese name
    for sid, cfg in SECTORS.items():
        if cfg["name"] == sector_input or cfg["name"].lower() == sector_input_lower:
            return {"id": sid, **cfg}

    # Match by concept_id
    for sid, cfg in SECTORS.items():
        if cfg.get("concept_id") == sector_input:
            return {"id": sid, **cfg}

    return None


def get_sector_stocks_cached(sector_id: str) -> list[dict]:
    """Get sector stocks with caching (quarter-based)."""
    quarter = get_current_quarter()

    # Try cache first
    cached = _cache.get_sector(sector_id, quarter)
    if cached:
        return cached

    # Fetch from API
    cfg = SECTORS.get(sector_id)
    if not cfg:
        return []

    concept_id = cfg.get("concept_id")
    if not concept_id:
        return []

    stocks = get_sector_stocks(concept_id)
    if stocks:
        _cache.save_sector(sector_id, quarter, stocks)

    return stocks


def list_all_sectors() -> list[dict]:
    """List all configured sectors with stock counts."""
    result = []
    for sid, cfg in SECTORS.items():
        stocks = get_sector_stocks_cached(sid)
        result.append({
            "id": sid,
            "name": cfg["name"],
            "n_stocks": len(stocks),
            "source": cfg["source"],
        })
    return result
