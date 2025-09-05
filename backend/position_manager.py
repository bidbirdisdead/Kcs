"""Position management helpers for Kalshi trading bot."""

from __future__ import annotations
from typing import Any, Dict, Optional
from api.kalshi import get_positions


def get_open_positions() -> Dict[str, Any]:
    """Fetch all open positions from Kalshi API."""
    return get_positions()


def get_position_for_market(market_ticker: str) -> Optional[Dict[str, Any]]:
    """Return the open position for a specific market, if any."""
    positions = get_positions(ticker=market_ticker)
    for pos in positions.get("market_positions", []):
        if pos.get("ticker") == market_ticker:
            return pos
    return None


def get_total_exposure() -> float:
    """Calculate total exposure across all open positions (in USD)."""
    positions = get_positions()
    total = 0.0
    for pos in positions.get("market_positions", []):
        count = pos.get("count", 0)
        price = pos.get("avg_price", 0)
        total += (count * price) / 100.0
    return total


def check_exposure_limit(max_exposure: float) -> bool:
    """Return True if total exposure is below the given limit (USD)."""
    return get_total_exposure() < max_exposure
