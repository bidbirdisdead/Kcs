"""Trade logic helpers for Kalshi trading bot."""

from __future__ import annotations
from typing import Any, Dict, Optional


def is_trade_opportunity(
    best_yes: Optional[int],
    true_yes_ask: Optional[int],
    yes_liq: int,
    min_spread: int,
    min_liquidity: int,
    max_price: int,
) -> bool:
    """Return True if market meets opportunity criteria."""
    if (
        best_yes is not None
        and true_yes_ask is not None
        and (true_yes_ask - best_yes) >= min_spread
        and yes_liq >= min_liquidity
        and best_yes <= max_price
    ):
        return True
    return False


def score_opportunity(
    best_yes: Optional[int],
    true_yes_ask: Optional[int],
    yes_liq: int,
    min_spread: int,
    min_liquidity: int,
    max_price: int,
) -> float:
    """Return a score for the opportunity (higher is better)."""
    if not is_trade_opportunity(best_yes, true_yes_ask, yes_liq, min_spread, min_liquidity, max_price):
        return 0.0
    if best_yes is None or true_yes_ask is None:
        return 0.0
    spread = true_yes_ask - best_yes
    return spread * yes_liq / (best_yes + 1)
