'''
Pure functions for calculating order book imbalance and other microstructure features.
'''
from __future__ import annotations
from typing import Any, Dict, Optional

# Ensure this import works for both package and flat layouts
try:
    from ..orderbook_utils import yes_best_bid_ask_from_orderbook
except (ImportError, ValueError):
    from orderbook_utils import yes_best_bid_ask_from_orderbook # type: ignore

def top_of_book(orderbook: dict) -> dict[str, float | None]:
    """Extracts the best bid and ask prices and sizes from a Kalshi orderbook dict."""
    if not orderbook or not isinstance(orderbook, dict):
        return {'yes_price': None, 'yes_size': None, 'no_price': None, 'no_size': None}

    yes_levels = orderbook.get("yes", [])
    no_levels = orderbook.get("no", [])

    yes_price = float(yes_levels[0][0]) if yes_levels and yes_levels[0] else None
    yes_size = float(yes_levels[0][1]) if yes_levels and yes_levels[0] else None
    no_price = float(no_levels[0][0]) if no_levels and no_levels[0] else None
    no_size = float(no_levels[0][1]) if no_levels and no_levels[0] else None

    return {'yes_price': yes_price, 'yes_size': yes_size, 'no_price': no_price, 'no_size': no_size}


def order_book_imbalance(orderbook: dict) -> float | None:
    """Calculates the simple top-of-book order book imbalance (OBI). OBI = bid_vol / (bid_vol + ask_vol)."""
    tob = top_of_book(orderbook)
    yes_size = tob.get('yes_size')
    no_size = tob.get('no_size')

    if yes_size is None or no_size is None:
        return None

    denominator = yes_size + no_size
    if denominator == 0:
        return None

    return yes_size / denominator


def weighted_mid_price(orderbook: dict) -> float | None:
    """Calculates the weighted mid-price as specified in the implementation plan."""
    tob = top_of_book(orderbook)
    yes_price, yes_size = tob.get('yes_price'), tob.get('yes_size')
    no_price, no_size = tob.get('no_price'), tob.get('no_size')

    # Explicitly check for None to handle cases where a size or price might be 0
    if any(v is None for v in [yes_price, yes_size, no_price, no_size]):
        return None

    denominator = yes_size + no_size
    if denominator == 0:
        return None

    # Formula from addplan.txt: (yes_price*yes_size + no_price*no_size) / (yes_size + no_size)
    wmp = ((yes_price * yes_size) + (no_price * no_size)) / denominator
    return wmp


def bid_ask_spread(orderbook: dict) -> int | None:
    """Calculates the bid-ask spread using the project's canonical helper."""
    if not orderbook:
        return None

    # Use the existing helper to ensure consistent derivation of "ask"
    yes_bid, yes_ask, _, _ = yes_best_bid_ask_from_orderbook(orderbook)

    if yes_bid is None or yes_ask is None:
        return None

    return yes_ask - yes_bid
