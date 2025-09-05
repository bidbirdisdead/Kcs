"""Kalshi API adapter used by the bot.

In DRY mode we still fetch real market data (orderbooks/markets) when API
credentials and base URL are configured. DRY mode only suppresses order
execution; it does not synthesize orderbooks. LIVE mode executes orders.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional
import logging

# Support both package-style (backend.api.kalshi) and flat (api.kalshi) imports
try:  # when imported as backend.api.kalshi
    from ..config import TRADING_MODE
except Exception:  # when imported as api.kalshi with backend on sys.path
    from config import TRADING_MODE  # type: ignore

try:
    from ..kalshi_rest import get_market_order_book as _rest_get_ob
except Exception:
    _rest_get_ob = None  # type: ignore

try:
    # Prefer package import when running as backend.api.kalshi
    from ..kalshi_lookup import (
        find_eth_hourly_above_below,
        find_btc_hourly_above_below,
    )
except Exception:
    try:
        # Fallback to top-level import when running module directly
        from kalshi_lookup import (
            find_eth_hourly_above_below,
            find_btc_hourly_above_below,
        )
    except Exception:
        find_eth_hourly_above_below = None  # type: ignore
        find_btc_hourly_above_below = None  # type: ignore


def get_orderbook(ticker: str) -> Dict[str, Any]:
    """Return the normalized orderbook for the ticker.

    Uses the REST adapter to fetch real orderbooks in both DRY and LIVE modes
    when configured; falls back to a deterministic structure only if the
    REST client is unavailable or errors.
    """
    if _rest_get_ob is None:
        raise RuntimeError("REST adapter for Kalshi unavailable; real data required")
    # Propagate errors from REST to the caller so they can be handled explicitly
    return _rest_get_ob(ticker)

def get_markets(**kwargs) -> Dict[str, Any]:
    """Return a synthetic markets list for DRY runs.

    Exists to satisfy imports in higher-level helpers. Tests often monkeypatch
    this function; keeping a simple default avoids import errors.
    """
    return {"markets": []}

def get_positions(**kwargs) -> Dict[str, Any]:
    """Return a synthetic empty positions payload for DRY runs."""
    return {"market_positions": []}

def normalize_time_in_force(tif: Optional[str]) -> Optional[str]:
    """Normalize various TIF spellings to Kalshi's canonical values.

    - GTC/GTT/None -> None (default good-till-cancel/time behavior)
    - IOC -> "immediate_or_cancel"
    - FOK -> "fill_or_kill"
    """
    if tif is None:
        return None
    v = str(tif).strip().lower()
    if v in ("gtc", "gtt", "good_till_cancelled", "good_till_time"):
        return None
    if v in ("fok", "fill_or_kill"):
        return "fill_or_kill"
    if v in ("ioc", "immediate_or_cancel"):
        return "immediate_or_cancel"
    # tolerate already-normalized strings
    if v in ("fill_or_kill", "immediate_or_cancel"):
        return v
    raise ValueError(f"unsupported time_in_force value: {tif}")


def create_order_compat(ticker: str, side: str, price_cents: int, qty: int, **kwargs) -> Dict[str, Any]:
    """Compatibility wrapper that simulates an order submission.

    In DRY mode the function returns a fake order result. In LIVE mode it
    raises NotImplementedError so developers are explicit when integrating.
    """
    if TRADING_MODE == "DRY":
        return {"order_id": "dry-" + str(int(time.time() * 1000)), "ticker": ticker, "side": side, "price_cents": price_cents, "qty": qty}
    raise NotImplementedError("LIVE Kalshi integration is not implemented in this shim")


def _strike_to_ticker(
    symbol: str, strike: float, relation: str = ">=", close_ts: Optional[int] = None
) -> Optional[str]:
    """Resolve a real Kalshi market ticker for the given symbol and strike.

    Uses the specialized resolver functions in `backend/kalshi_lookup.py` when
    available. Those functions query Kalshi's /series, /events and /markets
    endpoints to find the closest market for the requested strike and
    relation (>= or <=) and return a dict containing a `ticker` key.

    If the lookup helpers are unavailable or fail, fall back to the older
    synthetic ticker string to preserve behavior for DRY/testing.
    """
    base = (symbol.split("/")[0] if symbol else "").upper()

    # Prefer using the lookup helpers when present
    # Attempt to resolve using lookup helpers; raise explicit error on failure
    try:
        if base.startswith("ETH") and find_eth_hourly_above_below is not None:
            info = find_eth_hourly_above_below(strike, relation=relation, when_ts=close_ts)
            if info and isinstance(info, dict) and info.get("ticker"):
                return info.get("ticker")
        if base.startswith("BTC") and find_btc_hourly_above_below is not None:
            info = find_btc_hourly_above_below(strike, relation=relation, when_ts=close_ts)
            if info and isinstance(info, dict) and info.get("ticker"):
                return info.get("ticker")
    except Exception as e:
        raise RuntimeError(f"lookup helper failed: {e}") from e

    # If no helper returned a ticker, fail explicitly rather than generating a synthetic one
    raise RuntimeError(f"Could not resolve real ticker for {symbol} strike={strike} relation={relation}")
