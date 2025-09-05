from __future__ import annotations
from typing import Optional, Literal, Dict, Any
import uuid

# Import your existing low-level API wrapper
try:
    # if your kalshi_trade lives under backend/
    from backend.kalshi_trade import (
        place_order as _raw_place,
        amend_order as _raw_amend,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for flat layout
    from kalshi_trade import (  # type: ignore
        place_order as _raw_place,
        amend_order as _raw_amend,
    )

Side = Literal["yes", "no"]


def _cid() -> str:
    return str(uuid.uuid4())


def submit_entry_buy(
    *,
    ticker: str,
    side: Side,
    price_cents: int,
    qty: int,
    post_only: bool = True,  # kept for API parity; ignored by low-level wrapper
) -> Dict[str, Any]:
    """
    Places an ENTRY order (buy side of the position) and returns a dict.
    Forwards to the low-level Kalshi API wrapper.
    """
    # Forward using the signature expected by the low-level wrapper
    res = _raw_place(
        market_ticker=str(ticker),
        side=str(side),
        count=int(qty),
        price=int(price_cents),
        order_type="limit",
        action="buy",
        client_order_id=_cid(),
    )
    if not isinstance(res, dict):
        raise RuntimeError("submit_entry_buy: underlying place_order returned non-dict")
    return res


def submit_exit_sell(
    *,
    ticker: str,
    side: Side,
    price_cents: int,
    qty: int,
    tif: Optional[str] = None,  # retained for compatibility; not used by low-level wrapper
) -> Dict[str, Any]:
    """
    Places an EXIT order (sell to close) and returns a dict.
    Forwards to the low-level Kalshi API wrapper.
    """
    res = _raw_place(
        market_ticker=str(ticker),
        side=str(side),
        count=int(qty),
        price=int(price_cents),
        order_type="limit",
        action="sell",
        client_order_id=_cid(),
    )
    if not isinstance(res, dict):
        raise RuntimeError("submit_exit_sell: underlying place_order returned non-dict")
    return res


def amend_order(
    order_id: str,
    *,
    side: Optional[Side] = None,
    price_cents: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Amends an existing order (price/side). Returns a dict.
    Forwards to the low-level Kalshi API wrapper.
    """
    payload: Dict[str, Any] = {"order_id": str(order_id)}
    if side is not None:
        payload["side"] = str(side)
    if price_cents is not None:
        payload["price_cents"] = int(price_cents)
    res = _raw_amend(**payload)
    if not isinstance(res, dict):
        raise RuntimeError("amend_order: underlying amend_order returned non-dict")
    return res
# --- IGNORE ---
