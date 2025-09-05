import uuid


def build_v2_limit_order(ticker: str, action: str, side: str, price_cents: int, qty: int,
                         post_only: bool | None = None, client_order_id: str | None = None) -> dict:
    # Normalize & validate
    action = action.strip().lower()         # "buy" | "sell"
    side = side.strip().lower()             # "yes" | "no"
    assert action in ("buy", "sell")
    assert side in ("yes", "no")
    price_cents = int(price_cents)
    assert 1 <= price_cents <= 99
    qty = int(qty)
    assert qty >= 1

    # Build payload that matches Kalshi’s example exactly
    order = {
        "ticker": ticker,
        "action": action,
        "side": side,
        "count": qty,
        "type": "limit",
        f"{side}_price": price_cents,       # yes_price or no_price
        "client_order_id": client_order_id or str(uuid.uuid4()),
    }
    if post_only is True:
        order["post_only"] = True
    return order
