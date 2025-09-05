from typing import Optional, Tuple


def yes_best_bid_ask_from_orderbook(ob: dict) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Kalshi orderbook returns *bids* for yes and no. There are no explicit asks.
    yes_ask = 100 - best_no_bid ; no_ask = 100 - best_yes_bid
    Returns (yes_bid, yes_ask, no_bid, no_ask) as ints or None.
    """
    def best(levels):  # levels like [[price, size], ...] or [{'price': .., 'size': ..}, ...]
        try:
            best_price = None
            for lv in levels or []:
                price = None
                if isinstance(lv, (list, tuple)) and len(lv) >= 1:
                    price = float(lv[0])
                elif isinstance(lv, dict):
                    v = lv.get('price', lv.get('p'))
                    if v is not None:
                        price = float(v)
                if price is None:
                    continue
                if best_price is None or price > best_price:
                    best_price = price
            return int(round(best_price)) if best_price is not None else None
        except Exception:
            return None

    yes_bid = best(ob.get("yes", []))
    no_bid = best(ob.get("no", []))

    yes_ask = 100 - no_bid if no_bid is not None else None
    no_ask = 100 - yes_bid if yes_bid is not None else None
    return yes_bid, yes_ask, no_bid, no_ask


def clamp_post_only(side: str, action: str, target_px: int, yes_bid, yes_ask, no_bid, no_ask) -> int:
    """
    For post_only:
      - BUY YES must be < yes_ask
      - SELL YES must be > yes_bid
      - BUY NO  must be < no_ask
      - SELL NO  must be > no_bid
    """
    px = max(1, min(99, int(target_px)))
    side = side.lower()
    action = action.lower()
    if side == "yes":
        if action == "buy":
            return min(px, max(1, (yes_ask or 100) - 1))
        else:  # sell yes
            return max(px, (yes_bid or 0) + 1)
    else:  # side == "no"
        if action == "buy":
            return min(px, max(1, (no_ask or 100) - 1))
        else:  # sell no
            return max(px, (no_bid or 0) + 1)
