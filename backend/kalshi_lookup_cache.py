from functools import lru_cache
from typing import Dict

from .kalshi_lookup import find_eth_hourly_above_below, find_btc_hourly_above_below


@lru_cache(maxsize=2048)
def resolve_crypto_ticker(underlying: str, relation: str, step: int, strike: int, close_ts: int) -> Dict:
    """
    Cached resolver for crypto hourly above/below tickers.

    Keyed by (underlying, relation, step, strike, close_ts) so callers get a stable
    lookup based on the explicit event parameters rather than any global "last used" key.

    underlying: 'ETH' or 'BTC'
    relation: '>=' or '<='
    step: integer ladder spacing (20 for ETH, 250 for BTC)
    strike: rounded ladder strike (e.g., 4520)
    close_ts: UNIX seconds for the event's hour close
    Returns: dict as returned by the underlying finder (contains 'ticker', 'title', 'close_time', etc.)
    """
    u = (underlying or "").upper()
    if u == "ETH":
        return find_eth_hourly_above_below(strike, relation=relation, step=step, when_ts=close_ts)
    elif u == "BTC":
        return find_btc_hourly_above_below(strike, relation=relation, step=step, when_ts=close_ts)
    else:
        raise ValueError(f"Unsupported underlying for crypto ticker resolution: {underlying}")


def _extract_level_int(s: str) -> int:
    """Parse an integer strike/level from a market title or subtitle.

    Returns -1 if no numeric level is found.
    """
    import re
    s = (s or "").replace(",", "")
    nums = re.findall(r"\d{3,6}(?:\.\d+)?", s)
    return int(float(nums[0])) if nums else -1


def assert_market_matches(order_ctx: Dict, market_meta: Dict):
    """
    Validate that the resolved market metadata looks like the requested order context.

    order_ctx: {'symbol': 'ETH/USDT', 'strike': 4520, 'relation': '>=', 'close_ts': ... , 'step': 20}
    market_meta: resolver return dict
    Raises RuntimeError on mismatch.
    """
    title = (market_meta.get("title") or "").lower()
    sym_hint = "eth" if order_ctx["symbol"].lower().startswith("eth") else "btc"
    if sym_hint not in title:
        raise RuntimeError(f"Resolved ticker '{market_meta.get('ticker')}' does not look like {order_ctx['symbol']} (title={market_meta.get('title')})")
    # strike sanity: allow ± step/2 tolerance due to naming
    if abs(int(order_ctx["strike"]) - _extract_level_int(title)) > (int(order_ctx.get("step", 0)) // 2):
        raise RuntimeError(f"Resolved ticker strike mismatch: want ~{order_ctx['strike']} got title='{market_meta.get('title')}'")
