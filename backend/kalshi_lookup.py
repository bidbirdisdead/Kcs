import time, math, logging, os, requests
from datetime import datetime, timezone
import re

BASE = f"https://{os.getenv('KALSHI_BASE', 'api.elections.kalshi.com')}/trade-api/v2"
S = requests.Session()
S.trust_env = False
S.proxies = {"http": None, "https": None}
TIMEOUT = (4, 8)


def _get(path, **params):
    # Prefer the centralized bearer header helper from kalshi_rest when present
    # Use the centralized auth helper when available to prevent duplication
    try:
        from api.auth import get_auth_headers  # type: ignore
        headers = get_auth_headers(allow_empty=True)
    except Exception:
        token = os.getenv("KALSHI_API_KEY")
        headers = {"Authorization": f"Bearer {token}"} if token else None

    request_kwargs = {"params": params, "timeout": TIMEOUT}
    if headers:
        request_kwargs["headers"] = headers

    r = S.get(f"{BASE}{path}", **request_kwargs)
    r.raise_for_status()
    return r.json()


def _unix_now():
    return int(time.time())


def _round_to_step(x, step):
    return int(round(x / step) * step)


def _parse_strike_from_ticker(t: str):
    try:
        m = re.search(r"-T([0-9]+(?:\.[0-9]+)?)$", str(t))
        return float(m.group(1)) if m else None
    except Exception:
        return None


def find_eth_hourly_above_below(strike, relation=">=", step=20, when_ts=None):
    """
    Resolve the Kalshi market ticker for the ETH hourly 'above/below' style markets.

    relation: '>=' or '<=' (we'll map to the correct direction)
    step: Kalshi strike grid spacing (you said ETH uses $20)
    when_ts: UNIX seconds for the *hour close* you care about; default = next top-of-hour (exchange is ET; we just pick the next hour universally)
    Returns: dict {ticker, close_time, title, yes_bid, yes_ask, no_bid, no_ask}
    """
    # 1) Normalize target hour (next top-of-hour by default)
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    if when_ts is None:
        target_hour = now.replace(minute=0, second=0, microsecond=0)
        if now.minute or now.second or now.microsecond:
            target_hour = target_hour.replace(hour=(target_hour.hour + 1) % 24)
        when_ts = int(target_hour.timestamp())

    # 2) Pull markets directly for the ETH series (per docs: GET /markets with series_ticker)
    series_ticker = os.getenv("KALSHI_SERIES_ETH", "KXETHD")
    markets = _get("/markets", series_ticker=series_ticker, status="open", limit=1000).get("markets") or []
    target_markets = []
    for mkt in markets:
        ct_iso = mkt.get("close_time")
        try:
            ct = int(datetime.fromisoformat(str(ct_iso).replace("Z","+00:00")).timestamp()) if ct_iso else None
        except Exception:
            ct = None
        if ct is None:
            continue
        if abs(ct - when_ts) <= 2 * 3600:
            target_markets.append(mkt)

    if not target_markets:
        raise RuntimeError("No ETH markets are currently open/visible via API.")

    # 4) Choose the market whose strike in the title is closest to our rounded ladder strike
    #    (Titles usually contain the level, e.g., 'ETH ≥ 4520 at 3pm' or similar.)
    want = _round_to_step(float(strike), step)
    def extract_level(s):
        # grab first integer-ish token from title/subtitle
        import re
        nums = re.findall(r"\d{3,6}(?:[.,]\d+)?", s.replace(",", ""))
        return int(float(nums[0])) if nums else None

    candidates = []
    for m in target_markets:
        lvl_f = _parse_strike_from_ticker(m.get("ticker"))
        if lvl_f is None:
            continue
        lvl = int(round(lvl_f))
        dist = abs(lvl - want)
        txt = ((m.get("title") or "") + " " + (m.get("subtitle") or "")).lower()
        if relation == ">=" and ("above" not in txt and "over" not in txt and "≥" not in txt and "at or above" not in txt):
            dist += step // 2
        if relation == "<=" and ("below" not in txt and "under" not in txt and "≤" not in txt and "at or below" not in txt):
            dist += step // 2
        candidates.append((dist, m))

    if not candidates:
        raise RuntimeError(f"No ETH markets matched strike near {want}")

    candidates.sort(key=lambda x: x[0])
    chosen = candidates[0][1]
    ticker = chosen["ticker"]

    # 5) Enrich with live book (so you can clamp resting orders)
    mk = _get(f"/markets/{ticker}")["market"]               # docs: Get Market
    ob = _get(f"/markets/{ticker}/orderbook").get("orderbook", {})  # docs: Get Market Order Book

    def best(levels):
        # API returns bids only; choose highest by price
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
    no_bid  = best(ob.get("no", []))        # NB: no_bid is the price to sell NO at (your side)
    yes_ask = 100 - (no_bid or 0) if no_bid is not None else None
    no_ask  = 100 - (yes_bid or 0) if yes_bid is not None else None

    return {
        "ticker": ticker,
        "title": mk.get("title"),
        "close_time": mk.get("close_time"),
        "yes_bid": yes_bid, "yes_ask": yes_ask,
        "no_bid": no_bid,   "no_ask": no_ask,
    }


def find_btc_hourly_above_below(strike, relation=">=", step=250, when_ts=None):
    """
    Resolve the Kalshi market ticker for the BTC hourly 'above/below' style markets.

    This mirrors find_eth_hourly_above_below but looks for BTC series and
    uses a larger default step size (e.g. $250).

    relation: '>=' or '<='
    step: Kalshi strike grid spacing (default 250 for BTC)
    when_ts: UNIX seconds for the *hour close* you care about; default = next top-of-hour
    Returns: dict {ticker, close_time, title, yes_bid, yes_ask, no_bid, no_ask}
    """
    # 1) Normalize target hour (next top-of-hour by default)
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    if when_ts is None:
        target_hour = now.replace(minute=0, second=0, microsecond=0)
        if now.minute or now.second or now.microsecond:
            target_hour = target_hour.replace(hour=(target_hour.hour + 1) % 24)
        when_ts = int(target_hour.timestamp())

    # 2) Pull markets directly for the BTC series (per docs: GET /markets with series_ticker)
    series_ticker = os.getenv("KALSHI_SERIES_BTC", "KXBTCD")
    markets = _get("/markets", series_ticker=series_ticker, status="open", limit=1000).get("markets") or []
    target_markets = []
    for mkt in markets:
        ct_iso = mkt.get("close_time")
        try:
            ct = int(datetime.fromisoformat(str(ct_iso).replace("Z","+00:00")).timestamp()) if ct_iso else None
        except Exception:
            ct = None
        if ct is None:
            continue
        if abs(ct - when_ts) <= 2 * 3600:
            target_markets.append(mkt)

    if not target_markets:
        raise RuntimeError("No BTC markets are currently open/visible via API.")

    # 4) Choose the market whose strike in the title is closest to our rounded ladder strike
    want = _round_to_step(float(strike), step)
    def extract_level(s):
        import re
        nums = re.findall(r"\d{3,6}(?:[.,]\d+)?", s.replace(",", ""))
        return int(float(nums[0])) if nums else None

    candidates = []
    for m in target_markets:
        lvl_f = _parse_strike_from_ticker(m.get("ticker"))
        if lvl_f is None:
            continue
        lvl = int(round(lvl_f))
        dist = abs(lvl - want)
        txt = ((m.get("title") or "") + " " + (m.get("subtitle") or "")).lower()
        if relation == ">=" and ("above" not in txt and "over" not in txt and "≥" not in txt and "at or above" not in txt):
            dist += step // 2
        if relation == "<=" and ("below" not in txt and "under" not in txt and "≤" not in txt and "at or below" not in txt):
            dist += step // 2
        candidates.append((dist, m))

    if not candidates:
        raise RuntimeError(f"No BTC markets matched strike near {want}")

    candidates.sort(key=lambda x: x[0])
    chosen = candidates[0][1]
    ticker = chosen["ticker"]

    # 5) Enrich with live book
    mk = _get(f"/markets/{ticker}")["market"]
    ob = _get(f"/markets/{ticker}/orderbook").get("orderbook", {})

    def best(levels):
        try:
            return int(levels[0][0])
        except Exception:
            return None

    yes_bid = best(ob.get("yes", []))
    no_bid = best(ob.get("no", []))
    yes_ask = 100 - (no_bid or 0) if no_bid is not None else None
    no_ask = 100 - (yes_bid or 0) if yes_bid is not None else None

    return {
        "ticker": ticker,
        "title": mk.get("title"),
        "close_time": mk.get("close_time"),
        "yes_bid": yes_bid, "yes_ask": yes_ask,
        "no_bid": no_bid,   "no_ask": no_ask,
    }
