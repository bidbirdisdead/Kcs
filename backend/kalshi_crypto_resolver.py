import os, re, time, logging, requests
from datetime import datetime, timezone
from functools import lru_cache

BASE = f"https://{os.getenv('KALSHI_BASE', 'api.elections.kalshi.com')}/trade-api/v2"
S = requests.Session()
S.trust_env = False
S.proxies = {"http": None, "https": None}
TIMEOUT = (4, 8)

def _get(path, **params):
    # Reuse centralized _auth_headers when available to avoid duplication and
    # keep bearer logic consistent with kalshi_rest._req()
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

def _iso_to_ts(s: str) -> int:
    # handles ...Z
    return int(datetime.fromisoformat(s.replace("Z","+00:00")).timestamp())

def _round_to_step(x: float, step: int) -> int:
    return int(round(x / step) * step)

def _extract_level_int(s: str) -> int | None:
    s = s.replace(",", "")
    m = re.search(r"\d{3,6}(?:\.\d+)?", s)
    return int(float(m.group())) if m else None

def _title_has_relation(title_l: str, relation: str) -> bool:
    if relation == ">=":
        return ("above" in title_l) or ("over" in title_l) or ("≥" in title_l) or ("at or above" in title_l)
    else:
        return ("below" in title_l) or ("under" in title_l) or ("≤" in title_l) or ("at or below" in title_l)

def _pick_event_for_hour(events, target_close_ts: int, tolerance_s: int = 2*3600):
    best = None
    best_d = 10**9
    for ev in events:
        # prefer events with nested markets (we asked for it)
        mkts = ev.get("markets") or []
        if not mkts:
            continue
        # take the close_time of any market (same across the event)
        try:
            ct = _iso_to_ts(mkts[0]["close_time"])
        except Exception:
            continue
        d = abs(ct - target_close_ts)
        if d <= tolerance_s and d < best_d:
            best = ev; best_d = d
    return best

@lru_cache(maxsize=4096)
def resolve_crypto_market(series_ticker: str, strike: int, relation: str, close_ts: int, step: int):
    """
    series_ticker: 'KXETHD' or 'KXBTCD'
    strike: desired rounded strike (e.g., 4540 or 111750)
    relation: '>=' or '<='
    close_ts: UNIX seconds of the hour you're targeting
    step: 20 for ETH, 250 for BTC
    Returns: dict {ticker,title,close_time, yes_bid, yes_ask, no_bid, no_ask}
    """
    assert relation in (">=", "<=")
    want = _round_to_step(strike, step)

    # 1) get open events for the series, with nested markets
    evs = _get(f"/series/{series_ticker}/events", status="open", with_nested_markets=True).get("events", [])
    ev = _pick_event_for_hour(evs, close_ts)
    if not ev:
        raise RuntimeError(f"No open event near close_ts for series {series_ticker}")

    # 2) pick market: filter by relation words and closest strike in title/subtitle
    candidates = []
    for m in ev.get("markets", []):
        title = ((m.get("title") or "") + " " + (m.get("subtitle") or "")).strip()
        tl = title.lower()
        lvl = _extract_level_int(title)
        if lvl is None:
            continue
        rel_score = 0 if _title_has_relation(tl, relation) else (step // 2)  # penalize wrong side
        dist = abs(lvl - want) + rel_score
        candidates.append((dist, m))

    if not candidates:
        raise RuntimeError(f"No candidate markets in {series_ticker} event for strike~{want} {relation}")

        if not candidates:
            raise RuntimeError(f"No candidate markets in {series_ticker} event for strike~{want} {relation}")
        except Exception: return None

    yes_bid = _best(ob.get("yes", []))
    no_bid  = _best(ob.get("no", []))
    yes_ask = None if no_bid  is None else 100 - no_bid
    no_ask  = None if yes_bid is None else 100 - yes_bid

    return {
        "ticker": ticker,
        "title": mk.get("title"),
        "subtitle": mk.get("subtitle"),
        "close_time": mk.get("close_time"),
        "series_ticker": series_ticker,
        "yes_bid": yes_bid, "yes_ask": yes_ask,
        "no_bid": no_bid,   "no_ask": no_ask,
    }

# Example usage (from a trading worker):
# underlying = "ETH" or "BTC"
# series = "KXETHD" if underlying=="ETH" else "KXBTCD"
# step = 20 if underlying=="ETH" else 250
# meta = resolve_crypto_market(series, strike_rounded, relation, close_ts, step)
# ticker = meta["ticker"]
