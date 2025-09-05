"""Tiny Kalshi REST portfolio client used by the bot.

This is a small, dependency-light wrapper around Kalshi portfolio endpoints
so the bot can create/cancel/amend and fetch fills.

Adjust KALSHI_API_BASE, KALSHI_API_KEY and KALSHI_API_SECRET via env vars.
"""
import os
import uuid
import json
import logging
import requests

# Prefer the centralized kalshi helpers from kalshi_api when available. This
# ensures a single source-of-truth for the host and auto-correction logic.
try:
    from api.kalshi import kalshi_url
    _use_kalshi_url = True
except Exception:
    kalshi_url = None  # type: ignore
    _use_kalshi_url = False

# Default to the documented Kalshi production host; users may override with KALSHI_API_BASE
# Default to elections host for consistency with other Kalshi helpers
KALSHI_API_BASE = os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2")
# Optional demo base for non-PROD environments
KALSHI_DEMO_BASE = os.getenv("KALSHI_DEMO_BASE", "")
KALSHI_ENV = os.getenv("KALSHI_ENV", "").upper()
KALSHI_API_KEY = os.getenv("KALSHI_API_KEY", "")
KALSHI_API_SEC = os.getenv("KALSHI_API_SECRET", "")
TRADING_MODE = os.getenv("TRADING_MODE", "DRY").upper()

# Use the canonical auth provider to avoid duplication across modules (PLANNED.TXT C2)
try:
    from api.auth import get_auth_headers  # type: ignore
except Exception:
    # Fall back to local function when import not possible (tests/offline)
    def get_auth_headers(allow_empty: bool = False):
        token = KALSHI_API_KEY
        if not token:
            if allow_empty:
                return {}
            raise ValueError("KALSHI_API_KEY not set; cannot build Authorization header")
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def _base_url() -> str:
    """Resolve the REST base URL with clear DEMO/PROD semantics.

    Priority:
    1) If centralized kalshi_url() helper is available, callers will use it.
    2) If KALSHI_ENV=DEMO and KALSHI_DEMO_BASE is set, use that.
    3) Otherwise use KALSHI_API_BASE (defaults to production host).
    """
    if KALSHI_ENV == "DEMO" and KALSHI_DEMO_BASE:
        return KALSHI_DEMO_BASE.rstrip("/")
    return KALSHI_API_BASE.rstrip("/")


def _req(method: str, path: str, payload=None, timeout=10):
    # DRY mode policy:
    # - Allow GET market data (orderbooks, markets, etc.) so we can simulate with real quotes
    # - Block mutating calls (POST/PATCH/DELETE) to prevent live execution
    if TRADING_MODE == "DRY" and method.upper() != "GET":
        logging.info("DRY mode - skipping HTTP %s %s; payload=%s", method, path, payload)
        return {"dry_run": True, "method": method, "path": path, "payload": payload}

    if _use_kalshi_url and kalshi_url is not None:
        url = kalshi_url(path)
    else:
        url = _base_url() + path

    headers = get_auth_headers(allow_empty=(TRADING_MODE == "DRY" and method.upper() == "GET"))
    r = requests.request(method, url, headers=headers,
                         data=json.dumps(payload) if payload else None, timeout=timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError as exc:
        # Log response body for easier diagnosis (401 body often explains reason)
        logging.error("HTTP %s %s -> %s: %s", method, url, r.status_code, (r.text or "<no body>"))
        raise
    resp = r.json() if r.text else {}
    # Validate response shape for common endpoints to avoid synthetic/cached surprises
    try:
        _validate_response(resp, path)
    except Exception as ve:
        logging.error("Response validation failed for %s %s: %s", method, url, ve)
        raise
    return resp


def _validate_response(resp, path: str):
    # Simple, conservative checks: if caller requested an orderbook path,
    # ensure the response contains an 'orderbook' or 'yes'/'no' sides.
    lp = path.lower()
    if "/orderbook" in lp:
        if not isinstance(resp, dict):
            raise ValueError("orderbook response not a dict")
        if "orderbook" not in resp and not ("yes" in resp and "no" in resp):
            # Allow some flexibility but require clear market fields
            raise ValueError("orderbook missing required keys")


# --- Portfolio endpoints: create/cancel/amend/batch ---
def create_order(ticker: str, side: str, price_cents: int, qty: int,
                 tif: str = "GTC", post_only: bool | None = None, client_order_id: str | None = None):
    # If a simple bearer API key is configured, use the lightweight REST wrapper.
    if KALSHI_API_KEY:
        # Normalize TIF using the shared helper if available, otherwise use a
        # local fallback implementation.
        try:
            from api.kalshi import normalize_time_in_force  # type: ignore
            normalized_tif = normalize_time_in_force(tif)
        except Exception:
            def _normalize_time_in_force_local(tif_val: str | None) -> str | None:
                if not tif_val:
                    return None
                v = str(tif_val).strip().lower()
                if v in ("fill_or_kill", "fill-or-kill", "fok"):
                    return "fill_or_kill"
                if v in ("immediate_or_cancel", "immediate-or-cancel", "ioc"):
                    return "immediate_or_cancel"
                if v in ("gtc", "good_till_cancelled", "gtt", "good_till_time"):
                    return None
                if v in ("fill_or_kill", "immediate_or_cancel"):
                    return v
                raise ValueError(f"unsupported time_in_force value: {tif_val}")

            normalized_tif = _normalize_time_in_force_local(tif)

        body = {
            "ticker": ticker,
            "side": side.lower(),
            "type": "limit",
            # add time_in_force only when not default
            **({"time_in_force": normalized_tif} if normalized_tif is not None else {}),
            "count": int(qty),
            "yes_price": int(price_cents) if side.lower() == "yes" else None,
            "no_price": int(price_cents) if side.lower() == "no" else None,
        }
        # Ensure a client_order_id is always present; Kalshi examples include
        # a UUID client_order_id. Generate one when not provided.
        if client_order_id:
            body["client_order_id"] = client_order_id
        else:
            body["client_order_id"] = str(uuid.uuid4())
        if post_only is not None:
            body["post_only"] = bool(post_only)
        body = {k: v for k, v in body.items() if v is not None}

        # Optional guard: validate ticker early using centralized kalshi_api.get_market
        # This helps fail fast on bad/expired tickers before sending a POST. Keep
        # it disabled in DRY mode and tolerant when kalshi_api isn't importable.
        if TRADING_MODE != "DRY":
            try:
                from api.kalshi import get_market  # type: ignore
                try:
                    _ = get_market(ticker)
                except Exception as ex:  # pragma: no cover - defensive
                    raise RuntimeError(f"kalshi ticker validation failed for '{ticker}': {ex}") from ex
            except Exception:
                # If import fails or network isn't available, continue — this
                # guard is best-effort and must not block DRY or offline runs.
                pass

        # If this is a post-only order, try routing through the safe submitter
        # in kalshi_api to handle 'post only cross' responses and attempt a
        # single safe reprice-and-retry. If the import or submit fails, fall
        # back to the lightweight bearer HTTP path for compatibility.
        if body.get("post_only"):
            try:
                # import from package-local module name if available
                from api.kalshi import submit_post_only_once  # type: ignore
                # submit_post_only_once expects an order dict including 'action'
                action = "buy" if body.get("side") == "yes" else "sell"
                order_for_submit = {"action": action, **body}
                return submit_post_only_once(order_for_submit)
            except Exception:
                logging.exception("submit_post_only_once failed; falling back to bearer POST")

        # Attempt the normal bearer POST; on certain 400 invalid_parameters
        # responses retry using a canonical v2 schema via kalshi_api.create_order
        try:
            return _req("POST", "/portfolio/orders", body)
        except Exception as e:
            msg = str(e)
            try:
                if "400" in msg and "invalid_parameters" in msg:
                    logging.warning("Bearer POST 400 invalid_parameters; retrying with canonical v2 schema via kalshi_api.create_order")
                    # Build a canonical v2-style minimal order
                    order = {
                        "action": "buy" if body.get("side") == "yes" else "sell",
                        "side": (body.get("side") or "").upper(),
                        "type": "limit",
                        "ticker": body.get("ticker"),
                        "count": int(body.get("count") or qty),
                        "price": int(body.get("yes_price") or body.get("no_price") or price_cents),
                    }
                    # include time_in_force when present
                    if body.get("time_in_force") is not None:
                        order["time_in_force"] = body.get("time_in_force")
                    # include post_only on first retry if present
                    if body.get("post_only") is not None:
                        order["post_only"] = bool(body.get("post_only"))

                    from api.kalshi import create_order as v2_create  # type: ignore
                    try:
                        return v2_create(order)
                    except Exception:
                        # final fallback: retry without post_only if it was present
                        if order.get("post_only"):
                            order.pop("post_only", None)
                            return v2_create(order)
                        raise
            except Exception:
                # If fallback path fails or condition not met, re-raise original
                pass
            raise

    # Otherwise, prefer the signed RSA flow from kalshi_api if available
    try:
        from api.kalshi import create_order_compat  # type: ignore
        logging.info("No KALSHI_API_KEY found; using signed kalshi_api.create_order_compat fallback")
        try:
            return create_order_compat(ticker, side, price_cents, qty, tif=tif, post_only=post_only, client_order_id=client_order_id)
        except Exception as e:
            msg = str(e)
            try:
                if "400" in msg and "invalid_parameters" in msg:
                    logging.warning("Signed create_order_compat 400 invalid_parameters; retrying with canonical v2 schema via kalshi_api.create_order")
                    order = {
                        "action": "buy" if (str(side).lower() == "yes") else "sell",
                        "side": (side or "").upper(),
                        "type": "limit",
                        "ticker": ticker,
                        "count": int(qty),
                        "price": int(price_cents),
                        "time_in_force": (None if tif is None else tif),
                    }
                    if post_only is not None:
                        order["post_only"] = bool(post_only)
                    from api.kalshi import create_order as v2_create  # type: ignore
                    try:
                        return v2_create(order)
                    except Exception:
                        if order.get("post_only"):
                            order.pop("post_only", None)
                            return v2_create(order)
                        raise
            except Exception:
                pass
            raise
    except Exception as exc:  # pragma: no cover - best-effort fallback
        logging.exception("Signed Kalshi API fallback failed: %s", exc)
        # Re-raise so callers get a clear exception
        raise

def cancel_order(order_id: str):
    # Use DELETE on the documented orders path
    return _req("DELETE", f"/portfolio/orders/{order_id}")

def amend_order(order_id: str, *, price_cents: int | None = None, size: int | None = None):
    # Patch the existing order with new price/size using documented orders path
    body: dict = {}
    if price_cents is not None:
        # Align with create_order naming conventions
        body["yes_price"] = int(price_cents)
    if size is not None:
        body["count"] = int(size)
    if not body:
        raise ValueError("no fields to amend")
    return _req("PATCH", f"/portfolio/orders/{order_id}", body)

def batch_create(orders: list[dict]):
    return _req("POST", "/portfolio/orders/batched", {"orders": orders})

def batch_cancel(order_ids: list[str]):
    return _req("POST", "/portfolio/orders/batched/cancel", {"order_ids": order_ids})

def get_fills(since: int | None = None):
    params = {"since": since} if since else {}
    # documented path is /portfolio/fills
    return _req("GET", "/portfolio/fills", params)


# --- Market data endpoints (available in DRY for real quotes) ---
def get_market(ticker: str):
    """Fetch market metadata by ticker.

    Returns Kalshi market schema. In DRY mode, performs a real GET if API base/key
    are configured; otherwise returns a minimal synthetic open market payload.
    """
    # Require real API response; propagate errors to caller
    return _req("GET", f"/markets/{ticker}")


def get_market_order_book(ticker: str):
    """Fetch market order book by ticker and normalize to internal structure.

    Normalized shape: {"orderbook": {"yes": [[price, size], ...], "no": [[price, size], ...]}}
    """
    data = _req("GET", f"/markets/{ticker}/orderbook") or {}

    # Attempt to detect common layouts and normalize
    if isinstance(data, dict) and "orderbook" in data and isinstance(data["orderbook"], dict):
        ob = data["orderbook"]
    elif isinstance(data, dict) and all(k in data for k in ("yes", "no")):
        ob = {"yes": data.get("yes") or [], "no": data.get("no") or []}
    else:
        bids = data.get("bids") if isinstance(data, dict) else None
        yes = []
        no = []
        if isinstance(bids, list):
            for b in bids:
                try:
                    side = (b.get("side") or "").lower()
                    px = int(b.get("price") or 0)
                    sz = int(b.get("size") or b.get("quantity") or 0)
                    if side == "yes":
                        yes.append([px, sz])
                    elif side == "no":
                        no.append([px, sz])
                except Exception:
                    continue
        ob = {"yes": yes, "no": no}

    def _norm(levels):
        out = []
        for lv in levels or []:
            try:
                if isinstance(lv, (list, tuple)) and len(lv) >= 2:
                    out.append([int(lv[0]), int(lv[1])])
                elif isinstance(lv, dict):
                    out.append([int(lv.get("price", 0)), int(lv.get("size") or lv.get("quantity") or 0)])
            except Exception:
                continue
        return out

    return {"orderbook": {"yes": _norm(ob.get("yes")), "no": _norm(ob.get("no"))}}
