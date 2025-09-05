"""Tiny Kalshi WebSocket client for the user-fills stream.

Provides start_user_fills(on_fill) which connects in a background thread
and calls on_fill(dict) for every incoming fill message.

This uses the websocket-client package (imported as `websocket`).
"""
import os
import json
import threading
import websocket
from typing import Callable
from urllib.parse import urlparse

# Authoritative default: user-fills WS is hosted on the *elections* domain.
# Keep a single, explicit mapping (no REST-host guessing) and allow env override.
_DEFAULT_WS_URL = "wss://api.elections.kalshi.com/ws/user-fills"

def _resolve_ws_url() -> str:
    env_url = os.getenv("KALSHI_WS_USER_FILLS")
    url = env_url.strip() if env_url else _DEFAULT_WS_URL
    # Basic validation with a clear error if misconfigured
    try:
        u = urlparse(url)
        if u.scheme not in ("wss",):
            raise ValueError(f"invalid scheme: {u.scheme!r} (expected 'wss')")
        if not u.netloc:
            raise ValueError("missing host")
        # Guard against the common misroute: api.kalshi.com (CloudFront 404)
        if u.netloc.endswith("api.kalshi.com"):
            raise ValueError("user-fills WS is not served from 'api.kalshi.com'; use 'api.elections.kalshi.com'")
    except Exception as e:
        raise RuntimeError(f"KALSHI_WS_USER_FILLS invalid: {url!r} -> {e}") from e
    return url

KALSHI_WS_USER_FILLS = _resolve_ws_url()

def start_user_fills(on_fill: Callable[[dict], None]):
    """Connects to the user fills stream and invokes on_fill(dict) for each fill.

    Returns the WebSocketApp instance. The connection runs on a daemon thread.
    """
    # Prefer centralized auth helper to ensure consistent header format (PLANNED.TXT C2)
    try:
        from api.auth import get_bearer_token  # type: ignore
        token = get_bearer_token()
    except Exception:
        token = os.getenv("KALSHI_API_KEY")
    headers = [f"Authorization: Bearer {token}"] if token else []

    def _on_message(ws, msg):
        try:
            data = json.loads(msg)
            on_fill(data)
        except Exception as e:
            print("[KALSHI][WS] parse error:", e)

    def _on_error(ws, err):
        # Surface the raw error and provide immediate guidance for the common
        # "Handshake status 404 Not Found" case which typically means the
        # resolved host/path is incorrect (CloudFront 404), not an auth failure.
        try:
            err_s = str(err)
        except Exception:
            err_s = repr(err)
        print("[KALSHI][WS] error:", err_s)
        if "Handshake status 404" in err_s or "404 Not Found" in err_s:
            print("[KALSHI][WS] 404: wrong WS URL/host (CloudFront). Resolved:", KALSHI_WS_USER_FILLS)
            print("[KALSHI][WS] Use the elections WS host: wss://api.elections.kalshi.com/ws/user-fills")

    # websocket-client versions have different on_close signatures; accept both forms
    def _on_close(ws, code=None, reason=None):
        print("[KALSHI][WS] closed:", code, reason)

    ws = websocket.WebSocketApp(
        KALSHI_WS_USER_FILLS,
        header=headers,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )
    # Print the URL we're about to connect to for easier triage when CloudFront
    # returns a 404 during the handshake.
    print(f"[KALSHI][WS] connecting to: {KALSHI_WS_USER_FILLS}")
    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()
    print("[KALSHI][WS] user-fills started")
    return ws
