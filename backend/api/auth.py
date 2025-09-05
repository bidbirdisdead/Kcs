"""Canonical Kalshi auth helpers.

Expose get_bearer_token() and get_auth_headers(allow_empty=False).

This module centralizes how bearer tokens and headers are built so callers
don't duplicate logic across REST and WS helpers. (PLANNED.TXT change C1)
"""
from __future__ import annotations
import os
from typing import Dict


def get_bearer_token() -> str | None:
    """Return the configured bearer token or None if missing.

Reads environment variable KALSHI_API_KEY.
"""
    t = os.getenv("KALSHI_API_KEY")
    return t if t else None


def get_auth_headers(allow_empty: bool = False) -> Dict[str, str]:
    """Return a headers dict containing Authorization when available.

    If allow_empty is False and no token is present, raises ValueError. Callers
    that can operate unauthenticated (read-only in DRY mode) can set
    allow_empty=True to receive an empty dict.
    """
    token = get_bearer_token()
    if not token:
        if allow_empty:
            return {}
        raise ValueError("KALSHI_API_KEY not set; cannot build Authorization header")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
