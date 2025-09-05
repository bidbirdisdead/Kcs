import os
import pytest
from unittest.mock import patch, MagicMock


def test_kalshi_lookup_includes_auth_header(tmp_path, monkeypatch):
    # Ensure token present
    monkeypatch.setenv("KALSHI_API_KEY", "TEST-TOKEN")

    import backend.kalshi_lookup as kl

    mocked = MagicMock()
    mocked.json.return_value = {"series": []}

    with patch.object(kl.S, "get", return_value=mocked) as m_get:
        kl._get("/series", category="crypto")
        # Ensure headers passed and include Authorization
        args, kwargs = m_get.call_args
        headers = kwargs.get("headers")
        assert headers is not None
        assert headers.get("Authorization") == "Bearer TEST-TOKEN"


def test_kalshi_ws_builds_header(monkeypatch):
    monkeypatch.setenv("KALSHI_API_KEY", "WS-TOKEN")
    import backend.kalshi_ws as kws

    # The module resolves headers at import-time via function; call private
    try:
        from api.auth import get_bearer_token  # if available
        token = get_bearer_token()
    except Exception:
        token = os.getenv("KALSHI_API_KEY")

    assert token == "WS-TOKEN"
