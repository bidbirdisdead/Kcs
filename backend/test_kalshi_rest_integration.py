import os
import json
import types

import pytest


def test_get_market_order_book_uses_live_when_available(monkeypatch):
    # Ensure DRY mode (GETs allowed)
    os.environ["TRADING_MODE"] = "DRY"

    # Lazy import inside test to pick up env
    import importlib
    kalshi_rest = importlib.import_module("backend.kalshi_rest")

    # Stub requests.request to return a normalized payload
    class _Resp:
        def __init__(self, payload, status=200):
            self._payload = payload
            self.status_code = status
            self.text = json.dumps(payload)

        def raise_for_status(self):
            if self.status_code and int(self.status_code) >= 400:
                raise Exception(f"HTTP {self.status_code}")

        def json(self):
            return self._payload

    def _fake_request(method, url, headers=None, data=None, timeout=10):
        # Shape similar to expected live response
        payload = {
            "orderbook": {
                "yes": [{"price": 41, "size": 3}, {"price": 40, "size": 1}],
                "no": [{"price": 59, "size": 2}],
            }
        }
        return _Resp(payload)

    monkeypatch.setattr(kalshi_rest.requests, "request", _fake_request)

    ob = kalshi_rest.get_market_order_book("TEST-TICKER")
    assert isinstance(ob, dict)
    assert "orderbook" in ob
    assert list(ob["orderbook"].keys()) == ["yes", "no"]
    # Ensure normalization to [price, size] pairs with ints
    assert all(isinstance(x, list) and len(x) == 2 for x in ob["orderbook"]["yes"]) \
        and all(isinstance(x, int) for x in ob["orderbook"]["yes"][0])

