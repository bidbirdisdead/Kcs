import importlib

import backend.kalshi_api as ka


def test_strike_to_ticker_above(monkeypatch):
    mock_markets = [
        {"ticker": "KXBTCD-25AUG2817-T112999.99", "title": "112999.99 or above", "subtitle": "", "close_time": "2025-08-28T17:00:00Z"},
        {"ticker": "KXBTCD-25AUG2817-T113500.00", "title": "113500 or above", "subtitle": ""},
        {"ticker": "KXBTCD-25AUG2817-T112000.00", "title": "112000 or below", "subtitle": ""},
    ]

    monkeypatch.setattr(ka, "get_markets", lambda **kwargs: {"markets": mock_markets})
    res = ka._strike_to_ticker("BTC/USD", 113000, ">=")
    assert res == "KXBTCD-25AUG2817-T112999.99"


def test_strike_to_ticker_below(monkeypatch):
    mock_markets = [
        {"ticker": "KXBTCD-25AUG2817-T112000.00", "title": "112000 or below", "subtitle": ""},
        {"ticker": "KXBTCD-25AUG2817-T112500.00", "title": "112500 or below", "subtitle": ""},
    ]

    monkeypatch.setattr(ka, "get_markets", lambda **kwargs: {"markets": mock_markets})
    res = ka._strike_to_ticker("BTC/USD", 112250, "<=")
    # want+0.01 = 112250.01 -> smallest T >= that is 112500.00
    assert res == "KXBTCD-25AUG2817-T112500.00"
