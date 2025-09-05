import pytest
from api.kalshi import normalize_time_in_force


@pytest.mark.parametrize("inp,expected", [
    ("GTC", None),
    ("gtc", None),
    ("GTT", None),
    ("IOC", "immediate_or_cancel"),
    ("ioc", "immediate_or_cancel"),
    ("FOK", "fill_or_kill"),
    ("fill_or_kill", "fill_or_kill"),
    ("immediate_or_cancel", "immediate_or_cancel"),
])
def test_normalize_tif(inp, expected):
    assert normalize_time_in_force(inp) == expected
