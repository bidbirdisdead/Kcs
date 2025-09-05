"""Compatibility shim for legacy kalshi_api module used by tests.

Provides `_strike_to_ticker` that selects a market ticker from a provided
`get_markets()` payload. Tests typically monkeypatch `get_markets`.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def get_markets(**kwargs) -> Dict[str, Any]:  # pragma: no cover - test monkeypatches this
    """Placeholder that returns an empty markets list.

    Tests monkeypatch this function to inject fixture data.
    """
    return {"markets": []}


_NUM_RE = re.compile(r"\d{1,9}(?:[\.,]\d+)?")


def _extract_level(s: str) -> Optional[float]:
    """Extract numeric level from a market title/subtitle string."""
    if not s:
        return None
    m = _NUM_RE.search(s.replace(",", ""))
    return float(m.group()) if m else None


def _strike_to_ticker(symbol: str, strike: float, relation: str = ">=") -> Optional[str]:
    """Pick a market ticker matching the desired strike and relation.

    For relation ">=": pick the candidate with the largest level <= strike (floor).
    For relation "<=": pick the candidate with the smallest level >= strike+0.01 (ceil),
    mirroring the behavior expected by tests.
    """
    markets = (get_markets() or {}).get("markets", [])
    if not markets:
        return None

    # Build list of (level, ticker, title)
    cand: List[tuple[float, str]] = []
    for m in markets:
        t = (m.get("title") or "") + " " + (m.get("subtitle") or "")
        lvl = _extract_level(t)
        if lvl is None:
            continue
        tick = m.get("ticker")
        if not tick:
            continue
        cand.append((lvl, tick))

    if not cand:
        return None

    if relation == ">=":
        # floor to the closest not exceeding strike
        eligible = [(lvl, tick) for (lvl, tick) in cand if lvl <= float(strike) + 1e-6]
        if not eligible:
            # fallback: choose the smallest available
            lvl, tick = min(cand, key=lambda x: x[0])
            return tick
        lvl, tick = max(eligible, key=lambda x: x[0])
        return tick
    else:  # "<="
        target = float(strike) + 0.01
        eligible = [(lvl, tick) for (lvl, tick) in cand if lvl >= target - 1e-6]
        if not eligible:
            # fallback: choose the largest available
            lvl, tick = max(cand, key=lambda x: x[0])
            return tick
        lvl, tick = min(eligible, key=lambda x: x[0])
        return tick


__all__ = ["_strike_to_ticker", "get_markets"]

