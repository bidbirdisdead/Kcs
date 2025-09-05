"""Utilities for interacting with the Finazon price API."""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Optional, Union

import requests
from dotenv import load_dotenv

from ..data.database import create_crypto_prices_table, open_conn

# Always load .env from project root or ``trb/.env``
load_dotenv(
    dotenv_path=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "backend\.env")
    )
)

def fetch_and_log_prices(symbols: List[str]) -> Dict[str, Union[float, str, None]]:
    """Fetch latest prices for ``symbols`` and store them in the database."""

    api_url = "https://api.finazon.io/latest/binance/binance/time_series"
    api_key = os.getenv("FINAZON_API_KEY") or os.getenv("FINAZON_KEY")
    if not api_key:
        return {symbol: "Error: FINAZON_API_KEY not set in .env" for symbol in symbols}

    headers = {"Authorization": f"apikey {api_key}"}
    conn = open_conn()
    create_crypto_prices_table(conn)
    out: Dict[str, Union[float, str, None]] = {}

    for symbol in symbols:
        params = {"ticker": symbol, "interval": "1m", "page": "0", "page_size": "1"}
        try:
            resp = requests.get(api_url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                out[symbol] = None
                continue
            d = data[0]
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO crypto_prices
                (symbol, timestamp, open_price, high_price, low_price, close_price, volume, trades,
                 quote_volume, taker_base_volume, taker_quote_volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    d.get("t"),
                    d.get("o"),
                    d.get("h"),
                    d.get("l"),
                    d.get("c"),
                    d.get("bv"),
                    d.get("tr"),
                    d.get("qv"),
                    d.get("tbv"),
                    d.get("tqv"),
                    "FINAZON",
                ),
            )
            conn.commit()
            out[symbol] = d.get("c")
        except Exception as exc:  # pragma: no cover - network errors
            out[symbol] = f"Error: {exc}"

    conn.close()
    return out


def backfill_prices(symbols: List[str], max_pages: int = 1000, page_size: int = 1000) -> None:
    """Backfill historical 1m bars for each symbol using the Finazon API."""

    api_url = "https://api.finazon.io/latest/binance/binance/time_series"
    api_key = os.getenv("FINAZON_API_KEY") or os.getenv("FINAZON_KEY")
    if not api_key:
        raise RuntimeError("FINAZON_API_KEY not set in .env")

    headers = {"Authorization": f"apikey {api_key}"}
    conn = open_conn()
    create_crypto_prices_table(conn)

    for symbol in symbols:
        for page in range(max_pages):
            params = {
                "ticker": symbol,
                "interval": "1m",
                "page": str(page),
                "page_size": str(page_size),
            }
            try:
                resp = requests.get(api_url, params=params, headers=headers, timeout=30)
                if resp.status_code != 200:
                    print(f"Backfill failed for {symbol} page {page}: {resp.text}")
                    break
                data = resp.json().get("data", [])
                if not data:
                    print(f"No data for {symbol} page {page}, ending.")
                    break
                cur = conn.cursor()
                cur.executemany(
                    """
                    INSERT OR IGNORE INTO crypto_prices
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume, trades,
                     quote_volume, taker_base_volume, taker_quote_volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            symbol,
                            d.get("t"),
                            d.get("o"),
                            d.get("h"),
                            d.get("l"),
                            d.get("c"),
                            d.get("bv"),
                            d.get("tr"),
                            d.get("qv"),
                            d.get("tbv"),
                            d.get("tqv"),
                            "FINAZON",
                        )
                        for d in data
                    ],
                )
                conn.commit()
                print(f"Inserted {len(data)} rows for {symbol} page {page}")
                if len(data) < page_size:
                    print(f"Last page for {symbol} (page {page})")
                    break
            except Exception as exc:  # pragma: no cover - network errors
                print(f"Error during backfill for {symbol} page {page}: {exc}")
                break

    conn.close()

