import os
import sys
import time
import argparse
from datetime import datetime, timezone

# Ensure package imports work when run as a script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)  # backend/
PROJECT_ROOT = os.path.dirname(PKG_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

from backend.config import SYMBOLS, RATE_LIMIT_DELAY, REST_PAGE_SIZE, HOURS_BACK
from backend.data.ingestion import fetch_and_insert_range_page
from backend.data.database import rest_upsert_bar
import requests
import random
from backend.data.database import open_conn
import sqlite3


def _fmt_ts(ts: int) -> str:
    try:
        return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def _min_db_ts(symbol: str) -> int | None:
    conn = open_conn(read_only=True)
    cur = conn.cursor()
    cur.execute("SELECT MIN(timestamp) FROM crypto_prices WHERE symbol=?", (symbol,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    v = row[0]
    return int(v) if v is not None else None


def backfill_recent(symbol: str, hours_back: int, sleep_sec: float) -> None:
    """Fill the last `hours_back` hours for symbol.

    Requests the full time range in pages, inserting all returned rows. Existing
    rows are ignored by the DB's INSERT OR IGNORE logic.
    """
    now_ts = int(time.time()) // 60 * 60
    start_ts = now_ts - int(hours_back) * 3600
    if start_ts < 0:
        start_ts = 0

    print(f"[BACKFILL_FAST] {symbol}: recent window {_fmt_ts(start_ts)} → {_fmt_ts(now_ts)}")
    _report_completeness(symbol, start_ts, now_ts, header="[REPORT][BEFORE]")
    # Prefer surgical fill: target only minutes that are missing REST rows or
    # have incomplete REST fields, then fetch contiguous runs.
    missing_ts = _find_missing_and_incomplete_rest(symbol, start_ts, now_ts)
    if not missing_ts:
        print(f"[BACKFILL_FAST] {symbol}: nothing missing in the last {hours_back}h")
        return

    runs = _group_runs(missing_ts)
    total_rows = 0
    for (rs, re, nmin) in runs:
        print(f"[BACKFILL_FAST] {symbol}: filling run {_fmt_ts(rs)} → {_fmt_ts(re)} ({nmin}m)")
        inserted = _incremental_fill(symbol, rs, re, sleep_sec)
        total_rows += inserted
    print(f"[BACKFILL_FAST] {symbol}: inserted {total_rows} rows across {len(runs)} runs")
    _report_completeness(symbol, start_ts, now_ts, header="[REPORT][AFTER]")


def backfill_older(symbol: str, chunk_hours: int, sleep_sec: float, page_size: int) -> None:
    """Walk backward in time and fill older history for symbol.

    This respects the provided sleep_sec between requests to honor rate limits.
    """
    now_ts = int(time.time()) // 60 * 60
    min_ts = _min_db_ts(symbol)
    end_ts = (min_ts - 60) if min_ts else now_ts
    if end_ts <= 0:
        print(f"[BACKFILL_FAST] {symbol}: no valid end_ts to start from")
        return

    total_rows = 0
    windows = 0

    while end_ts > 0:
        start_ts = end_ts - chunk_hours * 3600 + 60
        if start_ts < 0:
            start_ts = 0

        print(f"[BACKFILL_FAST] {symbol}: window {_fmt_ts(start_ts)} → {_fmt_ts(end_ts)}")
        page = 0
        inserted_this_window = 0
        zero_pages = 0

        while True:
            n, has_more = fetch_and_insert_range_page(symbol, start_ts, end_ts, page=page)
            n = int(n)
            inserted_this_window += n
            total_rows += n
            page += 1
            if n == 0:
                zero_pages += 1
            else:
                zero_pages = 0
            if zero_pages >= 2 or not has_more:
                break
            # Respect rate limits
            time.sleep(sleep_sec)

        print(
            f"[BACKFILL_FAST] {symbol}: +{inserted_this_window} rows in window; total={total_rows}"
        )
        windows += 1

        # Stop if API returns no data for this window
        if inserted_this_window == 0:
            print(f"[BACKFILL_FAST] {symbol}: no data returned; stopping")
            break

        # Move the window older by one minute (avoid overlapping the previous start)
        end_ts = start_ts - 60


def _find_missing_and_incomplete_rest(symbol: str, start_ts: int, end_ts: int) -> list[int]:
    """Return sorted minute timestamps in [start_ts,end_ts] that have no REST row
    or have a REST row with missing core fields.

    Core fields: o,h,l,c,bv,tr,qv,tbv,tqv mapped to DB columns.
    """
    conn = open_conn(read_only=True)
    cur = conn.cursor()

    # Present minutes with a REST row
    cur.execute(
        """
        SELECT DISTINCT timestamp FROM crypto_prices
        WHERE symbol=? AND source='FINAZON_REST' AND timestamp BETWEEN ? AND ?
        """,
        (symbol, int(start_ts), int(end_ts)),
    )
    present_rest = {int(r[0]) for r in cur.fetchall()}

    # Minutes with incomplete REST rows (any important column is NULL)
    cur.execute(
        """
        SELECT timestamp FROM crypto_prices
        WHERE symbol=? AND source='FINAZON_REST' AND timestamp BETWEEN ? AND ?
          AND (
            open_price IS NULL OR high_price IS NULL OR low_price IS NULL OR close_price IS NULL OR
            volume IS NULL OR trades IS NULL OR quote_volume IS NULL OR taker_base_volume IS NULL OR taker_quote_volume IS NULL
          )
        """,
        (symbol, int(start_ts), int(end_ts)),
    )
    incomplete_rest = {int(r[0]) for r in cur.fetchall()}

    conn.close()

    # Build expected minute grid
    start_aligned = (start_ts // 60) * 60
    end_aligned = (end_ts // 60) * 60
    expected = set(range(start_aligned, end_aligned + 1, 60))

    # Missing: expected minutes without a REST row
    missing_without_rest = expected - present_rest
    # Also re-fetch minutes with incomplete REST data
    needed = set(sorted(missing_without_rest | incomplete_rest))
    return sorted(needed)


def _group_runs(ts_list: list[int], limit: int = 1000) -> list[tuple[int, int, int]]:
    """Group sorted minute timestamps into contiguous runs.

    Returns list of (start_ts, end_ts, n_minutes). Optionally splits runs larger
    than `limit` minutes into multiple chunks to keep request sizes reasonable.
    """
    if not ts_list:
        return []
    ts_list = sorted(ts_list)
    runs: list[tuple[int, int, int]] = []
    start = prev = ts_list[0]
    count = 1
    for ts in ts_list[1:]:
        if ts == prev + 60 and count < limit:
            prev = ts
            count += 1
            continue
        runs.append((start, prev, (prev - start) // 60 + 1))
        start = prev = ts
        count = 1
    runs.append((start, prev, (prev - start) // 60 + 1))
    return runs


def _report_completeness(symbol: str, start_ts: int, end_ts: int, header: str = "[REPORT]") -> None:
    """Print a DB completeness report for REST rows in the given window.

    Shows expected minute rows, present REST rows, missing minutes, incomplete
    REST rows, and up to the top 3 longest missing runs.
    """
    conn = open_conn(read_only=True)
    cur = conn.cursor()
    start_aligned = (start_ts // 60) * 60
    end_aligned = (end_ts // 60) * 60
    expected_count = (end_aligned - start_aligned) // 60 + 1 if end_aligned >= start_aligned else 0

    # Present distinct REST minutes
    cur.execute(
        """
        SELECT COUNT(DISTINCT timestamp) FROM crypto_prices
        WHERE symbol=? AND source='FINAZON_REST' AND timestamp BETWEEN ? AND ?
        """,
        (symbol, int(start_aligned), int(end_aligned)),
    )
    present_rest = int(cur.fetchone()[0] or 0)

    # Incomplete REST minutes
    cur.execute(
        """
        SELECT COUNT(1) FROM crypto_prices
        WHERE symbol=? AND source='FINAZON_REST' AND timestamp BETWEEN ? AND ?
          AND (
            open_price IS NULL OR high_price IS NULL OR low_price IS NULL OR close_price IS NULL OR
            volume IS NULL OR trades IS NULL OR quote_volume IS NULL OR taker_base_volume IS NULL OR taker_quote_volume IS NULL
          )
        """,
        (symbol, int(start_aligned), int(end_aligned)),
    )
    incomplete_rest = int(cur.fetchone()[0] or 0)

    # Build missing list for run summary
    cur.execute(
        """
        SELECT DISTINCT timestamp FROM crypto_prices
        WHERE symbol=? AND source='FINAZON_REST' AND timestamp BETWEEN ? AND ?
        """,
        (symbol, int(start_aligned), int(end_aligned)),
    )
    present_set = {int(r[0]) for r in cur.fetchall()}
    conn.close()
    expected_set = set(range(start_aligned, end_aligned + 1, 60)) if expected_count > 0 else set()
    missing_set = sorted(expected_set - present_set)
    missing_count = len(missing_set)
    runs = _group_runs(missing_set)
    runs_sorted = sorted(runs, key=lambda x: x[2], reverse=True)
    top = runs_sorted[:3]

    pct = (present_rest / expected_count * 100.0) if expected_count else 100.0
    print(
        f"{header} {symbol}: window={_fmt_ts(start_aligned)}→{_fmt_ts(end_aligned)} expected={expected_count} present_rest={present_rest} ({pct:.1f}%) missing={missing_count} incomplete_rest={incomplete_rest}"
    )
    if top:
        lines = ", ".join(f"{_fmt_ts(a)}→{_fmt_ts(b)} ({n}m)" for a, b, n in top)
        print(f"{header} {symbol}: top_missing_runs: {lines}")


def _incremental_fill(symbol: str, start_ts: int, end_ts: int, sleep_sec: float, page_size: int = 1000) -> int:
    """Fill a range by advancing a timestamp cursor rather than paging.

    Fetches up to page_size bars per request using start_time and end_time, then
    advances the cursor to the max returned timestamp + 60s. Stops when cursor
    passes end_ts or when two consecutive empty results are observed.
    """
    api_url = "https://api.finazon.io/latest/binance/binance/time_series"
    api_key = os.getenv("FINAZON_API_KEY") or os.getenv("FINAZON_KEY")
    if not api_key:
        print("[BACKFILL_FAST] FINAZON_API_KEY not set; aborting incremental fill")
        return 0

    headers = {"Authorization": f"apikey {api_key}"}
    cursor = int(start_ts)
    inserted = 0
    empty_streak = 0

    while cursor <= end_ts:
        params = {
            "ticker": symbol,
            "interval": "1m",
            # Finazon expects start_at / end_at (UNIX seconds)
            "start_at": cursor,
            "end_at": end_ts,
            "order": "asc",
            "page": "0",
            "page_size": str(page_size),
        }
        data = _rate_limited_get_json(api_url, params, headers, sleep_sec, symbol, cursor)
        if data is None:
            # give up on this run after repeated 429/5xx
            print(f"[BACKFILL_FAST] {symbol}: giving up on range at {_fmt_ts(cursor)} due to repeated errors")
            break

        if not data:
            empty_streak += 1
            if empty_streak >= 2:
                print(f"[BACKFILL_FAST] {symbol}: two empty pages at {_fmt_ts(cursor)}; stopping range")
                break
            # advance conservatively to avoid tight loop
            cursor += 60 * page_size
            continue

        # Insert rows and advance cursor
        max_t = cursor
        for d in data:
            try:
                if d.get("t") is not None:
                    max_t = max(max_t, int(d.get("t")))
                if rest_upsert_bar(symbol, d):
                    inserted += 1
            except Exception:
                continue

        print(f"[BACKFILL_FAST] {symbol}: cursor batch rows={len(data)} next_cursor={_fmt_ts(max_t+60)}")
        cursor = max_t + 60
        empty_streak = 0
        time.sleep(sleep_sec)

    return inserted


_last_req_ts = 0.0


def _rate_limited_get_json(url: str, params: dict, headers: dict, base_sleep: float, symbol: str, cursor: int,
                           max_retries: int = 8) -> list | None:
    """Perform a GET honoring a 5 rpm limit and backing off on 429/5xx.

    - Enforces at least `base_sleep` seconds between requests (process-wide).
    - On 429, uses Retry-After if present; otherwise exponential backoff with jitter.
    - Returns data list (resp.json()['data']) or None on persistent failure.
    """
    global _last_req_ts
    attempt = 0
    sleep_min = max(0.0, float(base_sleep))
    while attempt <= max_retries:
        # Throttle baseline rate across the process
        now = time.monotonic()
        wait = max(0.0, (_last_req_ts + sleep_min) - now)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            _last_req_ts = time.monotonic()
            if resp.status_code == 429:
                # Too many requests — respect Retry-After
                ra = resp.headers.get('Retry-After')
                retry_after = float(ra) if ra and str(ra).isdigit() else sleep_min * (2 ** attempt)
                retry_after = min(retry_after, 60.0)
                jitter = random.uniform(0, 0.5 * sleep_min)
                delay = max(sleep_min, retry_after) + jitter
                print(f"[BACKFILL_FAST] rate limited (429), sleeping {delay:.1f}s; symbol={symbol} cursor={_fmt_ts(cursor)}")
                time.sleep(delay)
                attempt += 1
                continue
            if 500 <= resp.status_code < 600:
                delay = min(sleep_min * (2 ** attempt), 60.0) + random.uniform(0, 0.5 * sleep_min)
                print(f"[BACKFILL_FAST] server {resp.status_code}, sleeping {delay:.1f}s; symbol={symbol} cursor={_fmt_ts(cursor)}")
                time.sleep(delay)
                attempt += 1
                continue
            resp.raise_for_status()
            js = resp.json() or {}
            return js.get('data') or []
        except requests.RequestException as exc:
            delay = min(sleep_min * (2 ** attempt), 60.0) + random.uniform(0, 0.5 * sleep_min)
            print(f"[BACKFILL_FAST] HTTP error: {exc}; sleeping {delay:.1f}s; symbol={symbol} cursor={_fmt_ts(cursor)}")
            time.sleep(delay)
            attempt += 1
        except Exception as exc:
            print(f"[BACKFILL_FAST] error: {exc}; symbol={symbol} cursor={_fmt_ts(cursor)}")
            time.sleep(sleep_min)
            attempt += 1
    return None


def main():
    if load_dotenv:
        # Try to load a .env from project or backend dir for convenience
        for p in (os.path.join(PROJECT_ROOT, ".env"), os.path.join(PKG_ROOT, ".env")):
            if os.path.exists(p):
                load_dotenv(p)
                break

    parser = argparse.ArgumentParser(
        description="Maintenance backfill for backend/cryptoprice.db (Finazon 1m bars)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated symbols (default: config.SYMBOLS)",
        default=",".join(SYMBOLS),
    )
    parser.add_argument(
        "--hours-back",
        type=int,
        default=int(HOURS_BACK),
        help="Backfill this many hours from now (fast full-range paging)",
    )
    parser.add_argument(
        "--chunk-hours",
        type=int,
        default=24 * 7,
        help="Optional: when using --mode older, hours per historical window",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=float(RATE_LIMIT_DELAY),
        help="Sleep seconds between HTTP requests to respect rate limits",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=int(REST_PAGE_SIZE),
        help="Page size for API calls (ingestion module sets this globally)",
    )
    parser.add_argument(
        "--mode",
        choices=["recent", "older"],
        default="recent",
        help="recent: fill last --hours-back hours. older: walk backwards in --chunk-hours windows.",
    )
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"[BACKFILL_FAST] Starting maintenance backfill for {symbols} | mode={args.mode} hours_back={args.hours_back} chunk={args.chunk_hours}h sleep={args.sleep}s")

    for sym in symbols:
        try:
            if args.mode == "recent":
                backfill_recent(sym, hours_back=args.hours_back, sleep_sec=args.sleep)
            else:
                backfill_older(sym, chunk_hours=args.chunk_hours, sleep_sec=args.sleep, page_size=args.page_size)
        except KeyboardInterrupt:
            print("Interrupted by user")
            return
        except Exception as e:
            print(f"[BACKFILL_FAST] {sym}: error: {e}")


if __name__ == "__main__":
    main()
