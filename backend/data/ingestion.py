import os
import json
import time
import threading
import requests
import websocket
import rel
import sqlite3
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from ..config import SYMBOLS, RATE_LIMIT_DELAY, HOURS_BACK, REST_PAGE_SIZE

import datetime
from .database import open_conn, find_missing_1m_timestamps, rest_upsert_bar, ws_upsert_bar

_REST_CONT: Dict[str, Dict[str, int]] = {}
_BACKFILL_LOCK = threading.Lock()
_BACKFILL_STARTED = False
_BF_METRICS = defaultdict(lambda: {
    "initial_missing": None,
    "filled_total": 0,
    "batches": 0,
    "avg_bars_per_req": 0.0,
    "last_report_missing": None,
})

_WS_LAST_WRITTEN_TS: Dict[str, int] = {s: 0 for s in SYMBOLS}
_WS_LAST_LOGGED: Dict[Tuple[str, int], float] = {}
_WS_LOG_INTERVAL = 10.0

def fetch_and_insert_range_page(symbol: str, start_ts: int, end_ts: int, page: int = 0) -> Tuple[int, bool]:
    api_url = "https://api.finazon.io/latest/binance/binance/time_series"
    api_key = os.getenv("FINAZON_API_KEY") or os.getenv("FINAZON_KEY")
    if not api_key:
        print("Error: FINAZON_API_KEY not set in .env")
        return 0, False

    headers = {"Authorization": f"apikey {api_key}"}
    params = {
        "ticker": symbol,
        "interval": "1m",
        # Finazon API uses start_at / end_at (UNIX seconds)
        "start_at": start_ts,
        "end_at": end_ts,
        "order": "asc",  # oldest → newest within the window
        "page": str(page),
        "page_size": str(REST_PAGE_SIZE),
    }

    try:
        resp = requests.get(api_url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json().get("data") or []
        n = 0
        for d in data:
            if rest_upsert_bar(symbol, d):
                n += 1
        has_more = len(data) == REST_PAGE_SIZE
        if n:
            print(f"[REST] Upserted {n} bars for {symbol} p={page} {start_ts}..{end_ts}")
        return n, has_more
    except Exception as exc:
        print(f"[REST] Range error for {symbol} p={page} {start_ts}..{end_ts}: {exc}")
        return 0, False

def start_backfill_worker():
    global _BACKFILL_STARTED
    with _BACKFILL_LOCK:
        if _BACKFILL_STARTED:
            print("[BACKFILL] already running; skipping duplicate start")
            return
        _BACKFILL_STARTED = True
    print("[BACKFILL] worker started (REST @ 5 rpm)")

    symbol_idx = 0
    while True:
        symbol = SYMBOLS[symbol_idx % len(SYMBOLS)]
        symbol_idx += 1

        cont = _REST_CONT.get(symbol)
        if cont:
            miss_before = find_missing_1m_timestamps(symbol, hours=HOURS_BACK)
            before_n = len(miss_before)
            upserted, has_more = fetch_and_insert_range_page(
                symbol, cont["start"], cont["end"], cont["page"]
            )
            miss_after = find_missing_1m_timestamps(symbol, hours=HOURS_BACK)
            after_n = len(miss_after)
            _progress_update(symbol, before_n, after_n, upserted)
            if has_more:
                cont["page"] += 1
                _REST_CONT[symbol] = cont
            else:
                _REST_CONT.pop(symbol, None)
            time.sleep(RATE_LIMIT_DELAY)
            continue

        miss = find_missing_1m_timestamps(symbol, hours=HOURS_BACK)
        if not miss:
            time.sleep(5)
            continue

        _report_missing_summary(symbol, miss)
        run_start = miss[0]
        run_end = run_start
        for ts in miss[1:]:
            if ts == run_end + 60:
                run_end = ts
            else:
                break

        before_n = len(miss)
        upserted, has_more = fetch_and_insert_range_page(symbol, run_start, run_end, page=0)
        miss_after = find_missing_1m_timestamps(symbol, hours=HOURS_BACK)
        after_n = len(miss_after)
        _progress_update(symbol, before_n, after_n, upserted)

        if has_more:
            _REST_CONT[symbol] = {"start": run_start, "end": run_end, "page": 1}

        time.sleep(RATE_LIMIT_DELAY)

def on_ws_open(wsapp):
    sub = {
        "event": "subscribe",
        "dataset": "binance",
        "tickers": SYMBOLS,
        "channel": "bars",
        "frequency": "1s",
        "aggregation": "1m",
    }
    wsapp.send(json.dumps(sub))
    print("[WS] Connection opened & subscribed", SYMBOLS)

def _maybe_log_ws(symbol: str, ts: int, close, force: bool = False):
    now = time.time()
    key = (symbol, ts)
    last = _WS_LAST_LOGGED.get(key, 0)
    if force or (now - last) >= _WS_LOG_INTERVAL:
        print(f"[WS] Upserted bar: {symbol} {ts} close={close}")
        _WS_LAST_LOGGED[key] = now

_WS_LAST_TS = {s: 0 for s in SYMBOLS}

def on_ws_message(wsapp, message):
    try:
        msg = json.loads(message)
        if msg.get("ch") == "bars":
            # Finazon WS sends an envelope with a `data` array of bar objects.
            # Handle both styles: envelope with `data: [...]` and single bar messages.
            if "data" in msg and isinstance(msg["data"], list):
                for bar in msg["data"]:
                    if not (isinstance(bar, dict) and "s" in bar and "t" in bar):
                        continue
                    s, ts = bar["s"], bar["t"]
                    if _WS_LAST_WRITTEN_TS.get(s) != ts:
                        _WS_LAST_WRITTEN_TS[s] = ts
                        try:
                            ws_upsert_bar(bar)
                        except sqlite3.OperationalError as e:
                            if "locked" in str(e).lower():
                                time.sleep(0.05)
                                ws_upsert_bar(bar)
                            else:
                                raise
                    last = _WS_LAST_TS.get(s)
                    if last != ts:
                        _WS_LAST_TS[s] = ts
            elif "s" in msg and "t" in msg:
                s, ts = msg["s"], msg["t"]
                if _WS_LAST_WRITTEN_TS.get(s) != ts:
                    _WS_LAST_WRITTEN_TS[s] = ts
                    try:
                        ws_upsert_bar(msg)
                    except sqlite3.OperationalError as e:
                        if "locked" in str(e).lower():
                            time.sleep(0.05)
                            ws_upsert_bar(msg)
                        else:
                            raise
                last = _WS_LAST_TS.get(s)
                if last != ts:
                    _WS_LAST_TS[s] = ts
            # Prediction queue logic will be handled separately
    except Exception as e:
        print(f"[WS] Error processing message: {e}")

def on_ws_error(wsapp, error):
    print(f"[WS] Error: {error}")

def on_ws_close(wsapp, code, reason):
    print(f"[WS] Connection closed: {code} {reason}")

def _fmt_ts(ts: int) -> str:
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

def _fmt_dur(sec: float) -> str:
    sec = int(max(0, sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"

def _runs_from_missing(missing: list[int], limit: int = 3) -> list[tuple[int,int,int]]:
    if not missing: return []
    runs = []
    start = prev = missing[0]
    for ts in missing[1:]:
        if ts == prev + 60:
            prev = ts
            continue
        runs.append((start, prev, (prev - start)//60 + 1))
        if len(runs) >= limit:
            return runs
        start = prev = ts
    runs.append((start, prev, (prev - start)//60 + 1))
    return runs[:limit]

def _report_missing_summary(symbol: str, missing: list[int], prefix: str = "[BACKFILL]"):
    if not missing:
        print(f"{prefix} {symbol}: no missing minutes (last {HOURS_BACK}h).")
        return
    runs = _runs_from_missing(missing, limit=3)
    runs_txt = ", ".join(f"{_fmt_ts(a)}→{_fmt_ts(b)} ({n}m)" for a,b,n in runs)
    tail = "" if len(runs) < 3 else " …"
    print(f"{prefix} {symbol}: missing={len(missing)}; top runs: {runs_txt}{tail}")

def _progress_update(symbol: str, before_missing: int, after_missing: int, batch_filled: int):
    m = _BF_METRICS[symbol]
    if m["initial_missing"] is None:
        m["initial_missing"] = int(before_missing)
    if m["filled_total"] is None:
        m["filled_total"] = 0
    if m["batches"] is None:
        m["batches"] = 0
    if m["avg_bars_per_req"] is None:
        m["avg_bars_per_req"] = 0.0

    delta = before_missing - after_missing
    batch_filled = max(batch_filled, delta)

    m["filled_total"] += int(batch_filled)
    m["batches"] += 1
    if m["batches"] > 0:
        m["avg_bars_per_req"] = (
            ((m["avg_bars_per_req"] * (m["batches"] - 1)) + batch_filled) / m["batches"]
        )
    else:
        m["avg_bars_per_req"] = float(batch_filled)
    remaining = after_missing
    eta_sec = (remaining / max(1.0, m["avg_bars_per_req"])) * RATE_LIMIT_DELAY

    print(
        f"[BACKFILL] {symbol} progress: {before_missing} → {after_missing} "
        f"(-{before_missing - after_missing}), batch={batch_filled}, "
        f"total_filled={m['filled_total']}, avg/req={m['avg_bars_per_req']:.1f}, "
        f"ETA≈{_fmt_dur(eta_sec)}"
    )


def start_ws_listener():
    api_key = os.getenv("FINAZON_API_KEY") or os.getenv("FINAZON_KEY")
    if not api_key:
        print("[WS] FINAZON_API_KEY not set in .env")
        return
    ws_url = f"wss://ws.finazon.io/v1?apikey={api_key}"
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_ws_open,
        on_message=on_ws_message,
        on_error=on_ws_error,
        on_close=on_ws_close,
    )
    ws.run_forever(dispatcher=rel, reconnect=5, ping_interval=30, ping_timeout=10)
    rel.signal(2, rel.abort)
    rel.dispatch()
