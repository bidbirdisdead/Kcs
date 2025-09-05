import argparse
import time
import datetime as dt



# Reuse the core pipeline from finfill
import finfill as ff


def _distinct_minute_ts(symbol: str, start_ts: int, end_ts: int) -> list[int]:
    conn = ff.open_conn(read_only=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT timestamp
        FROM crypto_prices
        WHERE symbol=? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
        """,
        (symbol, int(start_ts), int(end_ts)),
    )
    rows = cur.fetchall()
    conn.close()
    return [int(r[0]) for r in rows]


def main():
    ap = argparse.ArgumentParser(description="Offline backtest: minute-by-minute replay + calibration")
        ap.add_argument(
        "--symbol", action="append", help="Symbol(s) to backtest (e.g. BTC/USDT)"
    )
    ap.add_argument(
        "--months",
        type=int,
        default=3,
        help="Months lookback (approx 30d per month)",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="Max minutes to process (debug)"
    )
    args = ap.parse_args()

    symbols = args.symbol or ff.SYMBOLS
    months = max(1, int(args.months))
    end_ts = ff.last_closed_minute()
    start_ts = end_ts - months * 30 * 24 * 3600

    # Ensure DBs exist and evolve schema
    ff._init_db_pragmas()
    ff.init_modeldb()

    # Disable live trading effects during offline replay
    try:
        ff.ENABLE_TRADING = False
    except Exception:
        pass

    for sym in symbols:
        print(f"[BT] Backtesting {sym} months={months}")
        ts_list = _distinct_minute_ts(sym, start_ts, end_ts)
        if not ts_list:
            print(f"[BT] No data for {sym} in range")
            continue
        if args.limit is not None:
            ts_list = ts_list[: int(args.limit)]

        # Seed model bundle to avoid cold-start per minute; persist LGBM for warm-starts
        try:
            ff._fit_hour_return_model(sym)
        except Exception as e:
            print(f"[BT] initial fit failed for {sym}: {e}")

        n = 0
        t0 = time.time()
        for ts in ts_list:
            # Read the DB close at this minute (REST preferred inside helper)
            row = ff.select_one_bar_for_ts(sym, int(ts))
            close = float(row[3]) if row and row[3] is not None else None
            try:
                ff.predict_next_hour(sym, ws_close=close, ws_ts=int(ts))
                        except Exception:
                # Keep going on errors (data gaps expected)
                pass
            n += 1
            if n % 500 == 0:
                elapsed = time.time() - t0
                print(f"[BT] {sym} processed {n}/{len(ts_list)} minutes in {elapsed:.1f}s")

        # Settle what we can
        settled, missing = ff.settle_due_predictions()
        print(f"[BT] {sym} settle: settled={settled} missing_bars={missing}")

    # Report overall calibration over entire backtest window
    try:
        ff.score_report(hours=months * 30 * 24)
    except Exception:
        print(f"[BT] score report failed: {e}")


if __name__ == "__main__":
    main()

