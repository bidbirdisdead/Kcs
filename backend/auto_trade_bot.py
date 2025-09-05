"""Simple market scanner and trade helper."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional


from api.kalshi import get_markets, get_orderbook
from trading.execution import submit_entry_buy
from position_manager import get_position_for_market, check_exposure_limit
from trade_logic import is_trade_opportunity, score_opportunity


def scan_and_trade(
    *,
    min_spread: int = 2,
    min_liquidity: int = 10,
    max_price: int = 60,
    dry_run: bool = True,
    max_trades: int = 3,
    sleep_time: float = 1.0,
    return_results: bool = False,
    stop_check: Optional[Callable[[], bool]] = None,
    max_exposure: float = 1000.0,  # USD
) -> Optional[List[Dict[str, Any]]]:
    """Scan open markets and optionally place trades.

    Parameters
    ----------
    min_spread: int
        Minimum cents between best yes bid and implied yes ask.
    min_liquidity: int
        Minimum quantity required at the best bid.
    max_price: int
        Maximum yes price willing to pay.
    dry_run: bool
        When ``True`` no orders are submitted.
    max_trades: int
        Maximum number of trades to place.
    sleep_time: float
        Delay between market checks in seconds.
    return_results: bool
        When ``True`` return a list of decisions/trades.
    stop_check: Callable
        Optional function to signal early termination.
    """

    results: List[Dict[str, Any]] = []
    try:
        markets = get_markets(status="open", limit=100)
    except Exception as exc:  # pragma: no cover - network errors
        logging.error("Failed to fetch markets: %s", exc)
        if return_results:
            return [{"error": f"Failed to fetch markets: {exc}"}]
        return None

    trades_placed = 0
    for market in markets.get("markets", []):
        if stop_check and stop_check():
            logging.info("Auto trade bot stopped by user request.")
            break

        try:
            ticker = market.get("ticker")
            title = market.get("title", "")
            yes_sub_title = market.get("yes_sub_title", "")
            # Position check: skip if already have open position in this market
            if get_position_for_market(ticker):
                continue
            # Exposure check: skip if over limit
            if not check_exposure_limit(max_exposure):
                logging.info("Exposure limit reached, skipping further trades.")
                break
            try:
                ob = get_orderbook(ticker)
            except Exception as exc:  # pragma: no cover - network errors
                logging.error("Orderbook fetch failed for %s: %s", ticker, exc)
                continue

            orderbook = ob.get("orderbook") if ob else None
            if not orderbook:
                continue
            yes_bids = orderbook.get("yes") or []
            no_bids = orderbook.get("no") or []
            best_yes = yes_bids[0]["price"] if yes_bids and isinstance(yes_bids[0], dict) else None
            best_no = no_bids[0]["price"] if no_bids and isinstance(no_bids[0], dict) else None
            yes_liq = yes_bids[0]["quantity"] if yes_bids and isinstance(yes_bids[0], dict) else 0
            true_yes_ask = 100 - best_no if best_no is not None else None

            # Use trade_logic helpers
            if is_trade_opportunity(best_yes, true_yes_ask, yes_liq, min_spread, min_liquidity, max_price):
                score = score_opportunity(best_yes, true_yes_ask, yes_liq, min_spread, min_liquidity, max_price)
                logging.info(
                    "Opportunity: %s | %s | %s | Best YES: %s | True YES Ask: %s | Spread: %s | YES Qty: %s | Score: %.2f",
                    ticker,
                    title,
                    yes_sub_title,
                    best_yes,
                    true_yes_ask,
                    true_yes_ask - best_yes if (true_yes_ask is not None and best_yes is not None) else None,
                    yes_liq,
                    score,
                )
                trade_result: Optional[Dict[str, Any]] = None

                if not dry_run and trades_placed < max_trades and best_yes is not None:
                    try:
                        res = submit_entry_buy(ticker=ticker, side="YES", price_cents=int(best_yes), qty=1, post_only=True)
                        # unify return type: expect submit_entry_buy to return a dict-like result
                        trade_result = res if res is not None else None
                        logging.info("Trade placed: %s", trade_result)
                        trades_placed += 1
                    except Exception as exc:  # pragma: no cover - network errors
                        logging.error("Trade failed for %s: %s", ticker, exc)
                        trade_result = {"error": str(exc)}
                else:
                    logging.info("[DRY RUN] Would place order: %s YES 1 @ %s", ticker, best_yes)

                if return_results:
                                        results.append(
                        {
                            "ticker": ticker,
                            "title": title,
                            "yes_sub_title": yes_sub_title,
                            "best_yes": best_yes,
                            "true_yes_ask": true_yes_ask,
                            "spread": (
                                true_yes_ask - best_yes
                                if (true_yes_ask is not None and best_yes is not None)
                                else None
                            ),
                            "yes_liq": yes_liq,
                            "dry_run": dry_run,
                            "score": score,
                            "trade_result": trade_result,
                        }
                    )
            time.sleep(sleep_time)
        except Exception as exc:  # pragma: no cover - defensive catch
            logging.error("Error in scan loop for %s: %s", market.get("ticker", "?"), exc)

    logging.info("Scan complete. Trades placed: %s", trades_placed)
    if return_results:
        return results
    return None

