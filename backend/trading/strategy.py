import time
import math
import threading
import numpy as np
from typing import Dict, List, Optional
import os

from ..config import (
    SYMBOLS,
    TRADING_MODE,
    KALSHI_STEP,
    KALSHI_WINGS,
    RV_BOOST,
    MU_SOFT,
    KALSHI_SIGMA_FLOOR,
    PROB_FLOOR,
    MIN_LIQUIDITY,
    MAX_SPREAD_CENTS,
    DECISION_EDGE_CENTS,
)
from ..data.database import get_latest_price, get_db_path
from ..api.kalshi import get_orderbook, create_order_compat, _strike_to_ticker
from ..kalshi_lookup import find_btc_hourly_above_below, find_eth_hourly_above_below
from ..models.predictor import get_fast_predictor
from ..models.trainer import get_model_trainer
from ..orderbook_utils import yes_best_bid_ask_from_orderbook, clamp_post_only
from ..features.engineering import build_features_vectorized, add_flow_features, add_flow_plus

# In-memory trade state
_OPEN_POSITIONS: Dict[str, Dict] = {}
_COOLDOWNS: Dict[tuple, float] = {}
_KILL: Dict[str, Dict] = {}

# Local strategy knobs (simple defaults)
_TP_CENTS = 8
_SL_BASE_CENTS = 5
_SL_RATCHET_TRIGGER = 4
_SL_RATCHET_CENTS = 3
_COOLDOWN_SEC = 180
_KILL_WINDOW_SEC = 900
_KILL_MAX_STOPS = 3
_KILL_MAX_LOSS_CENTS = 15
_MU_REL_EPS = 2e-4  # require meaningful directional signal (~2 bps)
_MIN_PROB_ANCHOR = 0.50
_MIN_PROB_SAT = 0.50

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, p))
    return math.log(p / (1.0 - p))

def _get_feature_snapshot(symbol: str) -> Dict[str, float]:
    """Fetch a minimal set of engineered features for gating (last row)."""
    snap: Dict[str, float] = {}
    try:
        import sqlite3
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT timestamp, open_price, high_price, low_price, close_price, volume,
                   trades, quote_volume, taker_base_volume, taker_quote_volume
            FROM crypto_prices
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 240
            """,
            (symbol,),
        )
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return snap
        rows.reverse()
        import pandas as pd
        df = pd.DataFrame(
            rows,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trades",
                "quote_volume",
                "taker_base_volume",
                "taker_quote_volume",
            ],
        ).set_index("timestamp")
        df["volume"] = df["volume"].fillna(0)
        base = build_features_vectorized(df["open"], df["high"], df["low"], df["close"], df["volume"])  # includes 'adx'
        # add flow bundles for extended metrics when available
        try:
            flow = add_flow_features(df)
            for col in [
                "trade_count_1m",
                "avg_trade_size_1m",
                "dollar_volume_1m",
                "taker_buy_ratio",
                "taker_imbalance",
                "ema_vol_5",
            ]:
                if col in flow.columns and col not in base.columns:
                    base[col] = flow[col]
        except Exception:
            pass
        try:
            ext = add_flow_plus(df)
            for col in [
                "ADX_14",
                "RV_60m",
                "eff_ratio_10",
                "whipsaw_12",
                "tbi_5",
                "burst",
                "dollar_vol_1m",
                "donchian_pos_60",
                "ema_gap_5_15",
            ]:
                if col in ext.columns and col not in base.columns:
                    base[col] = ext[col]
        except Exception:
            pass
        last = base.iloc[[-1]].to_dict(orient="records")[0]
        # normalize keys we use
        snap["ADX_14"] = float(last.get("ADX_14", last.get("adx", 0.0)))
        snap["RV_60m"] = float(last.get("RV_60m", 0.0))
        snap["eff_ratio_10"] = float(last.get("eff_ratio_10", 0.0))
        snap["whipsaw_12"] = float(last.get("whipsaw_12", 0.0))
        snap["tbi_5"] = float(last.get("tbi_5", 0.0))
        snap["burst"] = float(last.get("burst", 0.0))
        snap["dollar_volume_1m"] = float(last.get("dollar_vol_1m", last.get("dollar_volume_1m", 0.0)))
        snap["donchian_pos_60"] = float(last.get("donchian_pos_60", 0.5))
        snap["ema_gap_5_15"] = float(last.get("ema_gap_5_15", 0.0))
        return snap
    except Exception:
        return snap


def start_trade_worker():
    """Fair value arbitrage worker using ML predictions from the crypto price database"""
    print("[TRADE] worker started (ML-based fair value arbitrage)")
    
    def _trade_worker():
        import time
        
        print("[TRADE] Waiting for ML models to be ready...")
        model_trainer = get_model_trainer()
        models_ready = model_trainer.wait_for_models_ready(SYMBOLS, timeout_seconds=300)
        
        if not models_ready:
            print("[TRADE] ERROR: ML models not ready within timeout, trading worker exiting")
            return
            
        print("[TRADE] ML models ready, starting trading loop")
        
        while True:
            try:
                print("[TRADE] Checking ML-based fair value arbitrage opportunities...")
                opportunities_found = 0
                debug_info = []
                
                for symbol in SYMBOLS:
                    try:
                        current_price = get_latest_price(symbol)
                        if not current_price:
                            debug_info.append(f"{symbol}: No current price available")
                            continue
                            
                        price_now = current_price
                        debug_info.append(f"{symbol}: Current price = {price_now}")
                        
                        try:
                            fast_predictor = get_fast_predictor()
                            prediction = fast_predictor.predict_next_hour(symbol, float(price_now))
                            debug_info.append(f"{symbol}: ML prediction = {prediction}")
                                
                            if prediction is None:
                                debug_info.append(f"{symbol}: ML prediction failed")
                                continue
                                
                            mu = math.log(prediction / float(price_now))
                            dir_txt = "UP" if mu >= 0 else "DOWN"
                            debug_info.append(f"{symbol}: ML-derived mu = {mu:.4f} ({dir_txt})")
                            
                        except Exception as e:
                            debug_info.append(f"{symbol}: ML prediction error: {e}")
                            continue
                        
                        try:
                            # ... (volatility calculation)
                            sigma_resid = 0.02
                            rv_hour = 0.01
                                
                        except Exception as e:
                            sigma_resid = 0.02
                            rv_hour = 0.01
                        
                        sigma_total = math.sqrt(max(sigma_resid, 0.0)**2 + (RV_BOOST * max(rv_hour, 0.0))**2)
                        sigma_total = max(sigma_total, KALSHI_SIGMA_FLOOR)
                        
                        shrink = 1.0 / (1.0 + (abs(mu) / (MU_SOFT * sigma_total))**2)
                        mu_eff = mu * shrink

                        step = KALSHI_STEP.get(symbol, 250)
                        ladder = _strike_ladder(price_now, step, KALSHI_WINGS)
                        # Relation determined by direction with epsilon; near-neutral -> conservative (n=0 only)
                        relation = ">=" if mu > _MU_REL_EPS else ("<=" if mu < -_MU_REL_EPS else "neutral")
                        
                        # Collect per-strike candidates first
                        candidates: List[Dict] = []

                        for strike in ladder:
                            try:
                                # Near-ladder bias: default to |n|<=1 only (n = (K-S)/step rounded)
                                n = int(round((strike - price_now) / step))
                                if relation == "neutral" and n != 0:
                                    print(f"[TRADE_DEBUG] {symbol} K={strike}: neutral signal; restrict to n=0")
                                    continue
                                if abs(n) > 1:
                                    print(f"[TRADE_DEBUG] {symbol} K={strike}: skip (n={n}) outside near-ladder")
                                    continue
                                # Resolve market info to obtain close_time, then compute TTE-scaled sigma
                                base = (symbol.split('/') [0] or '').upper()
                                try:
                                    if base.startswith('BTC'):
                                        meta = find_btc_hourly_above_below(strike, relation=relation, step=step)
                                    else:
                                        meta = find_eth_hourly_above_below(strike, relation=relation, step=step)
                                except Exception as _e:
                                    meta = None
                                    print(f"[TRADE_DEBUG] {symbol} K={strike}: resolver error relation={relation} step={step} err={_e}")

                                ticker = None
                                sigma_eff = sigma_total
                                tte_minutes = None
                                try:
                                    if isinstance(meta, dict):
                                        ticker = meta.get('ticker')
                                        close_iso = meta.get('close_time')
                                        if close_iso:
                                            import datetime as _dt
                                            close_ts = int(_dt.datetime.fromisoformat(str(close_iso).replace('Z','+00:00')).timestamp())
                                            now_ts = int(time.time())
                                            tte_sec = max(1, close_ts - now_ts)
                                            tte_hours = max(1.0/3600.0, tte_sec / 3600.0)
                                            tte_minutes = max(1, int(tte_sec // 60))
                                            # Scale sigma by sqrt(TTE_hours); enforce floor scaled similarly
                                            sigma_eff = max(KALSHI_SIGMA_FLOOR * math.sqrt(tte_hours), sigma_total * math.sqrt(tte_hours))
                                except Exception:
                                    sigma_eff = sigma_total

                                if not ticker:
                                    # Fallback to resolver wrapper if meta failed
                                    kalshi_symbol = symbol.replace('USDT', 'USD')
                                    try:
                                        ticker = _strike_to_ticker(kalshi_symbol, strike, relation)
                                    except Exception as _e2:
                                        print(f"[TRADE_DEBUG] {symbol} K={strike}: fallback resolve failed relation={relation} err={_e2}")
                                        ticker = None
                                    if not ticker:
                                        print(f"[TRADE_DEBUG] {symbol} K={strike}: could not resolve ticker (relation={relation})")
                                        continue

                                # Relation-aware YES probability at this strike
                                r_star = math.log(strike / price_now)
                                z = (r_star - mu_eff) / max(1e-12, sigma_eff)
                                p_model = (1.0 - _normal_cdf(z)) if relation != "<=" else _normal_cdf(z)
                                # Compute p_blend using market-implied probability from OB mid (if available later)
                                p_use = p_model
                                fair_prob = min(1.0 - PROB_FLOOR, max(PROB_FLOOR, p_use))
                                fair_value_cents = int(fair_prob * 100)

                                # Probability sanity gating to avoid far OTM picks on flat moves
                                if relation == ">=":
                                    if (n == 0 and fair_prob < _MIN_PROB_ANCHOR) or (n == 1 and fair_prob < _MIN_PROB_SAT):
                                        print(f"[TRADE_DEBUG] {symbol} K={strike}: skip YES (p_yes={fair_prob:.2f}) below min for n={n}")
                                        continue
                                elif relation == "<=" :
                                    p_no = 1.0 - fair_prob
                                    if (n == 0 and p_no < _MIN_PROB_ANCHOR) or (n == -1 and p_no < _MIN_PROB_SAT):
                                        print(f"[TRADE_DEBUG] {symbol} K={strike}: skip NO (p_no={p_no:.2f}) below min for n={n}")
                                        continue
                                
                                time.sleep(0.1)
                                orderbook = get_orderbook(ticker)
                                if not orderbook or not orderbook.get('orderbook'):
                                    print(f"[TRADE_DEBUG] {symbol} {ticker}: missing orderbook")
                                    continue
                                
                                ob = orderbook.get('orderbook') or {}
                                if not ob:
                                    continue

                                yes_bid, yes_ask, no_bid, no_ask = yes_best_bid_ask_from_orderbook(ob)
                                # Blend model with market implied mid, if enabled and both bids exist
                                try:
                                    if os.getenv("PBLEND_ENABLE", "1") == "1" and (yes_bid is not None) and (no_bid is not None):
                                        p_tape = ((yes_bid + (100 - no_bid)) / 200.0)
                                        wm = float(os.getenv("PBLEND_W_MODEL", "0.7"))
                                        wt = float(os.getenv("PBLEND_W_TAPE", "0.3"))
                                        lg = wm * _logit(p_model) + wt * _logit(p_tape)
                                        p_blend = _sigmoid(lg)
                                        p_blend = min(1.0 - PROB_FLOOR, max(PROB_FLOOR, p_blend))
                                        fair_value_cents = int(p_blend * 100)
                                        fair_prob = p_blend
                                except Exception:
                                    pass

                                # Extract top-of-book sizes when available (list format [[price,size],...])
                                def _best_size(levels):
                                    try:
                                        best_price = None
                                        best_qty = 0
                                        for lv in levels or []:
                                            price = None
                                            qty = 0
                                            if isinstance(lv, (list, tuple)):
                                                if len(lv) >= 1:
                                                    price = float(lv[0])
                                                if len(lv) >= 2:
                                                    qty = int(lv[1])
                                            elif isinstance(lv, dict):
                                                v = lv.get('price', lv.get('p'))
                                                if v is not None:
                                                    price = float(v)
                                                qty = int(lv.get('size') or lv.get('qty') or 0)
                                            if price is None:
                                                continue
                                            if best_price is None or price > best_price:
                                                best_price = price
                                                best_qty = qty
                                        return int(best_qty)
                                    except Exception:
                                        return 0

                                yes_levels = ob.get('yes', [])
                                no_levels = ob.get('no', [])
                                yes_bid_size = _best_size(yes_levels)
                                no_bid_size = _best_size(no_levels)

                                # Compute side-specific spreads from bids (asks are inferred)
                                spread_yes = (yes_ask - yes_bid) if (yes_ask is not None and yes_bid is not None) else None
                                spread_no  = (no_ask - no_bid)   if (no_ask  is not None and no_bid  is not None) else None

                                # Basic market hygiene gates will be applied per-side below using spread_yes/spread_no

                                # Dynamic edge threshold by TTE band (fallback to config edge when TTE unavailable)
                                def _edge_by_tte(mins: Optional[int]) -> int:
                                    base = max(1, int(DECISION_EDGE_CENTS))
                                    if mins is None:
                                        return base
                                    if mins >= 40:
                                        return max(base, 3)
                                    if mins >= 20:
                                        return max(base, 5)
                                    return max(base, 7)

                                req_edge = _edge_by_tte(tte_minutes)

                                # Compute EV (in cents) for buying YES at yes_ask and buying NO at no_ask
                                ev_yes = None if yes_ask is None else (fair_value_cents - yes_ask)
                                ev_no = None if no_ask is None else ((100 - fair_value_cents) - no_ask)

                                # Directional coherence: pick side implied by relation (or best EV at n=0 in neutral)
                                decision = None
                                if relation == ">=":
                                    if ev_yes is not None:
                                        # Side-specific spread gate for YES
                                        if (spread_yes is not None) and (spread_yes <= MAX_SPREAD_CENTS):
                                            decision = 'yes'
                                        else:
                                            # Spread too wide or missing for YES; leave undecided
                                            pass
                                elif relation == "<=" :
                                    if ev_no is not None:
                                        if (spread_no is not None) and (spread_no <= MAX_SPREAD_CENTS):
                                            decision = 'no'
                                        else:
                                            pass
                                else:  # neutral
                                    if n != 0:
                                        decision = None
                                    else:
                                        if (ev_yes is not None) and (spread_yes is not None) and (ev_yes >= req_edge) and (spread_yes <= MAX_SPREAD_CENTS):
                                            decision = 'yes'
                                        if (ev_no is not None) and (spread_no is not None) and (ev_no >= req_edge) and (spread_no <= MAX_SPREAD_CENTS):
                                            if decision is None or (ev_no > ev_yes):
                                                decision = 'no'

                                # Record candidate
                                candidates.append({
                                    'symbol': symbol,
                                    'strike': strike,
                                    'n': n,
                                    'ticker': ticker,
                                    'relation': relation,
                                    'tte_minutes': tte_minutes,
                                    'sigma_eff': sigma_eff,
                                    'fair_cents': fair_value_cents,
                                    'yes_bid': yes_bid, 'yes_ask': yes_ask,
                                    'no_bid': no_bid, 'no_ask': no_ask,
                                    'yes_bid_size': yes_bid_size, 'no_bid_size': no_bid_size,
                                    'inside_spread': spread_yes if relation == ">=" else spread_no,
                                    'spread_yes': spread_yes,
                                    'spread_no': spread_no,
                                    'req_edge': req_edge,
                                    'ev_yes': ev_yes, 'ev_no': ev_no,
                                    'decision': decision,
                                })
                                        
                            except Exception as e:
                                debug_info.append(f"  Error checking K={strike}: {e}")
                                
                        # Diagnostics for candidates (why taken/rejected)
                        if not candidates:
                            print(f"[TRADE_DEBUG] {symbol}: no viable candidates at n in {-1,0,1}")
                        else:
                            for c in candidates:
                                print(
                                    f"[TRADE_DEBUG] {symbol} K={c['strike']} n={c['n']} rel={c['relation']} "
                                    f"fair={c['fair_cents']}¢ tte={c['tte_minutes']}m "
                                    f"yes={c['yes_bid']}/{c['yes_ask']} no={c['no_bid']}/{c['no_ask']} "
                                    f"EVy={c['ev_yes']} EVn={c['ev_no']} req={c['req_edge']} spread={c['inside_spread']} "
                                    f"liq(noBid)={c['no_bid_size']} liq(yesBid)={c['yes_bid_size']} decision={c['decision']}"
                                )

                        # --- Manage open positions for this symbol (TP/SL and roll) ---
                        now_ts = time.time()
                        by_ticker = {c['ticker']: c for c in candidates}
                        for tkr, pos in list(_OPEN_POSITIONS.items()):
                            if pos.get('symbol') != symbol:
                                continue
                            c = by_ticker.get(tkr)
                            if c is None:
                                try:
                                    ob_resp = get_orderbook(tkr)
                                    ob = ob_resp.get('orderbook') or {}
                                    yb, ya, nb, na = yes_best_bid_ask_from_orderbook(ob)
                                    c = {'yes_bid': yb, 'no_bid': nb}
                                except Exception:
                                    c = None
                            if c is None:
                                continue
                            mark = c['yes_bid'] if pos['side'] == 'yes' else c['no_bid']
                            if mark is None:
                                continue
                            pnl = mark - pos['entry_cents']
                            pos['best_unreal'] = max(pos.get('best_unreal', 0), pnl)
                            pos['sl_cents_dyn'] = _SL_RATCHET_CENTS if pos['best_unreal'] >= _SL_RATCHET_TRIGGER else _SL_BASE_CENTS
                            curr_n = int(round((pos['strike'] - price_now) / step))
                            if abs(curr_n) > 1 and pnl >= 0:
                                print(f"[TRADE] ROLL EXIT {tkr}: pnl=+{pnl}c, drift n={curr_n}")
                                _COOLDOWNS[(pos['symbol'], pos['strike'], pos['side'])] = now_ts + _COOLDOWN_SEC
                                _OPEN_POSITIONS.pop(tkr, None)
                                continue
                            if pnl >= pos['tp_cents']:
                                print(f"[TRADE] TP EXIT {tkr}: +{pnl}c")
                                _COOLDOWNS[(pos['symbol'], pos['strike'], pos['side'])] = now_ts + _COOLDOWN_SEC
                                _OPEN_POSITIONS.pop(tkr, None)
                                continue
                            if pnl <= -pos['sl_cents_dyn']:
                                print(f"[TRADE] SL EXIT {tkr}: {pnl}c")
                                ks = _KILL.setdefault(symbol, {'events': [], 'kill_until': 0})
                                ks['events'] = [e for e in ks['events'] if (now_ts - e[0]) <= _KILL_WINDOW_SEC]
                                ks['events'].append((now_ts, pnl))
                                stops = sum(1 for _, x in ks['events'] if x < 0)
                                loss_sum = -sum(x for _, x in ks['events'] if x < 0)
                                if stops >= _KILL_MAX_STOPS or loss_sum >= _KILL_MAX_LOSS_CENTS:
                                    ks['kill_until'] = now_ts + _KILL_WINDOW_SEC
                                    print(f"[TRADE] KILL-SWITCH engaged for {symbol} ({stops} stops, {loss_sum}c loss)")
                                _COOLDOWNS[(pos['symbol'], pos['strike'], pos['side'])] = now_ts + _COOLDOWN_SEC
                                _OPEN_POSITIONS.pop(tkr, None)
                                continue

                        # Kill-switch guard
                        ks = _KILL.get(symbol, {})
                        if ks.get('kill_until', 0) > time.time():
                            until = int(ks.get('kill_until', 0) - time.time())
                            print(f"[TRADE_DEBUG] {symbol}: kill-switch active for {until}s; skipping new entries")
                            continue

                        # --- Build two-ticket slate: anchor (n=0) and satellite (n=+1) ---
                        def _best_for_n(target_n: int) -> Optional[Dict]:
                            best = None
                            best_ev = -1e9
                            for c in candidates:
                                if c['n'] != target_n:
                                    continue
                                if c['decision'] == 'yes':
                                    if c['ev_yes'] is None or c['ev_yes'] < c['req_edge'] or c['no_bid_size'] < MIN_LIQUIDITY:
                                        continue
                                    ev = c['ev_yes']
                                elif c['decision'] == 'no':
                                    if c['ev_no'] is None or c['ev_no'] < c['req_edge'] or c['yes_bid_size'] < MIN_LIQUIDITY:
                                        continue
                                    ev = c['ev_no']
                                else:
                                    continue
                                if ev > best_ev:
                                    best_ev = ev
                                    best = c
                            return best

                        anchor = _best_for_n(0)
                        satellite = _best_for_n(+1)

                        # Submit at most two entries; respect cooldowns and existing positions
                        for pick, label in [(anchor, 'ANCHOR'), (satellite, 'SAT')]:
                            if not pick:
                                continue
                            key_cd = (pick['symbol'], pick['strike'], pick['decision'])
                            if _COOLDOWNS.get(key_cd, 0) > time.time():
                                left = int(_COOLDOWNS.get(key_cd, 0) - time.time())
                                print(f"[TRADE_DEBUG] cooldown skip {pick['ticker']} ({left}s left) side={pick['decision']} K={pick['strike']}")
                                continue
                            if pick['ticker'] in _OPEN_POSITIONS:
                                print(f"[TRADE_DEBUG] already open {pick['ticker']} side={pick['decision']} K={pick['strike']}")
                                continue

                            if pick['decision'] == 'yes':
                                target_px = clamp_post_only("yes", "buy", pick['fair_cents'], pick['yes_bid'], pick['yes_ask'], pick['no_bid'], pick['no_ask'])
                                entry_px = pick['yes_ask']
                                side = 'yes'
                            else:
                                target_px = clamp_post_only("no", "buy", 100 - pick['fair_cents'], pick['yes_bid'], pick['yes_ask'], pick['no_bid'], pick['no_ask'])
                                entry_px = pick['no_ask']
                                side = 'no'

                            try:
                                if TRADING_MODE == 'DRY':
                                    print(f"[TRADE] [{label}] OPEN {side.upper()} {pick['ticker']} @ {entry_px}¢ (fair={pick['fair_cents']}¢, TTE={pick['tte_minutes']}m)")
                                    order_result = {'order_id': f'dry-{int(time.time()*1000)}'}
                                else:
                                    order_result = create_order_compat(ticker=pick['ticker'], side=side, price_cents=target_px, qty=1, post_only=True)
                                opportunities_found += 1
                                _OPEN_POSITIONS[pick['ticker']] = {
                                    'symbol': pick['symbol'],
                                    'ticker': pick['ticker'],
                                    'strike': pick['strike'],
                                    'side': side,
                                    'n': pick['n'],
                                    'entry_cents': int(entry_px) if entry_px is not None else int(target_px),
                                    'tp_cents': _TP_CENTS,
                                    'sl_cents_dyn': _SL_BASE_CENTS,
                                    'best_unreal': 0,
                                    'entry_ts': time.time(),
                                }
                            except Exception as trade_error:
                                print(f"[TRADE] OPEN failed {pick['ticker']}: {trade_error}")

                    except Exception as e:
                        debug_info.append(f"[TRADE] Error processing {symbol}: {e}")
                
                if opportunities_found == 0:
                    print(f"[TRADE] No arbitrage opportunities")
                else:
                    print(f"[TRADE] ✅ Found {opportunities_found} arbitrage opportunities!")
                
                print(f"[TRADE] Next scan in 60s...")
                time.sleep(60)
                
            except Exception as e:
                print(f"[TRADE] Error in ML trade worker: {e}")
                time.sleep(60)
    
    thread = threading.Thread(target=_trade_worker, daemon=True)
    thread.start()

def _strike_ladder(price_now: float, step: int, wings: int) -> list[int]:
    base = int(round(price_now / step) * step)
    strikes = [base + i*step for i in range(-wings, wings+1)]
    return [s for s in strikes if s > 0]

def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
