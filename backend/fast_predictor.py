"""
Fast Real-Time Prediction Pipeline - Separated from Model Training
Uses pre-trained models for millisecond-speed inference
"""
import time
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading

# Import the model trainer (package-relative)
from .model_trainer import get_model_trainer
from .features.engineering import build_features_vectorized, add_flow_features, add_flow_plus
from .api.kalshi import get_orderbook
from .features.imbalance import order_book_imbalance, weighted_mid_price, bid_ask_spread
import os
import warnings


def get_microstructure_features(ticker: str) -> dict:
    """
    Fetches microstructure features for a given ticker.
    Returns a dictionary of features or an empty dict on error.
    """
    try:
        orderbook_data = get_orderbook(ticker)
        if not orderbook_data or not orderbook_data.get('orderbook'):
            return {}

        orderbook = orderbook_data['orderbook']
        
        return {
            'obi': order_book_imbalance(orderbook),
            'wmp': weighted_mid_price(orderbook),
            'spread': bid_ask_spread(orderbook)
        }
    except Exception as e:
        # In a fast predictor, we never want this to fail loudly
        # print(f"[WARN] Could not fetch microstructure features for {ticker}: {e}")
        return {}

# Import GRU model availability flag (TensorFlow presence)
try:
    from .models.recurrent_model import HAS_TENSORFLOW
    HAS_GRU = HAS_TENSORFLOW
except ImportError:
    print("[WARNING] GRU model imports failed. GRU predictions will not be available.")
    HAS_GRU = False

def prepare_inference_data(recent_features, scaler, sequence_length: int = 60) -> Optional[np.ndarray]:
    """
    Prepare a single GRU inference batch from recent feature rows.
    """
    try:
        if recent_features is None:
            print(f"[FAST_PREDICTOR] prepare_inference_data: recent_features is None.")
            return None
        # Accept either numpy array or DataFrame; align to scaler's expected columns if available
        if isinstance(recent_features, pd.DataFrame):
            try:
                feat_names = getattr(scaler, 'feature_names_in_', None)
                if feat_names is not None:
                    recent_features = recent_features.reindex(columns=list(feat_names), fill_value=0.0)
            except Exception:
                pass
            arr = recent_features.values
        else:
            arr = np.asarray(recent_features)
        print(f"[FAST_PREDICTOR] prepare_inference_data: arr shape: {arr.shape}, ndim: {arr.ndim}.")
        if arr.ndim != 2 or arr.shape[0] < sequence_length:
            print(f"[FAST_PREDICTOR] prepare_inference_data: arr shape invalid ({arr.shape[0]} < {sequence_length}).")
            return None

        # --- NEW LOGGING HERE ---
        print(f"[FAST_PREDICTOR] prepare_inference_data: arr contains NaNs: {np.isnan(arr).any()}")
        print(f"[FAST_PREDICTOR] prepare_inference_data: arr contains Infs: {np.isinf(arr).any()}")
        print(f"[FAST_PREDICTOR] prepare_inference_data: scaler type: {type(scaler)}")
        print(f"[FAST_PREDICTOR] prepare_inference_data: scaler.scale_ is None: {getattr(scaler, 'scale_', None) is None}")
        print(f"[FAST_PREDICTOR] prepare_inference_data: scaler.min_ is None: {getattr(scaler, 'min_', None) is None}")
        if getattr(scaler, 'scale_', None) is not None:
            print(f"[FAST_PREDICTOR] prepare_inference_data: scaler.scale_ contains NaNs: {np.isnan(scaler.scale_).any()}")
            print(f"[FAST_PREDICTOR] prepare_inference_data: scaler.scale_ contains Infs: {np.isinf(scaler.scale_).any()}")
            print(f"[FAST_PREDICTOR] prepare_inference_data: scaler.scale_ contains Zeros: {(scaler.scale_ == 0).any()}")
        # --- END NEW LOGGING ---

        # Ensure feature count matches scaler expectation; try to correct if not
        try:
            expected = getattr(scaler, 'n_features_in_', None)
        except Exception:
            expected = None
        if expected is not None and arr.shape[1] != expected:
            try:
                feat_names = getattr(scaler, 'feature_names_in_', None)
                if isinstance(recent_features, pd.DataFrame) and feat_names is not None:
                    # Hard-align to scaler feature names and order
                    recent_features = recent_features.reindex(columns=list(feat_names), fill_value=0.0)
                    arr = recent_features.values
            except Exception:
                pass
            # If still mismatched, crop or pad as last-resort safety
            if arr.shape[1] > expected:
                print(f"[FAST_PREDICTOR] prepare_inference_data: trimming features {arr.shape[1]}→{expected}")
                arr = arr[:, :expected]
            elif arr.shape[1] < expected:
                pad = expected - arr.shape[1]
                print(f"[FAST_PREDICTOR] prepare_inference_data: padding features {arr.shape[1]}→{expected}")
                arr = np.pad(arr, ((0, 0), (0, pad)), mode='constant', constant_values=0.0)

        # Always use ndarray input to bypass strict feature-name checks at transform time.
        # We already aligned the order above when possible.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
                category=UserWarning,
            )
            scaled_feats = scaler.transform(arr)
        print(f"[FAST_PREDICTOR] prepare_inference_data: scaled_feats shape: {scaled_feats.shape}")
        print(f"[FAST_PREDICTOR] prepare_inference_data: scaled_feats contains NaNs: {np.isnan(scaled_feats).any()}")
        print(f"[FAST_PREDICTOR] prepare_inference_data: scaled_feats contains Infs: {np.isinf(scaled_feats).any()}")

        window = scaled_feats[-sequence_length:]
        return window[np.newaxis, ...].astype(np.float32)
    except Exception as e:
        print(f"[FAST_PREDICTOR] prepare_inference_data: Exception caught: {e}")
        return None

class FastPredictor:
    """Real-time prediction pipeline using pre-trained models"""
    
    def __init__(self):
        self.model_trainer = get_model_trainer()
        self.prediction_cache = {}  # symbol -> (prediction, timestamp)
        self.cache_ttl_seconds = 30  # Cache predictions for 30 seconds
        self.lock = threading.Lock()
        # Diagnostics: track why a fallback path was used most recently
        self._last_fallback_reason: Optional[str] = None
        # GRU diagnostics: track which symbols we've printed for
        self._gru_diag_printed = set()
        
        # print("[FAST_PREDICTOR] Initialized for real-time inference") # REDUCED: Only show on startup
    
    def predict_next_hour(self, symbol: str, current_price: float, 
                         ws_close: Optional[float] = None, 
                         ws_ts: Optional[int] = None) -> Optional[float]:
        try:
            # Always get the latest model info from the trainer.
            # The trainer's get_model_for_symbol now handles freshness checks and reloads.
            model_info = self.model_trainer.get_model_for_symbol(symbol)
            
            if not model_info:
                # If no model is available (e.g., first run, or training failed),
                # trigger a retrain and use current price as fallback.
                self.model_trainer.trigger_retrain(symbol)
                self.clear_cache(symbol) # Clear predictor's cache as well
                return current_price
            
            # If model is available but stale, trigger retrain and use current price as fallback.
            # The next prediction cycle will pick up the newly trained model.
            if not self.model_trainer.is_model_fresh(symbol):
                print(f"[FAST_PREDICTOR] {symbol}: Model is stale. Triggering retrain and using current price as fallback.")
                self.model_trainer.trigger_retrain(symbol)
                self.clear_cache(symbol) # Clear predictor's cache as well
                return current_price
            
            # Fast feature calculation (only current state)
            features = self._get_current_features(symbol, current_price, ws_close, ws_ts)
            if features is None:
                return None
            
            # Fast inference
            prediction = self._fast_inference(model_info, features, symbol, current_price)
            
            # Cache the result
            self._cache_prediction(symbol, prediction)

            # Optional lightweight trace for observability (enable via env var)
            try:
                import os
                if os.environ.get('FAST_PREDICTOR_TRACE', '0') == '1':
                    print(f"[FAST_PREDICTOR_TRACE] {symbol} -> {prediction:.6f}")
            except Exception:
                pass
            
            return prediction
            
        except Exception as e:
            # print(f"[FAST_PREDICTOR] Error predicting {symbol}: {e}") # REDUCED: Too frequent
            return None
    
    def _get_cached_prediction(self, symbol: str, model_timestamp: Optional[datetime] = None) -> Optional[float]:
        """Get cached prediction if still valid and not older than model"""
        with self.lock:
            if symbol not in self.prediction_cache:
                return None
            
            prediction, cache_timestamp = self.prediction_cache[symbol]
            age = (datetime.now() - cache_timestamp).total_seconds()
            
            # Check if cache is still within TTL
            if age > self.cache_ttl_seconds:
                del self.prediction_cache[symbol]
                return None
            
            # Check if model is newer than cache
            if model_timestamp and cache_timestamp < model_timestamp:
                # print(f"[FAST_PREDICTOR] {symbol}: Cache ({cache_timestamp}) older than model ({model_timestamp}), clearing cache") # REDUCED: Too frequent
                del self.prediction_cache[symbol]
                return None
            
            return prediction
    
    def _cache_prediction(self, symbol: str, prediction: float):
        """Cache prediction with timestamp"""
        with self.lock:
            self.prediction_cache[symbol] = (prediction, datetime.now())
    
    def _get_current_features(self, symbol: str, current_price: float,
                            ws_close: Optional[float] = None, 
                            ws_ts: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Get features for current state - OPTIMIZED for speed
        Uses lightweight feature calculation instead of heavy build_features()
        """
        try:
            # Prefer deriving indicators from the canonical crypto_prices DB
            # Query the last N OHLCV rows and append current tick, then run
            # the exact training feature builder so runtime features match.
            try:
                from .data.database import get_db_path
                import sqlite3
                dbp = get_db_path()
                conn = sqlite3.connect(dbp)
                cur = conn.cursor()
                # request a reasonable window for indicators (e.g. 240 bars)
                cur.execute("""
                    SELECT timestamp, open_price, high_price, low_price, close_price, volume,
                           trades, quote_volume, taker_base_volume, taker_quote_volume
                    FROM crypto_prices
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT 240
                """, (symbol,))
                rows = cur.fetchall()
                conn.close()

                if not rows:
                    # fallback to lightweight calculation when DB is empty
                    self._last_fallback_reason = "DB_EMPTY"
                    recent_prices = self._get_cached_recent_prices(symbol, current_price)
                    if recent_prices is None:
                        self._last_fallback_reason = "CACHE_MISS"
                        return None
                    return self._calculate_lightweight_features(recent_prices, current_price)

                # build DataFrame in chronological order
                rows.reverse()
                df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                 'trades', 'quote_volume', 'taker_base_volume', 'taker_quote_volume'])
                # append a synthetic current bar if provided (volume 0)
                if current_price is not None:
                    now_ts = int(datetime.now().timestamp())
                    df = pd.concat([df, pd.DataFrame([{
                        'timestamp': now_ts,
                        'open': current_price,
                        'high': current_price,
                        'low': current_price,
                        'close': current_price,
                        'volume': 0.0
                    }])], ignore_index=True)

                df.set_index('timestamp', inplace=True)
                df['volume'] = df['volume'].fillna(0)

                features_full = build_features_vectorized(df['open'], df['high'], df['low'], df['close'], df['volume'])
                try:
                    from .config import TRAIN_WITH_FLOW_FEATS, ENABLE_FLOW_FEATS
                except Exception:
                    TRAIN_WITH_FLOW_FEATS = False
                    ENABLE_FLOW_FEATS = False
                if TRAIN_WITH_FLOW_FEATS and ENABLE_FLOW_FEATS:
                    df_flow = add_flow_features(df)
                    base_flow_cols = [
                        'trade_count_1m','avg_trade_size_1m','dollar_volume_1m',
                        'taker_buy_ratio','taker_imbalance',
                        'ema_vol_5','ema_vol_15','ema_trades_5','ema_trades_15','ema_tbr_5'
                    ]
                    for col in base_flow_cols:
                        if col in df_flow.columns and col not in features_full.columns:
                            features_full[col] = df_flow[col]
                    # Extended
                    df_ext = add_flow_plus(df)
                    ext_cols = [
                        'ema_gap_5_15','ema_gap_15_60','pct_up_10','pct_up_30','runlen_up_15','roc_5',
                        'bb_width_20','squeeze','donchian_pos_60','range_ratio',
                        'RV_60m','rv_term','vol_of_vol_60','ATR_14','ADX_14','trend_state','chop_flag',
                        'taker_buy_ratio_1m','tbi_5','tbi_15','tbi_slope','dollar_vol_1m','ema_dv_5','ema_dv_15','burst',
                        'tr_rate_1m','ema_tr_5','ema_tr_15','cadence_ratio','taker_imbalance','ti_5','ti_cross','d_obv_15','flow_price_beta_60',
                        'eff_ratio_10','whipsaw_12'
                    ]
                    for col in ext_cols:
                        if col in df_ext.columns and col not in features_full.columns:
                            features_full[col] = df_ext[col]
                # select last row of features (most recent)
                last_row = features_full.iloc[[-1]]
                # Keep same columns as training order
                base_feat_order = [
                    "gap_ema5","gap_ema15","gap_ema30","gap_ema60",
                    "macd","macd_sig","macd_hist",
                    "rsi14","bb_pctB",
                    "rv5","rv15","rv60",
                    "vwap_gap","donch20",
                    "min_of_hour_sin","min_of_hour_cos",
                    "atr14", "stoch_k", "stoch_d",
                    "obv", "adx",
                    "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b",
                    "williams_r",
                ]
                ext_order = []
                if TRAIN_WITH_FLOW_FEATS and ENABLE_FLOW_FEATS:
                    ext_order = [
                        'trade_count_1m','avg_trade_size_1m','dollar_volume_1m',
                        'taker_buy_ratio','taker_imbalance','ema_vol_5','ema_vol_15','ema_trades_5','ema_trades_15','ema_tbr_5',
                        'ema_gap_5_15','ema_gap_15_60','pct_up_10','pct_up_30','runlen_up_15','roc_5','bb_width_20','squeeze','donchian_pos_60','range_ratio',
                        'RV_60m','rv_term','vol_of_vol_60','ATR_14','ADX_14','trend_state','chop_flag',
                        'taker_buy_ratio_1m','tbi_5','tbi_15','tbi_slope','dollar_vol_1m','ema_dv_5','ema_dv_15','burst','tr_rate_1m','ema_tr_5','ema_tr_15','cadence_ratio','ti_5','ti_cross','d_obv_15','flow_price_beta_60','eff_ratio_10','whipsaw_12'
                    ]
                feat_order = base_feat_order + ext_order
                # De-duplicate while preserving order (avoid duplicate columns like 'taker_imbalance')
                seen = set()
                dedup_order = []
                for col in feat_order:
                    if col not in seen:
                        seen.add(col)
                        dedup_order.append(col)
                feat_order = dedup_order
                # If any expected column missing, fall back to lightweight
                missing = [c for c in feat_order if c not in last_row.columns]
                if missing:
                    self._last_fallback_reason = "MISSING_FEATURES"
                    recent_prices = self._get_cached_recent_prices(symbol, current_price)
                    if recent_prices is None:
                        self._last_fallback_reason = "CACHE_MISS"
                        return None
                    return self._calculate_lightweight_features(recent_prices, current_price)

                features_df = last_row[feat_order].reset_index(drop=True)

                # --- Optional: Add Microstructure Features ---
                if os.environ.get('ENABLE_MICRO_FEATS', '0') == '1':
                    micro_features = get_microstructure_features(symbol)
                    if micro_features:
                        for key, value in micro_features.items():
                            features_df[key] = value
                
                # Success path: using DB-derived features
                self._last_fallback_reason = None
                return features_df
            except Exception:
                # On any DB or feature building error, gracefully fallback
                self._last_fallback_reason = "DB_ERROR"
                recent_prices = self._get_cached_recent_prices(symbol, current_price)
                if recent_prices is None:
                    self._last_fallback_reason = "CACHE_MISS"
                    return None
                return self._calculate_lightweight_features(recent_prices, current_price)
            
        except Exception as e:
            # print(f"[FAST_PREDICTOR] Error getting features for {symbol}: {e}") # REDUCED: Too frequent
            return None
    
    def _get_cached_recent_prices(self, symbol: str, current_price: float) -> Optional[np.ndarray]:
        """Get recent prices with caching for speed"""
        cache_key = f"recent_prices_{symbol}"
        
        # Check if we have cached recent prices
        if hasattr(self, '_price_cache'):
            cached_data = self._price_cache.get(cache_key)
            if cached_data is not None:
                prices, timestamp = cached_data
                # Use cache if less than 60 seconds old
                if (datetime.now() - timestamp).total_seconds() < 60:
                    # Append current price and return
                    updated_prices = np.append(prices, current_price)
                    return updated_prices[-50:]  # Keep last 50 prices
        
        # Cache miss - get from database (minimal query)
        try:
            from .data.database import get_db_path
            import sqlite3
            
            db_path = get_db_path()
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            
            # Get only last 50 prices - much smaller query
            cur.execute("""
                SELECT close_price FROM crypto_prices 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 50
            """, (symbol,))
            
            rows = cur.fetchall()
            conn.close()
            
            if not rows:
                return None
                
            prices = np.array([float(r[0]) for r in reversed(rows)], dtype=float)
            prices = np.append(prices, current_price)
            
            # Cache the result
            if not hasattr(self, '_price_cache'):
                self._price_cache = {}
            self._price_cache[cache_key] = (prices, datetime.now())
            
            return prices[-50:]  # Keep last 50
            
        except Exception as e:
            # print(f"[FAST_PREDICTOR] Error getting cached prices for {symbol}: {e}") # REDUCED: Too frequent
            return None
    
    def _calculate_lightweight_features(self, prices: np.ndarray, current_price: float) -> pd.DataFrame:
        """
        Calculate lightweight features - MUCH faster than build_features()
        Uses only essential features that can be computed quickly
        """
        try:
            if len(prices) < 20:
                # Not enough data, return zeros DataFrame with expected columns
                features_dict = {
                    "gap_ema5": 0.0, "gap_ema15": 0.0, "gap_ema30": 0.0, "gap_ema60": 0.0,
                    "macd": 0.0, "macd_sig": 0.0, "macd_hist": 0.0,
                    "rsi14": 50.0, "bb_pctB": 0.5,
                    "rv5": 0.0, "rv15": 0.0, "rv60": 0.0,
                    "vwap_gap": 0.0, "donch20": 0.5,
                    "min_of_hour_sin": 0.0, "min_of_hour_cos": 0.0,
                }
                return pd.DataFrame([features_dict])
            
            # Ensure we have enough data for calculations
            prices = np.asarray(prices, dtype=float)
            
            # Simple moving averages (fast calculation)
            sma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else current_price
            sma_15 = np.mean(prices[-15:]) if len(prices) >= 15 else current_price
            sma_30 = np.mean(prices[-30:]) if len(prices) >= 30 else current_price
            
            # Price gaps (ratios)
            gap_5 = (current_price / sma_5 - 1.0) if sma_5 > 0 else 0.0
            gap_15 = (current_price / sma_15 - 1.0) if sma_15 > 0 else 0.0
            gap_30 = (current_price / sma_30 - 1.0) if sma_30 > 0 else 0.0
            
            # Simple volatility - fix array slicing
            if len(prices) >= 11:  # Need at least 11 for 10-length diff
                price_subset = prices[-11:]  # Get last 11 prices
                returns = np.diff(price_subset) / price_subset[:-1]  # 10 returns from 11 prices
                volatility = np.std(returns) if len(returns) > 1 else 0.0
            else:
                volatility = 0.0
            
            # Simple momentum
            momentum = (current_price / prices[-10] - 1.0) if len(prices) >= 10 and prices[-10] > 0 else 0.0
            
            # Price position (current vs recent high/low)
            if len(prices) >= 20:
                recent_high = np.max(prices[-20:])
                recent_low = np.min(prices[-20:])
                price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
            else:
                price_position = 0.5
            
            # Simple RSI approximation
            if len(prices) >= 15:
                price_changes = np.diff(prices[-15:])
                gains = price_changes[price_changes > 0]
                losses = -price_changes[price_changes < 0]
                avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
                rs = avg_gain / avg_loss if avg_loss > 0 else 1.0
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50.0
            
            # Time-based features (minimal)
            import math
            hour = datetime.now().hour
            minute = datetime.now().minute
            hour_sin = math.sin(2 * math.pi * hour / 24)
            minute_sin = math.sin(2 * math.pi * minute / 60)
            
            # Return lightweight feature vector as DataFrame with feature names (same order as training)
            features_dict = {
                "gap_ema5": gap_5, "gap_ema15": gap_15, "gap_ema30": gap_30, "gap_ema60": 0.0,
                "macd": 0.0, "macd_sig": 0.0, "macd_hist": 0.0,
                "rsi14": rsi, "bb_pctB": price_position,
                "rv5": volatility, "rv15": volatility, "rv60": volatility,
                "vwap_gap": gap_15, "donch20": price_position,
                "min_of_hour_sin": minute_sin, "min_of_hour_cos": hour_sin,
            }
            
            # Create DataFrame with feature names to match training
            features_df = pd.DataFrame([features_dict])
            return features_df
            
        except Exception as e:
            # print(f"[FAST_PREDICTOR] Error calculating lightweight features: {e}") # REDUCED: Too frequent
            # On error, return a single-row DataFrame of zeros with expected columns
            features_dict = {
                "gap_ema5": 0.0, "gap_ema15": 0.0, "gap_ema30": 0.0, "gap_ema60": 0.0,
                "macd": 0.0, "macd_sig": 0.0, "macd_hist": 0.0,
                "rsi14": 50.0, "bb_pctB": 0.5,
                "rv5": 0.0, "rv15": 0.0, "rv60": 0.0,
                "vwap_gap": 0.0, "donch20": 0.5,
                "min_of_hour_sin": 0.0, "min_of_hour_cos": 0.0,
            }
            return pd.DataFrame([features_dict])
    
    def _predict_with_gru(self, symbol: str, gru_model, gru_scaler, current_price: float) -> Optional[float]:
        """Make prediction using GRU model with proper sequence preparation"""
        try:
            if not HAS_GRU:
                print(f"[FAST_PREDICTOR] {symbol}: GRU prediction skipped (TensorFlow not available).")
                return None
            
            # Get recent historical features for sequence creation
            recent_features = self._get_recent_features_for_gru(symbol, sequence_length=60)
            if recent_features is None:
                print(f"[FAST_PREDICTOR] {symbol}: GRU prediction skipped (Insufficient recent features).")
                return None

            # Optional diagnostics about scaler vs. runtime features (print once per symbol)
            try:
                if os.environ.get('GRU_DIAGNOSTIC', '0') == '1' and symbol not in self._gru_diag_printed:
                    expected_n = getattr(gru_scaler, 'n_features_in_', None)
                    expected_names = list(getattr(gru_scaler, 'feature_names_in_', []))
                    runtime_names = list(recent_features.columns) if hasattr(recent_features, 'columns') else []
                    extra = [c for c in runtime_names if expected_names and c not in expected_names]
                    missing = [c for c in expected_names if c not in runtime_names]
                    print(f"[GRU_DIAG] symbol={symbol} expected_n_features={expected_n}")
                    if expected_names:
                        print(f"[GRU_DIAG] scaler_feature_names={expected_names}")
                    print(f"[GRU_DIAG] runtime_feature_names={runtime_names}")
                    print(f"[GRU_DIAG] extra_cols={extra}")
                    print(f"[GRU_DIAG] missing_cols={missing}")
                    self._gru_diag_printed.add(symbol)
            except Exception:
                pass
            
            # Prepare data for GRU inference
            inference_input = prepare_inference_data(recent_features, gru_scaler, sequence_length=60)
            if inference_input is None:
                print(f"[FAST_PREDICTOR] {symbol}: GRU prediction skipped (Failed to prepare inference input).")
                return None
            
            # --- NEW LOGGING HERE ---
            print(f"[FAST_PREDICTOR] {symbol}: GRU inference_input shape: {inference_input.shape}")
            print(f"[FAST_PREDICTOR] {symbol}: GRU inference_input contains NaNs: {np.isnan(inference_input).any()}")
            print(f"[FAST_PREDICTOR] {symbol}: GRU inference_input contains Infs: {np.isinf(inference_input).any()}")
            # --- END NEW LOGGING ---

            # Make prediction
            prediction = gru_model.predict(inference_input, verbose=0)[0][0]
            
            # --- NEW LOGGING HERE ---
            print(f"[FAST_PREDICTOR] {symbol}: GRU raw prediction: {prediction}")
            # --- END NEW LOGGING ---

            # Validate prediction
            if not np.isfinite(prediction) or prediction <= 0:
                print(f"[FAST_PREDICTOR] {symbol}: GRU prediction invalid (not finite or <= 0): {prediction}")
                return None
            
            return float(prediction)
            
        except Exception as e:
            print(f"[FAST_PREDICTOR] GRU prediction error for {symbol}: {e}")
            return None
    
    def _get_recent_features_for_gru(self, symbol: str, sequence_length: int = 60) -> Optional[pd.DataFrame]:
        """Get recent features formatted for GRU prediction"""
        try:
            from .data.database import get_db_path
            import sqlite3
            
            db_path = get_db_path()
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            
            # Get sufficient recent data for feature calculation + sequence
            cur.execute("""
                WITH ranked AS (
                  SELECT
                    timestamp, open_price, high_price, low_price, close_price, volume,
                    trades, quote_volume, taker_base_volume, taker_quote_volume,
                    ROW_NUMBER() OVER (
                      PARTITION BY timestamp
                      ORDER BY (source='FINAZON_REST') DESC, rowid DESC
                    ) rn
                  FROM crypto_prices
                  WHERE symbol = ?
                )
                SELECT timestamp, open_price, high_price, low_price, close_price, volume,
                       trades, quote_volume, taker_base_volume, taker_quote_volume
                FROM ranked
                WHERE rn = 1
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, sequence_length + 400))  # Extra buffer for feature calculation (satisfy 300-window features)
            
            rows = cur.fetchall()
            conn.close()
            
            if not rows or len(rows) < sequence_length + 50:
                return None
            
            # Convert to DataFrame in chronological order
            rows.reverse()
            df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                             'trades', 'quote_volume', 'taker_base_volume', 'taker_quote_volume'])
            df['volume'] = df['volume'].fillna(0)
            df.set_index('timestamp', inplace=True)
            
            # Build features using same pipeline as training
            features_df = build_features_vectorized(df['open'], df['high'], df['low'], df['close'], df['volume'])
            # IMPORTANT: Mirror GRU training feature set exactly (training uses TRAIN_WITH_FLOW_FEATS only)
            try:
                from .config import TRAIN_WITH_FLOW_FEATS
            except Exception:
                TRAIN_WITH_FLOW_FEATS = False

            flow_cols = [
                'trade_count_1m','avg_trade_size_1m','dollar_volume_1m',
                'taker_buy_ratio','taker_imbalance',
                'ema_vol_5','ema_vol_15','ema_trades_5','ema_trades_15','ema_tbr_5'
            ]
            ext_cols = [
                'ema_gap_5_15','ema_gap_15_60','pct_up_10','pct_up_30','runlen_up_15','roc_5',
                'bb_width_20','squeeze','donchian_pos_60','range_ratio',
                'RV_60m','rv_term','vol_of_vol_60','ATR_14','ADX_14','trend_state','chop_flag',
                'taker_buy_ratio_1m','tbi_5','tbi_15','tbi_slope','dollar_vol_1m','ema_dv_5','ema_dv_15','burst',
                'tr_rate_1m','ema_tr_5','ema_tr_15','cadence_ratio','taker_imbalance','ti_5','ti_cross','d_obv_15','flow_price_beta_60',
                'eff_ratio_10','whipsaw_12'
            ]

            if TRAIN_WITH_FLOW_FEATS:
                # Base flow bundle
                try:
                    df_flow = add_flow_features(df)
                    for col in flow_cols:
                        if col in df_flow.columns and col not in features_df.columns:
                            features_df[col] = df_flow[col]
                except Exception:
                    pass
                # Extended bundle
                try:
                    df_ext = add_flow_plus(df)
                    for col in ext_cols:
                        if col in df_ext.columns and col not in features_df.columns:
                            features_df[col] = df_ext[col]
                except Exception:
                    pass

            # Use same feature order as training
            base_feat_order = [
                "gap_ema5", "gap_ema15", "gap_ema30", "gap_ema60",
                "macd", "macd_sig", "macd_hist",
                "rsi14", "bb_pctB",
                "rv5", "rv15", "rv60",
                "vwap_gap", "donch20",
                "min_of_hour_sin", "min_of_hour_cos",
                "atr14", "stoch_k", "stoch_d",
                "obv", "adx",
                "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b",
                "williams_r",
            ]
            feat_order = base_feat_order + (flow_cols + ext_cols if TRAIN_WITH_FLOW_FEATS else [])
            # De-duplicate while preserving order
            seen = set()
            dedup_order = []
            for col in feat_order:
                if col not in seen:
                    seen.add(col)
                    dedup_order.append(col)
            feat_order = dedup_order
            
            # Align to training feature order; tolerate missing columns (will be filled)
            missing = [col for col in feat_order if col not in features_df.columns]
            if missing:
                # Add missing columns as NaN; they will be forward-filled/filled below
                for col in missing:
                    features_df[col] = np.nan

            # Keep only ordered columns
            features_df = features_df[feat_order]

            # Clean up: replace inf, forward-fill to handle trailing NaNs (e.g., shifted/rolling features)
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.ffill()
            # Fill any remaining NaNs (e.g., columns with no data at all) with 0.0
            features_df = features_df.fillna(0.0)

            # Return the recent feature values as DataFrame (keep names for scaler alignment)
            return features_df
            
        except Exception as e:
            print(f"[FAST_PREDICTOR] Error getting GRU features for {symbol}: {e}")
            return None
    
    def _fast_inference(self, model_info: dict, features: pd.DataFrame, symbol: str, current_price: float) -> float:
        """Fast inference using a hybrid of pre-trained models"""
        try:
            lgbm_model = model_info.get('lgbm_model')
            ridge_model = model_info.get('ridge_model')
            huber_model = model_info.get('huber_model')
            gru_model = model_info.get('gru_model')
            gru_scaler = model_info.get('gru_scaler')
            scaler = model_info['scaler']
            # Align runtime features to the training feature order to avoid shape errors
            feat_order_saved = model_info.get('feat_order')
            if isinstance(feat_order_saved, (list, tuple)) and len(feat_order_saved) > 0:
                # Fill any missing columns with 0 and drop unexpected extras
                try:
                    features = features.reindex(columns=list(feat_order_saved), fill_value=0.0)
                except Exception:
                    pass

            # Defensive cleanup: replace non-finite with zeros to avoid scaler NaNs/Infs
            try:
                features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            except Exception:
                pass
            
            # Convert DataFrame to numpy array for scaling
            features_array = features.values
            
            # Scale features
            features_scaled = scaler.transform(features_array)
            
            # Create scaled DataFrame with original (aligned) feature names for LightGBM
            features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
            
            # Validate scaled features
            if not np.all(np.isfinite(features_scaled)):
                print(f"[FAST_PREDICTOR] {symbol}: Invalid scaled features, using current price")
                return current_price
            
            # Predictions from all models
            predictions = []
            
            # Traditional models
            if lgbm_model:
                # If the trained model saved a best iteration, pass it to predict to ensure using best booster
                try:
                    best_it = getattr(lgbm_model, '_best_iteration', None)
                    if best_it:
                        lgbm_pred = lgbm_model.predict(features_scaled_df, num_iteration=best_it)[0]
                    else:
                        lgbm_pred = lgbm_model.predict(features_scaled_df)[0]
                except TypeError:
                    # Some LightGBM wrappers may not accept num_iteration; fallback
                    lgbm_pred = lgbm_model.predict(features_scaled_df)[0]
                except Exception:
                    lgbm_pred = lgbm_model.predict(features_scaled_df)[0]
                predictions.append(lgbm_pred)
                # print(f"[FAST_PREDICTOR] {symbol}: LightGBM prediction = {lgbm_pred:.2f}") # REDUCED: Too frequent
            if ridge_model:
                ridge_pred = ridge_model.predict(features_scaled)[0]
                predictions.append(ridge_pred)
                # print(f"[FAST_PREDICTOR] {symbol}: Ridge prediction = {ridge_pred:.2f}") # REDUCED: Too frequent
            if huber_model:
                huber_pred = huber_model.predict(features_scaled)[0]
                predictions.append(huber_pred)
                # print(f"[FAST_PREDICTOR] {symbol}: Huber prediction = {huber_pred:.2f}") # REDUCED: Too frequent
            
            # GRU model prediction
            if HAS_GRU:
                if gru_model and gru_scaler:
                    try:
                        gru_pred = self._predict_with_gru(symbol, gru_model, gru_scaler, current_price)
                        if gru_pred is not None:
                            predictions.append(gru_pred)
                        else:
                            # Add a log to explain why prediction was None
                            print(f"[FAST_PREDICTOR] {symbol}: _predict_with_gru returned None. Check data availability/quality for GRU.")
                    except Exception as e:
                        print(f"[FAST_PREDICTOR] {symbol}: GRU prediction failed with exception: {e}")
                else:
                    # Add a log to explain why model/scaler is missing
                    print(f"[FAST_PREDICTOR] {symbol}: GRU model or scaler not available. Skipping GRU prediction.")

            if not predictions:
                raise ValueError("No models available for prediction")

            # Simple averaging for hybrid prediction
            final_prediction = np.mean(predictions)
            final_prediction = float(final_prediction)
            
            # Validate prediction is reasonable
            if not np.isfinite(final_prediction) or final_prediction <= 0:
                # print(f"[FAST_PREDICTOR] {symbol}: Invalid prediction {final_prediction}, using current price {current_price}") # REDUCED: Too frequent
                return current_price
            
            # print(f"[FAST_PREDICTOR] {symbol}: Hybrid prediction = {final_prediction:.2f}") # REDUCED: Too frequent
            
            # --- VERBOSE DIAGNOSTICS (enable with FAST_PREDICTOR_TRACE=2) ---
            try:
                import os
                if os.environ.get('FAST_PREDICTOR_TRACE', '0') == '2':
                    # try to fetch last DB close to compare
                    last_db_price = None
                    try:
                        from .data.database import get_db_path
                        import sqlite3
                        dbp = get_db_path()
                        conn = sqlite3.connect(dbp)
                        cur = conn.cursor()
                        cur.execute("SELECT close_price FROM crypto_prices WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1", (symbol,))
                        r = cur.fetchone()
                        conn.close()
                        if r:
                            last_db_price = float(r[0])
                    except Exception:
                        last_db_price = None

                    # Prepare per-model preds (guarded in case some weren't computed)
                    lgbm_val = locals().get('lgbm_pred', None)
                    ridge_val = locals().get('ridge_pred', None)
                    huber_val = locals().get('huber_pred', None)
                    gru_val = locals().get('gru_pred', None)

                    # Show compact diagnostics
                    try:
                        raw_feats = features.to_dict(orient='records')[0] if hasattr(features, 'to_dict') else str(features)
                    except Exception:
                        raw_feats = str(features)
                    try:
                        scaled_feats = features_scaled.tolist()[0] if hasattr(features_scaled, 'tolist') else str(features_scaled)
                    except Exception:
                        scaled_feats = str(features_scaled)

                    diff_pct = None
                    try:
                        diff_pct = ((final_prediction - current_price) / current_price) * 100 if current_price else None
                    except Exception:
                        diff_pct = None

                    print(f"[FAST_PREDICTOR_VERBOSE] symbol={symbol} current_price={current_price} last_db={last_db_price} model_type={model_info.get('model_type')} trained_at={model_info.get('trained_at')} training_samples={model_info.get('training_samples')}")
                    print(f"[FAST_PREDICTOR_VERBOSE] raw_features={raw_feats}")
                    print(f"[FAST_PREDICTOR_VERBOSE] scaled_features={scaled_feats}")
                    try:
                        print(f"[FAST_PREDICTOR_VERBOSE] feat_order_saved_len={len(feat_order_saved) if feat_order_saved is not None else None} runtime_cols={len(features.columns)}")
                    except Exception:
                        pass
                    print(f"[FAST_PREDICTOR_VERBOSE] preds -> lgbm={lgbm_val} ridge={ridge_val} huber={huber_val} gru={gru_val}")
                    print(f"[FAST_PREDICTOR_VERBOSE] final_pred={final_prediction} diff_pct={diff_pct}")
                    
                    # show scaler distribution to detect training-vs-runtime mismatch
                    try:
                        # scaler is from model_info['scaler']
                        scaler_mean = getattr(scaler, "mean_", None)
                        scaler_scale = getattr(scaler, "scale_", None)
                        if scaler_mean is not None:
                            # print compactly
                            print(f"[FAST_PREDICTOR_VERBOSE] scaler_mean={list(scaler_mean)} scaler_scale={list(scaler_scale) if scaler_scale is not None else scaler_scale}")
                    except Exception as _e:
                        print(f"[FAST_PREDICTOR_VERBOSE] scaler diagnostic error: {_e}")
            except Exception as _e:
                print(f"[FAST_PREDICTOR_VERBOSE] diagnostic error: {_e}")
            # --- end diagnostics ---
            
            return final_prediction
            
        except Exception as e:
            # print(f"[FAST_PREDICTOR] Error in inference for {symbol}: {e}") # REDUCED: Too frequent
            return 0.0
    
    def get_model_info(self, symbol: str) -> Optional[dict]:
        """Get information about the current model for a symbol"""
        model_info = self.model_trainer.get_model_for_symbol(symbol)
        if not model_info:
            return None
        
        return {
            'model_type': model_info['model_type'],
            'trained_at': model_info['trained_at'],
            'training_samples': model_info['training_samples'],
            'is_fresh': self.model_trainer.is_model_fresh(symbol)
        }
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear prediction cache"""
        with self.lock:
            if symbol:
                self.prediction_cache.pop(symbol, None)
            else:
                self.prediction_cache.clear()
        # print(f"[FAST_PREDICTOR] Cache cleared for {symbol or 'all symbols'}") # REDUCED: Too frequent


# Global fast predictor instance
_fast_predictor = None

def get_fast_predictor() -> FastPredictor:
    """Get the global fast predictor instance"""
    global _fast_predictor
    if _fast_predictor is None:
        _fast_predictor = FastPredictor()
    return _fast_predictor
