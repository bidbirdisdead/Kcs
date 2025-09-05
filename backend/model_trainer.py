"""
Background Model Training Pipeline - Separated from Real-Time Inference
Trains LightGBM models periodically with large datasets for maximum accuracy
"""
import threading
import time
import pickle
import os
from datetime import datetime
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sqlite3
from pathlib import Path
import queue

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    print("[WARNING] LightGBM not installed. Using Ridge regression only.")
    HAS_LIGHTGBM = False
    from sklearn.linear_model import Ridge

# Import GRU model functionality
try:
    from .models.recurrent_model import prepare_data_for_recurrent_model, train_gru_model, HAS_TENSORFLOW
    HAS_GRU = HAS_TENSORFLOW
except ImportError:
    print("[WARNING] GRU model imports failed. GRU models will not be available.")
    HAS_GRU = False

from .config import TRAIN_WINDOW_M, HORIZON, LGBM_N_ESTIMATORS, LGBM_EARLY_STOPPING_ROUNDS, LGBM_VALIDATION_RATIO, TRAIN_WITH_FLOW_FEATS
from .data.database import get_db_path
from .features.engineering import build_features_vectorized, add_flow_features, add_flow_plus

class ModelTrainer:
    """Background model training pipeline - separated from real-time inference"""

    def __init__(self, training_interval_minutes=10):
        # Persist models next to this module to avoid CWD-dependent paths
        self.models_dir = (Path(__file__).parent / "models").resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.trained_models = {}  # symbol -> model info
        self.active_models = {}   # symbol -> model info (pinned for a freeze window)
        self.last_deploy_at = {}  # symbol -> datetime of last promotion
        try:
            self.freeze_seconds = int(os.getenv("MODEL_FREEZE_MINUTES", "10")) * 60
        except Exception:
            self.freeze_seconds = 600
        self.training_interval = training_interval_minutes * 60  # seconds
        self.is_running = False
        self.training_thread = None
        self.retrain_queue = queue.Queue()

        self.max_training_samples = 600000
        self.min_training_samples = 2000
        # Per-symbol training guards to prevent duplicate concurrent runs
        self._symbol_locks: dict[str, threading.Lock] = {}
        self._last_train_started_at: dict[str, float] = {}
        # Guards for concurrent model activation/promotion to avoid duplicate loads/prints
        self._activation_locks: dict[str, threading.Lock] = {}
        # Global lock to serialize GRU data prep and training across symbols
        self._gru_train_lock = threading.Lock()

        print(f"[MODEL_TRAINER] Initialized with {training_interval_minutes}min intervals")
        print(f"[MODEL_TRAINER] Models directory: {self.models_dir}")

    def start_background_training(self):
        """Start the background training pipeline"""
        if self.is_running:
            return

        self.is_running = True
        self.training_thread = threading.Thread(
            target=self._background_training_loop,
            daemon=True,
            name="ModelTrainer"
        )
        self.training_thread.start()
        print("[MODEL_TRAINER] Background training pipeline started")

    def stop_background_training(self):
        """Stop the background training pipeline"""
        self.is_running = False
        if self.training_thread:
            self.training_thread.join(timeout=5)
        print("[MODEL_TRAINER] Background training pipeline stopped")

    def trigger_retrain(self, symbol: str):
        """Trigger an immediate retrain for a symbol."""
        if not self.retrain_queue.full():
            if not self._can_schedule(symbol):
                print(f"[MODEL_TRAINER] {symbol}: Retrain request ignored (recently started or in progress)")
                return
            print(f"[MODEL_TRAINER] Queuing immediate retrain for {symbol}")
            self.retrain_queue.put(symbol)
            self._process_pending_retrains()

    def _process_pending_retrains(self):
        """Process any pending retrains immediately"""
        try:
            while not self.retrain_queue.empty():
                symbol = self.retrain_queue.get_nowait()
                print(f"[MODEL_TRAINER] Processing immediate retrain for {symbol}")
                self._train_model_for_symbol(symbol)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[MODEL_TRAINER] Error processing pending retrains: {e}")

    def _background_training_loop(self):
        """Main background training loop - runs independently"""
        symbols = ["BTC/USDT", "ETH/USDT"]

        while self.is_running:
            try:
                # Process all pending retrains from the queue first
                while not self.retrain_queue.empty():
                    try:
                        symbol_to_retrain = self.retrain_queue.get_nowait()
                        print(f"[MODEL_TRAINER] High-priority retrain for {symbol_to_retrain}")
                        self._train_model_for_symbol(symbol_to_retrain)
                    except queue.Empty:
                        break # Queue is empty

                # Then, perform periodic training for stale models
                for symbol in symbols:
                    if not self.is_running:
                        break

                    if (not self.is_model_fresh(symbol)) and self._can_schedule(symbol):
                        print(f"[MODEL_TRAINER] Training model for {symbol} (stale or not found)")
                        self._train_model_for_symbol(symbol)
                    else:
                        print(f"[MODEL_TRAINER] {symbol}: Model is fresh, skipping periodic training.")

                for _ in range(self.training_interval):
                    if not self.is_running:
                        break
                    time.sleep(1)

            except Exception as e:
                print(f"[MODEL_TRAINER] Error in training loop: {e}")
                time.sleep(30)

    def _train_model_for_symbol(self, symbol: str):
        """Train a model for specific symbol with large dataset"""
        lock = self._symbol_locks.setdefault(symbol, threading.Lock())
        acquired = lock.acquire(blocking=False)
        if not acquired:
            print(f"[MODEL_TRAINER] {symbol}: Training already in progress; skipping duplicate call")
            return
        self._last_train_started_at[symbol] = time.time()
        try:
            training_data = self._get_training_data(symbol)
            if training_data is None:
                return

            X, y, scaler, feat_order = training_data

            if len(X) < self.min_training_samples:
                print(f"[MODEL_TRAINER] {symbol}: Not enough data ({len(X)} samples)")
                return

            trained_models = self._train_all_models(X, y, symbol)

            if not any(trained_models.values()):
                print(f"[MODEL_TRAINER] {symbol}: No models trained successfully")
                return

            model_info = {
                'lgbm_model': trained_models.get('lgbm'),
                'ridge_model': trained_models.get('ridge'),
                'huber_model': trained_models.get('huber'),
                'gru_model': trained_models.get('gru'),
                'gru_scaler': trained_models.get('gru_scaler'),
                'scaler': scaler,
                'feat_order': feat_order,
                'model_type': self._determine_primary_model_type(trained_models),
                'trained_at': datetime.now(),
                'training_samples': len(X),
                'symbol': symbol
            }

            self.trained_models[symbol] = model_info
            self._save_model_to_disk(symbol, model_info)

            try:
                from .predictor import get_fast_predictor
                predictor = get_fast_predictor()
                predictor.clear_cache(symbol)
                print(f"[MODEL_TRAINER] {symbol}: Cleared prediction cache for fresh model")
            except Exception as e:
                print(f"[MODEL_TRAINER] {symbol}: Warning - could not clear prediction cache: {e}")

            print(f"[MODEL_TRAINER] {symbol}: Trained models with {len(X)} samples")

        except Exception as e:
            print(f"[MODEL_TRAINER] Error training {symbol}: {e}")
        finally:
            try:
                lock.release()
            except Exception:
                pass

    def _can_schedule(self, symbol: str, debounce_sec: int = 120) -> bool:
        """Return True if a new training job can be scheduled for symbol.

        Prevents duplicate scheduling when a job is already running or started
        very recently (debounce window).
        """
        lock = self._symbol_locks.get(symbol)
        if lock is not None and lock.locked():
            return False
        last = self._last_train_started_at.get(symbol)
        if last is not None and (time.time() - last) < debounce_sec:
            return False
        return True

    def _train_all_models(self, X, y, symbol):
        """Train all available models with error isolation"""
        trained_models = {}

        if HAS_LIGHTGBM and len(X) >= 1000:
            try:
                trained_models['lgbm'] = self._train_lightgbm(X, y, symbol)
                print(f"[MODEL_TRAINER] {symbol}: LightGBM trained successfully")
            except Exception as e:
                print(f"[MODEL_TRAINER] {symbol}: LightGBM training failed: {e}")
                trained_models['lgbm'] = None

        try:
            from sklearn.linear_model import Ridge
            trained_models['ridge'] = Ridge(alpha=3.0).fit(X, y)
            print(f"[MODEL_TRAINER] {symbol}: Ridge trained successfully")
        except Exception as e:
            print(f"[MODEL_TRAINER] {symbol}: Ridge training failed: {e}")
            trained_models['ridge'] = None

        try:
            from sklearn.linear_model import HuberRegressor
            trained_models['huber'] = HuberRegressor().fit(X, y)
            print(f"[MODEL_TRAINER] {symbol}: Huber trained successfully")
        except Exception as e:
            print(f"[MODEL_TRAINER] {symbol}: Huber training failed: {e}")
            trained_models['huber'] = None

        # Train GRU model with separate data preparation
        if HAS_GRU and len(X) >= 500:
            try:
                trained_models['gru'], trained_models['gru_scaler'] = self._train_gru_model(symbol)
                if trained_models['gru'] is not None:
                    print(f"[MODEL_TRAINER] {symbol}: GRU trained successfully")
                else:
                    print(f"[MODEL_TRAINER] {symbol}: GRU training returned None")
                    trained_models['gru'] = None
                    trained_models['gru_scaler'] = None
            except Exception as e:
                print(f"[MODEL_TRAINER] {symbol}: GRU training failed: {e}")
                trained_models['gru'] = None
                trained_models['gru_scaler'] = None

        return trained_models

    def _determine_primary_model_type(self, trained_models):
        """Determine the primary model type for reporting"""
        if trained_models.get('gru'):
            return "gru+hybrid"
        elif trained_models.get('lgbm'):
            return "lightgbm"
        elif trained_models.get('ridge'):
            return "ridge"
        elif trained_models.get('huber'):
            return "huber"
        else:
            return "none"

    def _get_training_data(self, symbol: str):
        """Get training data for the symbol - large dataset for accuracy"""
        try:
            max_rows = min(TRAIN_WINDOW_M * 2, self.max_training_samples)

            db_path = get_db_path()
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()

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
            """, (symbol, max_rows))

            rows = cur.fetchall()
            conn.close()

            if not rows or len(rows) < (HORIZON + 100):
                return None

            rows.reverse()
            df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                             'trades', 'quote_volume', 'taker_base_volume', 'taker_quote_volume'])
            df['volume'] = df['volume'].fillna(0)
            df.set_index('timestamp', inplace=True)

            features_df = build_features_vectorized(df['open'], df['high'], df['low'], df['close'], df['volume'])
            flow_order = []
            ext_order = []
            if TRAIN_WITH_FLOW_FEATS:
                # Base flow bundle
                df_flow = add_flow_features(df)
                flow_cols = [
                    'trade_count_1m','avg_trade_size_1m','dollar_volume_1m',
                    'taker_buy_ratio','taker_imbalance',
                    'ema_vol_5','ema_vol_15','ema_trades_5','ema_trades_15','ema_tbr_5'
                ]
                for col in flow_cols:
                    if col in df_flow.columns and col not in features_df.columns:
                        features_df[col] = df_flow[col]
                flow_order = [c for c in flow_cols if c in features_df.columns]

                # Extended bundle (Top-25+)
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
                    if col in df_ext.columns and col not in features_df.columns:
                        features_df[col] = df_ext[col]
                ext_order = [c for c in ext_cols if c in features_df.columns]

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
            feat_order = base_feat_order + flow_order + ext_order
            # De-duplicate while preserving order (avoid duplicates like 'taker_imbalance')
            seen = set()
            dedup_order = []
            for col in feat_order:
                if col not in seen:
                    seen.add(col)
                    dedup_order.append(col)
            feat_order = dedup_order

            features_df['y'] = features_df['close'].shift(-HORIZON)
            features_df = features_df.dropna()

            if features_df.shape[0] < self.min_training_samples:
                print(f"[MODEL_TRAINER] {symbol}: Not enough data after feature engineering ({features_df.shape[0]} samples)")
                return None

            X = features_df[feat_order].values
            y = features_df['y'].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            return X_scaled, y, scaler, feat_order

        except Exception as e:
            print(f"[MODEL_TRAINER] Error getting training data for {symbol}: {e}")
            return None

    def get_training_data_stats(self, symbol: str) -> dict:
        """Return diagnostic counts about training data for a symbol.

        Returns a dict with:
          - total_rows: total rows present for symbol
          - per_source: dict mapping source->count
          - unique_timestamps: rows after dedup by timestamp (same logic as training)
          - final_feature_rows: rows after feature engineering and target shift
        """
        stats = {
            'total_rows': 0,
            'per_source': {},
            'unique_timestamps': 0,
            'final_feature_rows': 0,
        }

        try:
            max_rows = min(TRAIN_WINDOW_M * 2, self.max_training_samples)
            db_path = get_db_path()
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()

            # Total rows
            cur.execute("SELECT COUNT(*) FROM crypto_prices WHERE symbol = ?", (symbol,))
            stats['total_rows'] = int(cur.fetchone()[0] or 0)

            # Per-source breakdown
            cur.execute("SELECT source, COUNT(*) FROM crypto_prices WHERE symbol = ? GROUP BY source", (symbol,))
            for src, cnt in cur.fetchall():
                stats['per_source'][src] = int(cnt)

            # Unique timestamps after dedup (same logic as _get_training_data)
            cur.execute("""
                WITH ranked AS (
                  SELECT
                    timestamp, source,
                    ROW_NUMBER() OVER (
                      PARTITION BY timestamp
                      ORDER BY (source='FINAZON_REST') DESC, rowid DESC
                    ) rn
                  FROM crypto_prices
                  WHERE symbol = ?
                )
                SELECT COUNT(*) FROM ranked WHERE rn = 1
            """, (symbol,))
            stats['unique_timestamps'] = int(cur.fetchone()[0] or 0)

            # Now compute final feature rows by re-running the feature pipeline on limited recent rows
            cur.execute("""
                WITH ranked AS (
                  SELECT
                    timestamp, open_price, high_price, low_price, close_price, volume,
                    ROW_NUMBER() OVER (
                      PARTITION BY timestamp
                      ORDER BY (source='FINAZON_REST') DESC, rowid DESC
                    ) rn
                  FROM crypto_prices
                  WHERE symbol = ?
                )
                SELECT timestamp, open_price, high_price, low_price, close_price, volume
                FROM ranked
                WHERE rn = 1
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, max_rows))

            rows = cur.fetchall()
            conn.close()

            if rows:
                rows.reverse()
                import pandas as _pd
                df = _pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['volume'] = df['volume'].fillna(0)
                df.set_index('timestamp', inplace=True)
                features_df = build_features_vectorized(df['open'], df['high'], df['low'], df['close'], df['volume'])
                features_df['y'] = features_df['close'].shift(-HORIZON)
                features_df = features_df.dropna()
                stats['final_feature_rows'] = int(features_df.shape[0])

        except Exception as e:
            print(f"[MODEL_TRAINER] Error computing training data stats for {symbol}: {e}")

        return stats



    def _train_lightgbm(self, X, y, symbol):
        """Train LightGBM model"""
        # Use config-driven n_estimators to allow increasing iterations safely
        n_estimators = int(LGBM_N_ESTIMATORS)

        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )

        # Optionally create a small validation split from the most recent rows
        val_ratio = float(LGBM_VALIDATION_RATIO)
        es_rounds = int(LGBM_EARLY_STOPPING_ROUNDS)

        if val_ratio > 0 and es_rounds > 0 and len(X) >= 1000:
            # keep the last val_size samples for validation (time-ordered split)
            val_size = max(int(len(X) * val_ratio), 50)
            train_X, train_y = X[:-val_size], y[:-val_size]
            val_X, val_y = X[-val_size:], y[-val_size:]

            # fit with eval set and early stopping
            # Use callback-based early stopping to avoid signature issues
            callbacks = [lgb.early_stopping(es_rounds)]
            model.fit(
                train_X, train_y,
                eval_set=[(val_X, val_y)],
                eval_metric='l2',
                callbacks=callbacks
            )
            # capture best iteration when available
            best_iter = getattr(model, 'best_iteration_', None)
            if best_iter is None:
                try:
                    booster = getattr(model, 'booster_', None)
                    best_iter = getattr(booster, 'best_iteration', None)
                except Exception:
                    best_iter = None
        else:
            # no early stopping / validation — full fit
            model.fit(X, y)
            best_iter = getattr(model, 'best_iteration_', None)

        # attach best iteration to model for later use
        try:
            setattr(model, '_best_iteration', best_iter)
        except Exception:
            pass

        return model

    def _train_gru_model(self, symbol: str):
        """Train GRU model with separate data preparation for time series sequences"""
        if not HAS_GRU:
            return None, None

        try:
            # Serialize GRU training across symbols to avoid interleaved epochs/logs and resource contention
            with self._gru_train_lock:
                print(f"[MODEL_TRAINER] {symbol}: Attempting to prepare GRU data.")
                # Prepare data specifically for GRU (creates sequences)
                gru_data = prepare_data_for_recurrent_model(symbol, sequence_length=120, max_rows=100500)

                print(f"[MODEL_TRAINER] {symbol}: GRU data preparation result: {gru_data is not None}.")
                if gru_data is None:
                    print(f"[MODEL_TRAINER] {symbol}: Could not prepare GRU training data")
                    return None, None

                X_gru, y_gru, gru_scaler = gru_data

                print(f"[MODEL_TRAINER] {symbol}: Number of GRU sequences: {len(X_gru)}. Minimum required: 100.")
                if len(X_gru) < 100:
                    print(f"[MODEL_TRAINER] {symbol}: Insufficient GRU training data ({len(X_gru)} sequences)")
                    return None, None

                # Train the GRU model (train_gru_model signature is: X, y, symbol)
                gru_model = train_gru_model(X_gru, y_gru, symbol)

                if gru_model is None:
                    print(f"[MODEL_TRAINER] {symbol}: GRU model training returned None")
                    return None, None

                return gru_model, gru_scaler
        except Exception as e:
            print(f"[MODEL_TRAINER] {symbol}: Error in GRU training: {e}")
            return None, None

    def _save_model_to_disk(self, symbol: str, model_info: dict):
        """Save trained model to disk with proper Keras model serialization"""
        try:
            filename = f"{symbol.replace('/', '_')}_model.pkl"
            filepath = self.models_dir / filename
            tmp_filepath = self.models_dir / (filename + ".tmp")

            # Handle GRU model separately - use Keras native serialization
            if 'gru_model' in model_info and model_info['gru_model'] is not None and HAS_GRU:
                try:
                    # Define paths for old (directory) and new (.keras file) formats
                    gru_model_dir_path = self.models_dir / f"{symbol.replace('/', '_')}_gru_model"
                    gru_model_keras_path = self.models_dir / f"{symbol.replace('/', '_')}_gru_model.keras"
                    print(f"[MODEL_TRAINER] Attempting to save GRU model to: {gru_model_keras_path}")

                    # Clean up old directory-based model if it exists
                    if gru_model_dir_path.exists() and gru_model_dir_path.is_dir():
                        import shutil
                        shutil.rmtree(gru_model_dir_path)
                        print(f"[MODEL_TRAINER] Removed old GRU model directory for {symbol}")

                    # Save Keras model using the recommended .keras format via temp path then atomic replace
                    tmp_keras_path = self.models_dir / f"{symbol.replace('/', '_')}_gru_model.tmp.keras"
                    model_info['gru_model'].save(tmp_keras_path)
                    import os as _os
                    if _os.path.exists(gru_model_keras_path):
                        _os.replace(tmp_keras_path, gru_model_keras_path)
                    else:
                        _os.rename(tmp_keras_path, gru_model_keras_path)
                    print(f"[MODEL_TRAINER] Successfully saved GRU model to: {gru_model_keras_path}")

                    # Replace the model object with the path for pickle serialization
                    model_info = model_info.copy()  # Don't modify original
                    model_info['gru_model_path'] = str(gru_model_keras_path) # Use new path
                    model_info['gru_model'] = None  # Remove the model object

                except Exception as e:
                    print(f"[MODEL_TRAINER] Failed to save GRU model for {symbol}: {e}")
                    model_info = model_info.copy()
                    model_info['gru_model'] = None
                    model_info['gru_model_path'] = None

            print(f"[MODEL_TRAINER] Attempting to save main model info to: {filepath}")
            with open(tmp_filepath, 'wb') as f:
                pickle.dump(model_info, f)
            import os as _os
            if _os.path.exists(filepath):
                _os.replace(tmp_filepath, filepath)
            else:
                _os.rename(tmp_filepath, filepath)
            print(f"[MODEL_TRAINER] Successfully saved main model info to: {filepath}")

        except Exception as e:
            print(f"[MODEL_TRAINER] Error saving model for {symbol}: {e}")

    def load_model_from_disk(self, symbol: str, load_gru_model: bool = True) -> Optional[dict]:
        """Load trained model from disk.

        Args:
            symbol: Trading symbol like 'BTC/USDT'.
            load_gru_model: When False, do not load the heavy Keras model from disk; only return
                            metadata and non-deep models. Use True when promoting/activating.
        """
        try:
            filename = f"{symbol.replace('/', '_')}_model.pkl"
            filepath = self.models_dir / filename

            if not filepath.exists():
                return None

            with open(filepath, 'rb') as f:
                model_info = pickle.load(f)

            # Ensure gru_model is None initially unless loaded properly
            model_info['gru_model'] = None

            # Handle GRU model loading from a path (optional heavy load)
            if load_gru_model and 'gru_model_path' in model_info and model_info['gru_model_path'] and HAS_GRU:
                try:
                    gru_model_path = Path(model_info['gru_model_path'])
                    if gru_model_path.exists():
                        import tensorflow as tf
                        model_info['gru_model'] = tf.keras.models.load_model(gru_model_path)
                        print(f"[MODEL_TRAINER] Loaded GRU model for {symbol} from {gru_model_path}")
                    else:
                        # Fallback for backward compatibility: check for old directory format if .keras file not found
                        gru_model_dir_path = self.models_dir / f"{symbol.replace('/', '_')}_gru_model"
                        if gru_model_dir_path.exists() and gru_model_dir_path.is_dir():
                            import tensorflow as tf
                            model_info['gru_model'] = tf.keras.models.load_model(gru_model_dir_path)
                            print(f"[MODEL_TRAINER] Loaded legacy GRU model for {symbol} from directory {gru_model_dir_path}")
                        else:
                            print(f"[MODEL_TRAINER] GRU model path {gru_model_path} not found for {symbol}")

                except Exception as e:
                    print(f"[MODEL_TRAINER] Failed to load GRU model for {symbol}: {e}")
                    model_info['gru_model'] = None # Ensure it's None on failure

            # This removes the dangerous attempt to use a pickled Keras model object.
            # If a 'gru_model' key exists from an old pickle file, it's ignored,
            # and only a model loaded from a path is considered valid.

            return model_info

        except Exception as e:
            print(f"[MODEL_TRAINER] Error loading model for {symbol}: {e}")
            return None

    def _should_promote(self, symbol: str, candidate: dict) -> bool:
        """Return True if the candidate model can be promoted to active based on freeze window."""
        last = self.last_deploy_at.get(symbol)
        if last is None:
            return True
        elapsed = (datetime.now() - last).total_seconds()
        return elapsed >= self.freeze_seconds

    def get_model_for_symbol(self, symbol: str) -> Optional[dict]:
        """Return the ACTIVE (possibly pinned) model for symbol; promote after freeze window."""
        # Load most recent model metadata from disk (avoid heavy GRU load here)
        candidate_meta = self.load_model_from_disk(symbol, load_gru_model=False)
        active = self.active_models.get(symbol)

        # Initialize active if missing and a candidate exists
        if active is None and candidate_meta is not None:
            lock = self._activation_locks.setdefault(symbol, threading.Lock())
            with lock:
                # Re-check after acquiring lock to avoid duplicate activation
                active2 = self.active_models.get(symbol)
                if active2 is None:
                    # Load full candidate with GRU only once when activating
                    candidate_full = self.load_model_from_disk(symbol, load_gru_model=True)
                    self.active_models[symbol] = candidate_full
                    self.last_deploy_at[symbol] = datetime.now()
                    try:
                        lgbm_ok = candidate_full.get('lgbm_model') is not None
                        ridge_ok = candidate_full.get('ridge_model') is not None
                        huber_ok = candidate_full.get('huber_model') is not None
                        gru_ok = candidate_full.get('gru_model') is not None
                        scaler = candidate_full.get('scaler')
                        feat_n = getattr(scaler, 'n_features_in_', None)
                        if feat_n is None:
                            try:
                                feat_n = len(getattr(scaler, 'mean_', []))
                            except Exception:
                                feat_n = None
                        print(f"[MODEL_TRAINER] {symbol}: Activated model bundle (lgbm={lgbm_ok}, ridge={ridge_ok}, huber={huber_ok}, gru={gru_ok}, n_features={feat_n})")
                    except Exception:
                        pass
                    return candidate_full
                return active2

        # If both exist and candidate is newer than active, consider promotion
        if active is not None and candidate_meta is not None:
            try:
                if candidate_meta['trained_at'] > active['trained_at'] and self._should_promote(symbol, candidate_meta):
                    lock = self._activation_locks.setdefault(symbol, threading.Lock())
                    with lock:
                        # Re-check under lock before promoting
                        active2 = self.active_models.get(symbol)
                        if candidate_meta['trained_at'] > active2['trained_at'] and self._should_promote(symbol, candidate_meta):
                            print(f"[MODEL_TRAINER] {symbol}: Promoting newer model (freeze window elapsed)")
                            candidate_full = self.load_model_from_disk(symbol, load_gru_model=True)
                            self.active_models[symbol] = candidate_full
                            self.last_deploy_at[symbol] = datetime.now()
                            try:
                                lgbm_ok = candidate_full.get('lgbm_model') is not None
                                ridge_ok = candidate_full.get('ridge_model') is not None
                                huber_ok = candidate_full.get('huber_model') is not None
                                gru_ok = candidate_full.get('gru_model') is not None
                                scaler = candidate_full.get('scaler')
                                feat_n = getattr(scaler, 'n_features_in_', None)
                                if feat_n is None:
                                    try:
                                        feat_n = len(getattr(scaler, 'mean_', []))
                                    except Exception:
                                        feat_n = None
                                print(f"[MODEL_TRAINER] {symbol}: Activated model bundle (lgbm={lgbm_ok}, ridge={ridge_ok}, huber={huber_ok}, gru={gru_ok}, n_features={feat_n})")
                            except Exception:
                                pass
                            return candidate_full
            except Exception:
                pass
            # Otherwise keep current active model
            return self.active_models.get(symbol)

        # Fallbacks
        if active is not None:
            return active
        # If no disk model but we have a trained in-memory model, use it and set active
        in_mem = self.trained_models.get(symbol)
        if in_mem is not None:
            self.active_models[symbol] = in_mem
            self.last_deploy_at[symbol] = datetime.now()
            try:
                lgbm_ok = in_mem.get('lgbm_model') is not None
                ridge_ok = in_mem.get('ridge_model') is not None
                huber_ok = in_mem.get('huber_model') is not None
                gru_ok = in_mem.get('gru_model') is not None
                scaler = in_mem.get('scaler')
                feat_n = getattr(scaler, 'n_features_in_', None)
                if feat_n is None:
                    try:
                        feat_n = len(getattr(scaler, 'mean_', []))
                    except Exception:
                        feat_n = None
                print(f"[MODEL_TRAINER] {symbol}: Activated in-memory model bundle (lgbm={lgbm_ok}, ridge={ridge_ok}, huber={huber_ok}, gru={gru_ok}, n_features={feat_n})")
            except Exception:
                pass
            return in_mem
        return None

    def is_model_ready(self, symbol: str) -> bool:
        """Check if a model is ready and fresh for a symbol"""
        model_info = self.get_model_for_symbol(symbol)
        if not model_info:
            return False

        has_model = (
            model_info.get('lgbm_model') is not None
            or model_info.get('ridge_model') is not None
            or model_info.get('huber_model') is not None
            or model_info.get('gru_model') is not None
        )
        if not has_model:
            return False

        trained_at = model_info.get('trained_at')
        if not trained_at:
            return False
        age_hours = (datetime.now() - trained_at).total_seconds() / 3600
        return age_hours <= 24

    def wait_for_models_ready(self, symbols: list, timeout_seconds: int = 300) -> bool:
        """Wait for models to be ready for all specified symbols"""
        print(f"[MODEL_TRAINER] Waiting for models to be ready for {symbols} (timeout: {timeout_seconds}s)")

        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            all_ready = True
            for symbol in symbols:
                if not self.is_model_ready(symbol):
                    all_ready = False
                    print(f"[MODEL_TRAINER] {symbol}: Model not ready yet, waiting...")
                    break

            if all_ready:
                print(f"[MODEL_TRAINER] All models ready for {symbols}")
                return True

            for symbol in symbols:
                if not self.is_model_ready(symbol):
                    self.trigger_retrain(symbol)

            time.sleep(5)

        print(f"[MODEL_TRAINER] Timeout waiting for models to be ready")
        return False

    def is_model_fresh(self, symbol: str, max_age_hours: int = 24) -> bool:
        """Check if a model is fresh enough (within max_age_hours)"""
        model_info = self.get_model_for_symbol(symbol)
        if not model_info:
            return False

        trained_at = model_info.get('trained_at')
        if not trained_at:
            return False

        age_hours = (datetime.now() - trained_at).total_seconds() / 3600
        return age_hours <= max_age_hours


_model_trainer = None

def get_model_trainer() -> ModelTrainer:
    """Get the global model trainer instance"""
    global _model_trainer
    if _model_trainer is None:
        _model_trainer = ModelTrainer()
        _model_trainer.start_background_training()
    return _model_trainer
