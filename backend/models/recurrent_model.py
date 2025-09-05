"""
GRU (Gated Recurrent Unit) Model for Time Series Forecasting
Implements recurrent neural network models to capture temporal patterns in price data
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import sqlite3
import os

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    HAS_TENSORFLOW = True
except ImportError:
    print("[WARNING] TensorFlow not installed. GRU models will not be available.")
    HAS_TENSORFLOW = False

try:
    # Preferred: package-relative imports when used as part of the `backend` package
    from ..config import HORIZON, TRAIN_WITH_FLOW_FEATS
    from ..data.database import get_db_path
    from ..features.engineering import build_features_vectorized, add_flow_features, add_flow_plus
except Exception:
    # Fallback: allow importing the module when backend/ is added to sys.path
    # (useful for running quick scripts or tests that append 'backend' to sys.path)
    from backend.config import HORIZON, TRAIN_WITH_FLOW_FEATS
    from backend.data.database import get_db_path
    from backend.features.engineering import build_features_vectorized, add_flow_features, add_flow_plus

def create_gru_model(input_shape: Tuple[int, int]) -> Optional[Sequential]:
    """
    Create a GRU model architecture.
    """
    if not HAS_TENSORFLOW:
        return None

    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        GRU(32, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def create_sequences(features: np.ndarray, target: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping sequences from time series data.
    """
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:(i + sequence_length)])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

def prepare_data_for_recurrent_model(symbol: str, sequence_length: int = 60, max_rows: int = 20000) -> Optional[Tuple[np.ndarray, np.ndarray, MinMaxScaler]]:
    """
    Prepare data from the database for the recurrent model.
    """
    if not HAS_TENSORFLOW:
        return None

    try:
        db_path = get_db_path()
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        query = f"""
            WITH ranked AS (
              SELECT
                timestamp, open_price, high_price, low_price, close_price, volume,
                trades, quote_volume, taker_base_volume, taker_quote_volume,
                ROW_NUMBER() OVER (
                  PARTITION BY timestamp ORDER BY (source='FINAZON_REST') DESC, rowid DESC
                ) as rn
              FROM crypto_prices WHERE symbol = ?
            )
            SELECT timestamp, open_price, high_price, low_price, close_price, volume,
                   trades, quote_volume, taker_base_volume, taker_quote_volume
            FROM ranked WHERE rn = 1 ORDER BY timestamp DESC LIMIT ?
        """
        df = pd.read_sql(query, conn, params=(symbol, max_rows))
        conn.close()

        # Normalize column names coming from SQL to match other modules
        # (some callers expect 'open','high','low','close' while SQL returns 'open_price',...)
        col_map = {}
        if 'open_price' in df.columns:
            col_map.update({'open_price': 'open'})
        if 'high_price' in df.columns:
            col_map.update({'high_price': 'high'})
        if 'low_price' in df.columns:
            col_map.update({'low_price': 'low'})
        if 'close_price' in df.columns:
            col_map.update({'close_price': 'close'})
        if col_map:
            df.rename(columns=col_map, inplace=True)

        print(f"[GRU_DATA] Fetched {len(df)} rows for {symbol}.") # Added log
        required_rows = sequence_length + HORIZON + 100 # Define for clarity
        if len(df) < required_rows:
            print(f"[GRU_DATA] Insufficient data for {symbol}: found {len(df)} rows, need {required_rows}.") # Modified log
            return None

        df = df.sort_values('timestamp').reset_index(drop=True)
        df['volume'] = df['volume'].fillna(0)
        df.set_index('timestamp', inplace=True)

        features_df = build_features_vectorized(df['open'], df['high'], df['low'], df['close'], df['volume'])
        if TRAIN_WITH_FLOW_FEATS:
            df_flow = add_flow_features(df)
            flow_cols = [
                'trade_count_1m','avg_trade_size_1m','dollar_volume_1m',
                'taker_buy_ratio','taker_imbalance',
                'ema_vol_5','ema_vol_15','ema_trades_5','ema_trades_15','ema_tbr_5'
            ]
            for col in flow_cols:
                if col in df_flow.columns and col not in features_df.columns:
                    features_df[col] = df_flow[col]
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
                if col in df_ext.columns and col not in features_df.columns:
                    features_df[col] = df_ext[col]

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
        if TRAIN_WITH_FLOW_FEATS:
            ext_order = [
                'trade_count_1m','avg_trade_size_1m','dollar_volume_1m',
                'taker_buy_ratio','taker_imbalance','ema_vol_5','ema_vol_15','ema_trades_5','ema_trades_15','ema_tbr_5',
                'ema_gap_5_15','ema_gap_15_60','pct_up_10','pct_up_30','runlen_up_15','roc_5','bb_width_20','squeeze','donchian_pos_60','range_ratio',
                'RV_60m','rv_term','vol_of_vol_60','ATR_14','ADX_14','trend_state','chop_flag',
                'taker_buy_ratio_1m','tbi_5','tbi_15','tbi_slope','dollar_vol_1m','ema_dv_5','ema_dv_15','burst','tr_rate_1m','ema_tr_5','ema_tr_15','cadence_ratio','ti_5','ti_cross','d_obv_15','flow_price_beta_60','eff_ratio_10','whipsaw_12'
            ]
        else:
            ext_order = []
        feat_order = base_feat_order + ext_order
        features_df['target'] = features_df['close'].shift(-HORIZON)
        features_df = features_df.dropna()

        # Ensure all features are present and get target
        target = features_df['target'].values
        features_df = features_df[feat_order]

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features_df)

        X, y = create_sequences(scaled_features, target, sequence_length)
        print(f"[GRU_DATA] Created {len(X)} sequences for {symbol}.") # Added log

        return X, y, scaler

    except Exception as e:
        print(f"[GRU_DATA] Error preparing data for {symbol}: {e}")
        return None

def train_gru_model(X: np.ndarray, y: np.ndarray, symbol: str) -> Optional[Sequential]:
    """
    Train the GRU model.
    """
    if not HAS_TENSORFLOW:
        return None

    try:
        input_shape = (X.shape[1], X.shape[2])
        model = create_gru_model(input_shape)

        if model is None: return None

        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

        # Control verbosity via env var to avoid interleaved progress bars when training concurrently
        verbose = 0
        try:
            verbose = int(os.getenv('GRU_TRAIN_VERBOSE', '2'))
        except Exception:
            verbose = 2
        model.fit(
            X, y,
            epochs=150,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=verbose
        )
        return model
    except Exception as e:
        print(f"[GRU_TRAIN] Error training model for {symbol}: {e}")
        return None
