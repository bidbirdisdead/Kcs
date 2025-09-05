import os
import numpy as np
import pandas as pd
from .extended import build_flow_plus

def _safe_divide(numer: pd.Series, denom: pd.Series, eps: float = 1e-12, default: float = 0.0) -> pd.Series:
    d = denom.copy()
    d = d.where(np.abs(d) > eps, np.nan)
    out = numer / d
    return out.fillna(default)


def build_features_vectorized(o, h, l, c, v):
    """Builds a consistent set of technical indicators on OHLCV series.

    - Ensures index is monotonically increasing (sorts when needed)
    - Handles divide-by-zero cases robustly (Bollinger, Donchian, etc.)
    - Clips indicators to reasonable numeric ranges and fills NaNs where appropriate
    """
    df = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})

    # Ensure monotonic increasing index (many callers already sorted by timestamp)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    
    # Basic hygiene
    df['volume'] = df['volume'].fillna(0)
    # Guard against obviously bad bars
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    

    # EMAs
    for span in [5, 15, 30, 60]:
        df[f'ema{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        df[f'gap_ema{span}'] = df['close'] - df[f'ema{span}']

    # MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_sig']

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    ma20 = df['close'].rolling(window=20).mean()
    sd20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = ma20 + (sd20 * 2)
    df['bb_lower'] = ma20 - (sd20 * 2)
    bb_width = (df['bb_upper'] - df['bb_lower'])
    df['bb_pctB'] = _safe_divide((df['close'] - df['bb_lower']), bb_width, default=0.5)

    # Realized Volatility
    ret = df['close'].pct_change()
    df['rv5'] = ret.rolling(window=5).std()
    df['rv15'] = ret.rolling(window=15).std()
    df['rv60'] = ret.rolling(window=60).std()

    # VWAP
    tp = (df['high'] + df['low'] + df['close']) / 3
    vol_sum = df['volume'].rolling(window=60).sum()
    vwap_num = (tp * df['volume']).rolling(window=60).sum()
    vwap = _safe_divide(vwap_num, vol_sum, default=df['close'])
    df['vwap_gap'] = df['close'] - vwap

    # Donchian Channel
    donch_hi = df['high'].rolling(window=20).max()
    donch_lo = df['low'].rolling(window=20).min()
    donch_width = (donch_hi - donch_lo)
    df['donch20'] = _safe_divide((df['close'] - donch_lo), donch_width, default=0.5)

    # Time-based features
    df['timestamp_dt'] = pd.to_datetime(df.index, unit='s')
    df['min_of_hour_sin'] = np.sin((df['timestamp_dt'].dt.minute/60)*2*np.pi)
    df['min_of_hour_cos'] = np.cos((df['timestamp_dt'].dt.minute/60)*2*np.pi)

    # ATR (Average True Range)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr14'] = tr.ewm(span=14, adjust=False).mean()

    # Stochastic Oscillator
    low14 = df['low'].rolling(window=14).min()
    high14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - low14) / (high14 - low14))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Average Directional Index (ADX)
    plus_dm = df['high'].diff()
    minus_dm = (df['low'].diff() * -1)
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm >= minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm >= plus_dm), 0)
    tr_adx = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_adx = tr_adx.ewm(span=14, adjust=False).mean()
    plus_di = 100 * _safe_divide(plus_dm.ewm(span=14, adjust=False).mean(), atr_adx)
    minus_di = 100 * _safe_divide(minus_dm.ewm(span=14, adjust=False).mean(), atr_adx)
    denom_di = (plus_di + minus_di)
    dx = 100 * _safe_divide((plus_di - minus_di).abs(), denom_di)
    df['adx'] = dx.ewm(span=14, adjust=False).mean()

    # Ichimoku Cloud Components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    df['ichimoku_tenkan'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    df['ichimoku_kijun'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead.
    df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead.
    df['ichimoku_senkou_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)

    # Williams %R
    highest_high = df['high'].rolling(window=14).max()
    lowest_low = df['low'].rolling(window=14).min()
    denom_wr = (highest_high - lowest_low)
    df['williams_r'] = -100 * _safe_divide((highest_high - df['close']), denom_wr, default=0.0)

    # Clip/clean some known bounded indicators
    df['rsi14'] = df['rsi14'].clip(lower=0, upper=100)
    df['bb_pctB'] = df['bb_pctB'].clip(lower=0, upper=1)
    df['donch20'] = df['donch20'].clip(lower=0, upper=1)

    # Optional diagnostics
    if os.environ.get('FEATURES_TRACE', '0') == '1':
        try:
            msg = {
                'rows': int(df.shape[0]),
                'nans': int(df.isna().sum().sum()),
                'start_ts': int(df.index[0]) if len(df.index) else None,
                'end_ts': int(df.index[-1]) if len(df.index) else None,
            }
            print(f"[FEATURES_TRACE] {msg}")
        except Exception:
            pass

    return df


def add_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Augment a DataFrame (indexed by timestamp) with engineered flow features.

    Expects columns when available (tolerant to missing):
      - close, volume (base volume), trades, quote_volume, taker_base_volume, taker_quote_volume
    Adds columns (NaN-safe, clipped):
      - trade_count_1m, avg_trade_size_1m, dollar_volume_1m
      - taker_buy_ratio, taker_imbalance
      - ema_vol_5, ema_vol_15, ema_trades_5, ema_trades_15, ema_tbr_5
    """
    out = df.copy()
    close = out.get('close')
    vol = out.get('volume')
    trades = out.get('trades')
    qv = out.get('quote_volume')
    tbv = out.get('taker_base_volume')

    # Base definitions with safe fallbacks
    trade_count = pd.to_numeric(trades, errors='coerce') if trades is not None else pd.Series(index=out.index, dtype=float)
    base_vol = pd.to_numeric(vol, errors='coerce') if vol is not None else pd.Series(index=out.index, dtype=float)
    price = pd.to_numeric(close, errors='coerce') if close is not None else pd.Series(index=out.index, dtype=float)
    quote_vol = pd.to_numeric(qv, errors='coerce') if qv is not None else (price * base_vol)
    taker_b = pd.to_numeric(tbv, errors='coerce') if tbv is not None else pd.Series(index=out.index, dtype=float)

    eps = 1e-12
    out['trade_count_1m'] = trade_count.fillna(0)
    out['avg_trade_size_1m'] = (base_vol / (trade_count.replace(0, np.nan))).fillna(0)
    out['dollar_volume_1m'] = quote_vol.fillna(price * base_vol)

    tbr = (taker_b / base_vol.replace(0, np.nan)).clip(lower=0, upper=1)
    out['taker_buy_ratio'] = tbr.fillna(0)
    out['taker_imbalance'] = (2 * out['taker_buy_ratio'] - 1).clip(lower=-1, upper=1)

    # EMAs for short and medium context
    out['ema_vol_5'] = base_vol.ewm(span=5, adjust=False).mean().fillna(0)
    out['ema_vol_15'] = base_vol.ewm(span=15, adjust=False).mean().fillna(0)
    out['ema_trades_5'] = out['trade_count_1m'].ewm(span=5, adjust=False).mean()
    out['ema_trades_15'] = out['trade_count_1m'].ewm(span=15, adjust=False).mean()
    out['ema_tbr_5'] = out['taker_buy_ratio'].ewm(span=5, adjust=False).mean()

    return out


def add_flow_plus(df: pd.DataFrame) -> pd.DataFrame:
    """Extended feature bundle using OHLCV + flow.

    Returns a DataFrame with only the new columns; safe to merge into the
    base feature frame. Does not duplicate columns already present.
    """
    raw = pd.DataFrame(index=df.index)
    raw['open'] = df['open']
    raw['high'] = df['high']
    raw['low'] = df['low']
    raw['close'] = df['close']
    raw['volume'] = df.get('volume')
    raw['trades'] = df.get('trades')
    raw['quote_volume'] = df.get('quote_volume')
    raw['taker_base_volume'] = df.get('taker_base_volume')
    raw['taker_quote_volume'] = df.get('taker_quote_volume')
    ext = build_flow_plus(raw)
    return ext
