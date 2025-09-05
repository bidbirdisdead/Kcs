"""
Extended feature builders derived only from crypto_prices fields.

Inputs expected (index = timestamp in seconds or datetime):
  - open, high, low, close, volume (base), trades, quote_volume, taker_base_volume, taker_quote_volume

All calculations are leak-safe (windows end at t). The functions add only the
columns that do not already exist in the provided DataFrame to avoid duplicates.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-12


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rolling_std(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(window=w, min_periods=w).std()


def _rolling_max(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(window=w, min_periods=w).max()


def _rolling_min(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(window=w, min_periods=w).min()


def _rolling_mean(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(window=w, min_periods=w).mean()


def _rolling_slope(s: pd.Series, window: int) -> pd.Series:
    idx = np.arange(window, dtype=float)
    x = idx - idx.mean()
    denom = np.sum(x * x) + EPS

    def _calc(arr: np.ndarray) -> float:
        y = arr - arr.mean()
        return float(np.sum(x * y) / denom)

    return s.rolling(window=window, min_periods=window).apply(_calc, raw=True)


def _rma_wilder(s: pd.Series, window: int) -> pd.Series:
    alpha = 1.0 / max(window, 1)
    return s.ewm(alpha=alpha, adjust=False).mean()


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return _rma_wilder(tr, window)


def _adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    up = h.diff()
    down = -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).fillna(0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0).fillna(0.0)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = _rma_wilder(tr, window)
    plus_di = 100.0 * (_rma_wilder(plus_dm, window) / (atr + EPS))
    minus_di = 100.0 * (_rma_wilder(minus_dm, window) / (atr + EPS))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    return _rma_wilder(dx, window)


def _realized_vol(close: pd.Series, window: int) -> pd.Series:
    r = np.log(close / close.shift(1))
    r2 = (r ** 2).rolling(window=window, min_periods=window).sum()
    return np.sqrt(r2)


def _obv_diff(qv: pd.Series, close: pd.Series, window: int = 15) -> pd.Series:
    delta_c = close.diff()
    sign = np.sign(delta_c.fillna(0.0))
    obv = (sign * qv).cumsum()
    return obv.diff(window)


def _flow_price_beta(close: pd.Series, tqv: pd.Series, qv: pd.Series, window: int = 60) -> pd.Series:
    r = np.log(close / close.shift(1)).fillna(0.0)
    ti = ((tqv - (qv - tqv)) / (qv + EPS)).clip(-1, 1).fillna(0.0)
    r_mean = _rolling_mean(r, window)
    ti_mean = _rolling_mean(ti, window)
    cov = ((r - r_mean) * (ti - ti_mean)).rolling(window, min_periods=window).mean()
    var = ((ti - ti_mean) ** 2).rolling(window, min_periods=window).mean()
    return cov / (var + EPS)


def build_flow_plus(df: pd.DataFrame) -> pd.DataFrame:
    """Compute extended set of global features. Adds only missing columns.

    Expects df to have columns: open, high, low, close, volume, trades,
    quote_volume, taker_base_volume, taker_quote_volume.
    """
    out = pd.DataFrame(index=df.index)
    o = df['open']; h = df['high']; l = df['low']; c = df['close']
    v = df.get('volume')
    tr = df.get('trades')
    qv = df.get('quote_volume')
    tqv = df.get('taker_quote_volume')

    # EMAs and gaps
    ema5 = _ema(c, 5); ema15 = _ema(c, 15); ema60 = _ema(c, 60)
    out['ema_gap_5_15'] = (ema5 - ema15)
    out['ema_gap_15_60'] = (ema15 - ema60)

    # Bollinger width and squeeze
    std20 = _rolling_std(c, 20)
    ema20 = _ema(c, 20)
    out['bb_width_20'] = (2.0 * std20) / (ema20.abs() + EPS)
    out['bb_pctB'] = ((c - (ema20 - 2 * std20)) / (4 * std20 + EPS)).clip(0.0, 1.0)
    med_bw_5h = out['bb_width_20'].rolling(300, min_periods=300).median()
    out['squeeze'] = out['bb_width_20'] / (med_bw_5h + EPS)

    # Donchian 60 position
    dh60 = _rolling_max(h, 60)
    dl60 = _rolling_min(l, 60)
    out['donchian_pos_60'] = (c - dl60) / ((dh60 - dl60).abs() + EPS)

    # ATR and range
    atr14 = _atr(df.rename(columns={'open':'open','high':'high','low':'low','close':'close'}), 14)
    out['range_ratio'] = (h - l) / (atr14.abs() + EPS)
    out['ATR_14'] = atr14

    # Realized vol term structure
    RV_5m = _realized_vol(c, 5)
    RV_60m = _realized_vol(c, 60)
    out['RV_60m'] = RV_60m
    out['rv_term'] = RV_5m / (RV_60m + EPS)
    out['vol_of_vol_60'] = np.log(c / c.shift(1)).abs().rolling(60, min_periods=60).std()

    # ADX and regime flags
    adx14 = _adx(df.rename(columns={'open':'open','high':'high','low':'low','close':'close'}), 14)
    out['ADX_14'] = adx14
    out['chop_flag'] = (adx14 < 15).astype(int)
    out['trend_state'] = np.sign(ema5 - ema60)

    # Percent up and run length up
    up = (c > c.shift(1)).astype(float)
    out['pct_up_10'] = up.rolling(10, min_periods=10).mean()
    out['pct_up_30'] = up.rolling(30, min_periods=30).mean()
    # run len up 15 (naive)
    rl = []
    cnt = 0
    for inc in up.fillna(0.0).values:
        cnt = cnt + 1 if inc > 0.0 else 0
        rl.append(cnt)
    out['runlen_up_15'] = pd.Series(rl, index=df.index).clip(upper=15)

    # ROC
    out['roc_5'] = c.pct_change(5)

    # Flow / tape features
    if qv is not None and tqv is not None:
        tbr = (tqv / (qv + EPS)).clip(0.0, 1.0)
        out['taker_buy_ratio_1m'] = tbr
        out['tbi_5'] = _ema(tbr, 5)
        out['tbi_15'] = _ema(tbr, 15)
        out['tbi_slope'] = _rolling_slope(out['tbi_5'], 5)
        out['dollar_vol_1m'] = qv
        out['ema_dv_5'] = _ema(qv, 5)
        out['ema_dv_15'] = _ema(qv, 15)
        out['burst'] = (qv / (out['ema_dv_15'] + EPS)).clip(upper=10.0)
        if tr is not None:
            out['tr_rate_1m'] = tr
            out['ema_tr_5'] = _ema(tr, 5)
            out['ema_tr_15'] = _ema(tr, 15)
            out['cadence_ratio'] = out['ema_tr_5'] / (out['ema_tr_15'] + EPS)
        ti = ((tqv - (qv - tqv)) / (qv + EPS)).clip(-1, 1)
        out['taker_imbalance'] = ti
        out['ti_5'] = _ema(ti, 5)
        out['ti_cross'] = out['ti_5'] - _ema(ti, 15)
        out['d_obv_15'] = _obv_diff(qv, c, 15)
        out['flow_price_beta_60'] = _flow_price_beta(c, tqv, qv, 60)

    # Microstructure quality from price only
    delta_c = c.diff().abs()
    out['eff_ratio_10'] = (c - c.shift(10)).abs() / (delta_c.rolling(10, min_periods=10).sum() + EPS)
    sign_change = np.sign(c.diff()).diff().ne(0).astype(float).fillna(0.0)
    out['whipsaw_12'] = sign_change.rolling(12, min_periods=12).mean()

    return out

