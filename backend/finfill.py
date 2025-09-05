"""Lightweight utilities for robust fitting used in tests.

Exposes `_robust_std`, `_sanitize_Xy`, `_fit_huber_adaptive`, and `SIGMA_FLOOR`.
Designed to be importable as a flat module with `backend` on `sys.path`.
"""
from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np

# Import config in a flat layout (tests add backend/ to sys.path)
try:
    from config import SIGMA_FLOOR as _SIGMA_FLOOR
except Exception:
    _SIGMA_FLOOR = 0.0015

SIGMA_FLOOR: float = float(_SIGMA_FLOOR)


def _robust_std(x: np.ndarray) -> float:
    """Median absolute deviation scaled to std; floored for stability."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.isfinite(x).any():
        return float(SIGMA_FLOOR)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    s = 1.4826 * float(mad)
    if not np.isfinite(s) or s <= 0:
        return float(SIGMA_FLOOR)
    return max(float(SIGMA_FLOOR), s)


def _sanitize_Xy(X: np.ndarray, y: np.ndarray, var_eps: float = 1e-12) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional["StandardScalerLike"], Optional[List[int]], Optional[str]]:
    """Drop near-constant columns, scale features, and return metadata.

    Returns (Xs, y2, scaler, keep_cols, err). On success, err is None.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        return None, None, None, None, "shape_mismatch"

    # Remove rows with non-finite
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    if X.shape[0] == 0:
        return None, None, None, None, "no_rows"

    # Keep columns with variance above threshold
    col_std = np.nanstd(X, axis=0)
    keep = [j for j, s in enumerate(col_std) if float(s) > var_eps]
    if not keep:
        return None, None, None, None, "no_variance"
    X2 = X[:, keep]

    # Scale features
    try:
        from sklearn.preprocessing import StandardScaler  # type: ignore

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X2)
    except Exception:
        # Minimal scaler fallback
        mu = np.nanmean(X2, axis=0)
        sd = np.nanstd(X2, axis=0)
        sd = np.where(sd <= 0, 1.0, sd)

        class _Scaler:
            def __init__(self, mu, sd):
                self.mean_ = mu
                self.scale_ = sd

            def transform(self, Z):
                return (Z - self.mean_) / self.scale_

        scaler = _Scaler(mu, sd)  # type: ignore
        Xs = (X2 - mu) / sd

    return Xs, y, scaler, keep, None


def _fit_huber_adaptive(X: np.ndarray, y: np.ndarray):
    """Fit a robust linear model; fall back to ridge-like if sklearn missing."""
    Xs, y2, scaler, keep, err = _sanitize_Xy(X, y)
    if err is not None or Xs is None or y2 is None:
        return None, err

    try:
        from sklearn.linear_model import HuberRegressor  # type: ignore

        # Set epsilon guided by robust std of residual scale
        eps = 1.35
        model = HuberRegressor(epsilon=eps)
        model.fit(Xs, y2)
        return model, None
    except Exception:
        # Minimal ridge-like fallback with sklearn-like interface
        class _RidgeLike:
            def __init__(self, alpha: float = 1e-3):
                self.alpha = float(alpha)
                self.coef_ = None
                self.intercept_ = 0.0

            def fit(self, Z, t):
                Z = np.asarray(Z, dtype=float)
                t = np.asarray(t, dtype=float)
                # Add bias term
                Zb = np.c_[Z, np.ones(len(Z))]
                I = np.eye(Zb.shape[1])
                I[-1, -1] = 0.0  # do not regularize bias
                w = np.linalg.pinv(Zb.T @ Zb + self.alpha * I) @ (Zb.T @ t)
                self.coef_ = w[:-1]
                self.intercept_ = w[-1]
                return self

            def predict(self, Z):
                Z = np.asarray(Z, dtype=float)
                return Z @ self.coef_ + self.intercept_

        model = _RidgeLike().fit(Xs, y2)
        return model, None


__all__ = [
    "SIGMA_FLOOR",
    "_robust_std",
    "_sanitize_Xy",
    "_fit_huber_adaptive",
]

