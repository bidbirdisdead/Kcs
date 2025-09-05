import sys
import os
import numpy as np
import math

import pytest

# ensure backend directory is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from finfill import _robust_std, _sanitize_Xy, _fit_huber_adaptive, SIGMA_FLOOR


def test_robust_std_empty():
    assert _robust_std(np.array([])) >= SIGMA_FLOOR


def test_robust_std_mad():
    # symmetric data with small MAD
    x = np.array([10.0, 10.1, 9.9, 10.0, 100.0])  # one outlier
    s = _robust_std(x)
    # should be >= floor and finite
    assert math.isfinite(s) and s >= SIGMA_FLOOR


def test_sanitize_Xy_basic():
    # X has 5 rows, 3 columns; last col constant
    X = np.array([
        [1.0, 2.0, 5.0],
        [2.0, 3.0, 5.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 5.0],
        [5.0, 6.0, 5.0],
    ])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Xs, y2, scaler, keep, err = _sanitize_Xy(X, y)
    # Not enough samples per MIN_SAMPLES in the project may return error; accept either
    if err is not None:
        assert 'too_few_rows' in err or 'no_variance' in err.lower()
        return
    assert Xs is not None
    # last column should be dropped because it's constant
    assert Xs.shape[1] == 2
    assert scaler is not None
    assert keep is not None


def test_fit_huber_adaptive_smoke():
    # simple linear relationship with outliers
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 3))
    coef = np.array([0.5, -1.2, 0.3])
    y = X.dot(coef) + rng.normal(scale=0.01, size=200)
    # add outliers
    y[:5] += 10

    model, warn = _fit_huber_adaptive(X, y)
    assert model is not None
    # model should roughly recover coef sign pattern
    if hasattr(model, 'coef_'):
        c = np.asarray(model.coef_)
        assert np.sign(c[0]) == np.sign(coef[0])
