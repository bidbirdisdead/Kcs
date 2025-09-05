import os
import time
import sqlite3

import numpy as np
import pandas as pd


def _init_test_db(db_path: str):
    from backend.data.database import create_crypto_prices_table
    conn = sqlite3.connect(db_path)
    create_crypto_prices_table(conn)
    conn.close()


def _insert_bars(db_path: str, symbol: str, n: int = 200):
    now = int(time.time()) // 60 * 60
    rows = []
    base = 10000.0
    for i in range(n):
        ts = now - (n - i) * 60
        close = base + i * 5.0
        rows.append((symbol, ts, close - 2, close + 2, close - 3, close, 100 + i))
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO crypto_prices
        (symbol, timestamp, open_price, high_price, low_price, close_price, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'TEST')
        """,
        rows,
    )
    conn.commit()
    conn.close()


def test_fast_predictor_prefers_db_features(tmp_path, monkeypatch):
    # Point DB to a temp file
    db_file = tmp_path / "cryptoprice.db"
    monkeypatch.setenv("MODEL_DB", str(tmp_path / "modeldb.sqlite"))

    from backend.data import database as dbmod

    # Monkeypatch get_db_path to our temp path
    monkeypatch.setattr(dbmod, "get_db_path", lambda: str(db_file))

    _init_test_db(str(db_file))
    from backend.config import SYMBOLS
    symbol = SYMBOLS[0]
    _insert_bars(str(db_file), symbol, n=240)

    # Inject a lightweight sklearn shim to satisfy model_trainer imports
    import types, sys
    class _DummyScaler:
        def fit(self, X):
            return self
        def transform(self, X):
            return X
    class _DummyRidge:
        def fit(self, X, y):
            return self
        def predict(self, X):
            return getattr(X, 'mean', lambda : 0.0)() if hasattr(X, 'mean') else 0.0
    skl = types.SimpleNamespace(
        preprocessing=types.SimpleNamespace(StandardScaler=_DummyScaler),
        linear_model=types.SimpleNamespace(Ridge=_DummyRidge),
    )
    sys.modules.setdefault('sklearn', skl)
    sys.modules.setdefault('sklearn.preprocessing', skl.preprocessing)
    sys.modules.setdefault('sklearn.linear_model', skl.linear_model)

    # Stub out recurrent_model to avoid importing TensorFlow during tests
    recur = types.ModuleType('backend.models.recurrent_model')
    recur.prepare_data_for_recurrent_model = lambda *a, **k: (None, None)
    recur.train_gru_model = lambda *a, **k: (None, None)
    recur.HAS_TENSORFLOW = False
    sys.modules['backend.models.recurrent_model'] = recur

    # Import predictor fresh so it uses patched database
    import importlib
    fast_pred = importlib.import_module("backend.fast_predictor")

    fp = fast_pred.FastPredictor()
    feats = fp._get_current_features(symbol, current_price=10100.0)

    assert feats is not None, "expected non-empty features from DB"
    assert isinstance(feats, pd.DataFrame)
    assert feats.shape[0] == 1 and feats.shape[1] > 5
    # Ensure no fallback was recorded
    assert getattr(fp, "_last_fallback_reason", None) is None
