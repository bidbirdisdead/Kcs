import os
import sqlite3
import json
import time
import hashlib
import io
from typing import List, Dict, Tuple, Optional, Any, Union

import joblib
import numpy as np

# --- Constants ---
MODEL_DB = os.getenv("MODEL_DB", "modeldb.sqlite")
MODEL_VERSION = 1

# --- Database Path ---
def get_db_path() -> str:
    """Return the path to the sqlite database for price data."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../cryptoprice.db"))

# --- Connection Handling ---
def open_conn(read_only: bool = False) -> sqlite3.Connection:
    db_path = get_db_path()
    if read_only:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA query_only=1;")
    else:
        conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn

def _init_db_pragmas():
    try:
        conn = open_conn(read_only=False)
        conn.close()
    except Exception as e:
        print(f"[DB] init PRAGMAs failed: {e}")

# --- Model Database Functions ---
def _modeldb_exec(sql, params=()):
    with sqlite3.connect(MODEL_DB) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute(sql, params)
        conn.commit()

def _modeldb_query(sql, params=()):
    with sqlite3.connect(MODEL_DB) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        cur = conn.execute(sql, params)
        return cur.fetchall()

def init_modeldb():
    _modeldb_exec("""
    CREATE TABLE IF NOT EXISTS model_registry(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      symbol TEXT NOT NULL,
      horizon_min INTEGER NOT NULL,
      regime INTEGER NOT NULL,
      algo TEXT NOT NULL,
      version INTEGER NOT NULL,
      feat_hash TEXT NOT NULL,
      train_start INTEGER NOT NULL,
      train_end   INTEGER NOT NULL,
      n_obs INTEGER NOT NULL,
      sigma_resid REAL NOT NULL,
      rv_hour REAL NOT NULL,
      sigma_tot REAL NOT NULL,
      status TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      expires_at INTEGER,
      checksum TEXT,
      UNIQUE(symbol,horizon_min,regime,algo,version,feat_hash,train_end)
    );""")
    _modeldb_exec("""
    CREATE TABLE IF NOT EXISTS model_blobs(
      model_id INTEGER PRIMARY KEY,
      model_blob BLOB NOT NULL,
      scaler_blob BLOB,
      FOREIGN KEY(model_id) REFERENCES model_registry(id) ON DELETE CASCADE
    );""")
    _modeldb_exec("""
    CREATE TABLE IF NOT EXISTS predictions(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts_decision INTEGER NOT NULL,
      symbol TEXT NOT NULL,
      ticker TEXT NOT NULL,
      strike INTEGER NOT NULL,
      side TEXT NOT NULL,
      p_hat REAL NOT NULL,
      price_now REAL NOT NULL,
      mu_eff REAL NOT NULL,
      sigma_tot REAL NOT NULL,
      regime INTEGER NOT NULL,
      horizon_min INTEGER NOT NULL DEFAULT 60,
      ts_settle INTEGER,
      y REAL,
      brier REAL,
      logloss REAL,
      UNIQUE(ts_decision, symbol, strike, horizon_min)
    );""")

def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _feat_hash(feat_order) -> str:
    return hashlib.sha1("|".join(feat_order).encode()).hexdigest()

def save_model_to_db(symbol, horizon_min, regime, algo, feat_order,
                     estimator, scaler, sigma_resid, rv_hour, sigma_tot,
                     train_start, train_end, n_obs, ttl_sec=None):
    now = int(time.time())
    feat_h = _feat_hash(feat_order)
    mbuf = io.BytesIO()
    joblib.dump(estimator, mbuf, compress=("lz4", 3))
    model_bytes = mbuf.getvalue()
    sbuf = None
    if scaler is not None:
        sb = io.BytesIO()
        joblib.dump(scaler, sb, compress=("lz4", 3))
        sbuf = sb.getvalue()
    chk = _sha1_bytes(model_bytes + (sbuf or b""))
    exp = now + int(ttl_sec) if ttl_sec else None

    with sqlite3.connect(MODEL_DB) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        cur = conn.execute(
            """
INSERT INTO model_registry(
  symbol,horizon_min,regime,algo,version,feat_hash,
  train_start,train_end,n_obs,sigma_resid,rv_hour,sigma_tot,
  status,created_at,expires_at,checksum
)
VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(symbol,horizon_min,regime,algo,version,feat_hash,train_end)
DO UPDATE SET
  n_obs=excluded.n_obs,
  sigma_resid=excluded.sigma_resid,
  rv_hour=excluded.rv_hour,
  sigma_tot=excluded.sigma_tot,
  status='ready',
  created_at=excluded.created_at,
  expires_at=excluded.expires_at,
  checksum=excluded.checksum
            """,
            (symbol, horizon_min, regime, algo, MODEL_VERSION, feat_h,
             int(train_start), int(train_end), int(n_obs), float(sigma_resid), float(rv_hour), float(sigma_tot),
             "ready", now, exp, chk)
        )
        mid = cur.lastrowid
        if not mid:
            row = conn.execute(
                "SELECT id FROM model_registry WHERE symbol=? AND horizon_min=? AND regime=? AND algo=? AND version=? AND feat_hash=? AND train_end=?",
                (symbol, horizon_min, regime, algo, MODEL_VERSION, feat_h, int(train_end)),
            ).fetchone()
            mid = int(row[0]) if row else None
        if not mid:
            raise RuntimeError("model_registry UPSERT succeeded but id could not be resolved")
        conn.execute(
            """
INSERT INTO model_blobs(model_id,model_blob,scaler_blob)
VALUES(?,?,?)
ON CONFLICT(model_id) DO UPDATE SET
  model_blob=excluded.model_blob,
  scaler_blob=excluded.scaler_blob
            """,
            (mid, model_bytes, sbuf),
        )
        conn.commit()
    return mid

def load_best_model(symbol, horizon_min, regime, algo, feat_order, max_age_sec=6*3600):
    now = int(time.time())
    feat_h = _feat_hash(feat_order)
    rows = _modeldb_query(
        """
          SELECT id,sigma_resid,rv_hour,sigma_tot,train_end
          FROM model_registry
          WHERE symbol=? AND horizon_min=? AND regime=? AND algo=? AND version=? AND feat_hash=? AND status='ready'
          ORDER BY train_end DESC LIMIT 1
        """,
        (symbol, horizon_min, regime, algo, MODEL_VERSION, feat_h)
    )
    if not rows:
        return None
    mid, sigma_resid, rv_hour, sigma_tot, train_end = rows[0]
    if now - int(train_end) > max_age_sec:
        return None
    blob = _modeldb_query("SELECT model_blob,scaler_blob FROM model_blobs WHERE model_id=?", (mid,))
    if not blob:
        return None
    model_blob, scaler_blob = blob[0]
    est = joblib.load(io.BytesIO(model_blob))
    scaler = joblib.load(io.BytesIO(scaler_blob)) if scaler_blob else None
    feat_order_loaded = None
    if isinstance(est, dict) and 'estimator' in est:
        feat_order_loaded = est.get('feat_order')
        est_obj = est.get('estimator')
    else:
        est_obj = est
    return dict(
        mu_pred_fn=(lambda x: float(est_obj.predict(np.asarray(x, float).reshape(1, -1))[0])),
        sigma_resid=float(sigma_resid),
        rv_hour=float(rv_hour),
        sigma_tot=float(sigma_tot),
        estimator=est_obj,
        scaler=scaler,
        feat_order=(feat_order_loaded if feat_order_loaded is not None else feat_order),
        reason="db-cache",
    )

def load_latest_algo_model(symbol, horizon_min, algo, max_age_sec=6*3600):
    now = int(time.time())
    rows = _modeldb_query(
        """
          SELECT id,sigma_resid,rv_hour,sigma_tot,train_end
          FROM model_registry
          WHERE symbol=? AND horizon_min=? AND algo=? AND status='ready'
          ORDER BY train_end DESC LIMIT 1
        """,
        (symbol, horizon_min, algo)
    )
    if not rows:
        return None
    mid, sigma_resid, rv_hour, sigma_tot, train_end = rows[0]
    if now - int(train_end) > max_age_sec:
        return None
    blob = _modeldb_query("SELECT model_blob,scaler_blob FROM model_blobs WHERE model_id=?", (mid,))
    if not blob:
        return None
    model_blob, scaler_blob = blob[0]
    est = joblib.load(io.BytesIO(model_blob))
    scaler = joblib.load(io.BytesIO(scaler_blob)) if scaler_blob else None
    feat_order_loaded = None
    if isinstance(est, dict) and 'estimator' in est:
        feat_order_loaded = est.get('feat_order')
        est_obj = est.get('estimator')
    else:
        est_obj = est
    return dict(
        estimator=est_obj,
        scaler=scaler,
        sigma_resid=float(sigma_resid),
        rv_hour=float(rv_hour),
        sigma_tot=float(sigma_tot),
        feat_order=feat_order_loaded,
        reason="db-latest",
    )

# --- Price Database Functions ---
def create_crypto_prices_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS crypto_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            close_price REAL,
            volume REAL,
            trades INTEGER,
            quote_volume REAL,
            taker_base_volume REAL,
            taker_quote_volume REAL,
            source TEXT DEFAULT 'FINAZON',
            UNIQUE(symbol, timestamp, source)
        )
        """
    )
    conn.commit()

def ws_upsert_bar(bar: dict):
    conn = open_conn(read_only=False)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO crypto_prices
        (symbol, timestamp, open_price, high_price, low_price, close_price, volume, trades,
         quote_volume, taker_base_volume, taker_quote_volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'FINAZON_WS')
        ON CONFLICT(symbol, timestamp, source) DO UPDATE SET
            open_price         = COALESCE(crypto_prices.open_price, excluded.open_price),
            high_price         = COALESCE(crypto_prices.high_price, excluded.high_price),
            low_price          = COALESCE(crypto_prices.low_price, excluded.low_price),
            close_price        = COALESCE(crypto_prices.close_price, excluded.close_price),
            volume             = COALESCE(crypto_prices.volume, excluded.volume),
            trades             = COALESCE(crypto_prices.trades, excluded.trades),
            quote_volume       = COALESCE(crypto_prices.quote_volume, excluded.quote_volume),
            taker_base_volume  = COALESCE(crypto_prices.taker_base_volume, excluded.taker_base_volume),
            taker_quote_volume = COALESCE(crypto_prices.taker_quote_volume, excluded.taker_quote_volume)
        """,
        (
            bar["s"],
            bar["t"],
            bar.get("o"),
            bar.get("h"),
            bar.get("l"),
            bar.get("c"),
            # Finazon REST uses 'bv' for base volume and REST code maps bv -> volume.
            # WS payloads historically used 'v' for volume but some messages omit it.
            # Try several common keys and leave NULL when volume is not provided.
            (bar.get("v") or bar.get("bv") or bar.get("volume") or bar.get("vol") ),
            # Trades count may be provided as 'tr' (REST) or 'n'/'trades' in some feeds.
            (bar.get("tr") or bar.get("n") or bar.get("trades")),
            bar.get("qv"),
            bar.get("tbv"),
            bar.get("tqv"),
        ),
    )
    conn.commit()
    conn.close()

def rest_upsert_bar(symbol: str, d: dict) -> bool:
    """Insert or update a REST bar.

    On conflict, update only missing columns using COALESCE so that subsequent
    backfills can complete previously partial rows without overwriting newer
    non-null values.
    """
    conn = open_conn(read_only=False)
    cur = conn.cursor()
    ts = d.get("t")
    cur.execute(
        """
        INSERT INTO crypto_prices
        (symbol, timestamp, open_price, high_price, low_price, close_price, volume, trades,
         quote_volume, taker_base_volume, taker_quote_volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'FINAZON_REST')
        ON CONFLICT(symbol, timestamp, source) DO UPDATE SET
            open_price         = COALESCE(crypto_prices.open_price, excluded.open_price),
            high_price         = COALESCE(crypto_prices.high_price, excluded.high_price),
            low_price          = COALESCE(crypto_prices.low_price, excluded.low_price),
            close_price        = COALESCE(crypto_prices.close_price, excluded.close_price),
            volume             = COALESCE(crypto_prices.volume, excluded.volume),
            trades             = COALESCE(crypto_prices.trades, excluded.trades),
            quote_volume       = COALESCE(crypto_prices.quote_volume, excluded.quote_volume),
            taker_base_volume  = COALESCE(crypto_prices.taker_base_volume, excluded.taker_base_volume),
            taker_quote_volume = COALESCE(crypto_prices.taker_quote_volume, excluded.taker_quote_volume)
        """,
        (
            symbol, ts,
            d.get("o"), d.get("h"), d.get("l"), d.get("c"),
            d.get("bv"), d.get("tr"), d.get("qv"), d.get("tbv"), d.get("tqv"),
        ),
    )
    changed = cur.rowcount > 0
    conn.commit()
    conn.close()
    return changed

def select_one_bar_for_ts(symbol: str, ts: int):
    conn = open_conn(read_only=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT open_price, high_price, low_price, close_price, volume
        FROM crypto_prices
        WHERE symbol=? AND timestamp=? AND source='FINAZON_REST'
        ORDER BY rowid DESC LIMIT 1
        """,
        (symbol, ts),
    )
    row = cur.fetchone()
    if not row:
        cur.execute(
            """
            SELECT open_price, high_price, low_price, close_price, volume
            FROM crypto_prices
            WHERE symbol=? AND timestamp=?
            ORDER BY source='FINAZON_WS' DESC, rowid DESC
            LIMIT 1
            """,
            (symbol, ts),
        )
        row = cur.fetchone()
    conn.close()
    return row

def get_price_range(symbol: str, start_ts: int, end_ts: int) -> Tuple[float, float]:
    conn = open_conn(read_only=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT timestamp
        FROM crypto_prices
        WHERE symbol=? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
        """,
        (symbol, start_ts, end_ts),
    )
    ts_list = [r[0] for r in cur.fetchall()]
    conn.close()
    closes: List[float] = []
    for ts in ts_list:
        bar = select_one_bar_for_ts(symbol, ts)
        if bar and bar[3] is not None:
            closes.append(bar[3])
    if not closes:
        return (0.0, 0.0)
    return (min(closes), max(closes))

def find_missing_1m_timestamps(symbol: str, hours: int) -> List[int]:
    conn = open_conn(read_only=True)
    cur = conn.cursor()
    end_ts = int(time.time()) // 60 * 60 - 60
    start_ts = end_ts - hours * 3600
    start_ts -= start_ts % 60
    cur.execute(
        """
        SELECT DISTINCT timestamp
        FROM crypto_prices
        WHERE symbol=? AND timestamp BETWEEN ? AND ?
        """,
        (symbol, start_ts, end_ts),
    )
    present = {row[0] for row in cur.fetchall()}
    conn.close()
    expected = set(range(start_ts, end_ts + 1, 60))
    missing = sorted(expected - present)
    return missing

def get_latest_price(symbol: str) -> Union[float, str, None]:
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT close_price FROM crypto_prices WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1",
            (symbol,),
        )
        row = cur.fetchone()
        conn.close()
        if row and row[0] is not None:
            return float(row[0])
        return None
    except Exception as exc:
        return f"Error: {exc}"
