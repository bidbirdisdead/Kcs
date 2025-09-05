# --- Configuration Constants ---
import os

# -- Symbol & API Settings --
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
RATE_LIMIT_DELAY = 20   # seconds -> 5 rpm
HOURS_BACK = 12        # backfill horizon
REST_PAGE_SIZE = 1000  # max bars per REST page
# -- Trading Mode --
TRADING_MODE = os.getenv("TRADING_MODE", "DRY").upper()

# -- Kalshi-specific Settings --
KALSHI_DECISION_EDGE = 0.05
KALSHI_STEP = {"BTC/USDT": 250, "ETH/USDT": 20}
KALSHI_WINGS = 2

# Trading hygiene and gating (can be tuned via environment)
# Minimum displayed liquidity (contracts) required at the actionable side
MIN_LIQUIDITY = int(os.getenv("MIN_LIQUIDITY", "1"))
# Maximum allowed inside-market spread in cents (derived yes_ask - yes_bid)
MAX_SPREAD_CENTS = int(os.getenv("MAX_SPREAD_CENTS", "20"))
# Required edge in cents; falls back to KALSHI_DECISION_EDGE*100 if not provided
DECISION_EDGE_CENTS = int(os.getenv("DECISION_EDGE_CENTS", str(int(max(1, round(KALSHI_DECISION_EDGE * 100))))))

# -- Model & Prediction Parameters --
# Uncertainty & calibration knobs
RV_BOOST = 0.75
MU_SOFT  = 1.5
PROB_FLOOR = 0.001

# Training window config
TRAIN_WINDOW_M = 700000
HORIZON = 60
PRED_MAX_BARS = 500000

# LightGBM training knobs (override via environment)
# - number of boosting rounds/trees
LGBM_N_ESTIMATORS = int(os.getenv("LGBM_N_ESTIMATORS", "1000"))
# - early stopping on validation set (rounds). Set 0 to disable.
LGBM_EARLY_STOPPING_ROUNDS = int(os.getenv("LGBM_EARLY_STOPPING_ROUNDS", "50"))
# - fraction of recent data kept for validation when training (0..0.5)
LGBM_VALIDATION_RATIO = float(os.getenv("LGBM_VALIDATION_RATIO", "0.1"))

# REST API pagination
REST_PAGE_SIZE = int(os.getenv("REST_PAGE_SIZE", "500"))

# Robust-fit utility constants
MIN_SAMPLES = 80
VAR_EPS     = 1e-12
SIGMA_FLOOR = 0.0015
RET_CLIP    = globals().get("RET_CLIP", 0.03)

# Backwards-compatible alias for trading module
KALSHI_SIGMA_FLOOR = SIGMA_FLOOR

# --- Feature Flags (Training/Inference Parity) ---
# When enabled, include flow/volume-derived features (tr, qv, tbv/tqv engineering)
TRAIN_WITH_FLOW_FEATS = os.getenv("TRAIN_WITH_FLOW_FEATS", "0") == "1"
ENABLE_FLOW_FEATS = os.getenv("ENABLE_FLOW_FEATS", "0") == "1"


