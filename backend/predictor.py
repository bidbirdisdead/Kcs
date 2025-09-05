"""Top-level predictor shim for backward compatibility.

Some modules import `backend.predictor` (package-root). Provide a thin shim
that re-exports the fast predictor implementation to avoid import errors and
allow the model trainer to clear caches.
"""
from .fast_predictor import get_fast_predictor, FastPredictor

__all__ = ["get_fast_predictor", "FastPredictor"]
