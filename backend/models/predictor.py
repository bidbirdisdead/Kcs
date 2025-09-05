"""Compatibility shim to expose get_fast_predictor under models.predictor.

This keeps the new package layout working while preserving the original
`fast_predictor.py` implementation.
"""
from ..fast_predictor import get_fast_predictor, FastPredictor

__all__ = ["get_fast_predictor", "FastPredictor"]
