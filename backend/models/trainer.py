"""Compatibility shim exposing get_model_trainer from model_trainer.py

This keeps code that imports from models.trainer working while preserving the
single-file trainer implementation.
"""
from ..model_trainer import get_model_trainer, ModelTrainer

__all__ = ["get_model_trainer", "ModelTrainer"]
