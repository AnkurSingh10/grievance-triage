"""Grievance triage training and inference package."""

from .core.config import Config
from .ml.pipeline import train_and_predict

__all__ = ["Config", "train_and_predict"]
