"""Grievance triage training and inference package."""

from .config import Config
from .pipeline import train_and_predict

__all__ = ["Config", "train_and_predict"]
