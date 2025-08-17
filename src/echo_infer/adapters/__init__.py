"""Adapters for EchoPrime submodule integration."""

from .model_adapter import load_echoprime_model
from .dataset_adapter import process_dicoms

__all__ = ["load_echoprime_model", "process_dicoms"]
