"""Adapters for EchoPrime submodule integration."""

import os
import sys
from pathlib import Path

# Set up EchoPrime environment before any imports
def _setup_echoprime_environment():
    """Set up the environment for EchoPrime imports."""
    echoprime_path = Path(__file__).parent.parent.parent.parent / "modules" / "EchoPrime"
    
    # Add to Python path
    if str(echoprime_path) not in sys.path:
        sys.path.insert(0, str(echoprime_path))
    
    # Set environment variable
    os.environ['ECHOPRIME_ROOT'] = str(echoprime_path)
    
    # Change working directory temporarily for asset loading
    original_cwd = os.getcwd()
    os.chdir(echoprime_path)
    
    return original_cwd, echoprime_path

# Set up environment immediately
_original_cwd, _echoprime_path = _setup_echoprime_environment()

# Now import the adapters
from .model_adapter import load_echoprime_model
from .dataset_adapter import process_dicoms, get_view_predictions

# Restore working directory
os.chdir(_original_cwd)

__all__ = ["load_echoprime_model", "process_dicoms", "get_view_predictions"]
