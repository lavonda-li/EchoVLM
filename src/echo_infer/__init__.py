"""EchoPrime inference package - thin wrapper around EchoPrime submodule."""

from .config import load_config

__version__ = "0.1.0"
__all__ = ["load_config"]

# Lazy imports to avoid issues with EchoPrime submodule paths
def _import_pipeline():
    from .pipeline import run
    return run

def _import_model_adapter():
    from .adapters.model_adapter import load_echoprime_model
    return load_echoprime_model

def _import_dataset_adapter():
    from .adapters.dataset_adapter import process_dicoms
    return process_dicoms

# Add to __all__ for backward compatibility
__all__.extend(["run", "load_echoprime_model", "process_dicoms"])
