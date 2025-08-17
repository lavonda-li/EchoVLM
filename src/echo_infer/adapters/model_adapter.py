"""Model adapter for EchoPrime submodule."""

import os
import sys
from pathlib import Path
from typing import Optional

import torch

# Add EchoPrime submodule to path
echoprime_path = Path(__file__).parent.parent.parent.parent / "modules" / "EchoPrime"
sys.path.insert(0, str(echoprime_path))

from echo_prime import EchoPrime


def load_echoprime_model(
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs
) -> EchoPrime:
    """Load EchoPrime model from submodule.
    
    Args:
        weights_path: Path to model weights (optional, uses default if not provided)
        device: Device to load model on
        **kwargs: Additional arguments passed to EchoPrime constructor
        
    Returns:
        Loaded EchoPrime model instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Change working directory to EchoPrime submodule for relative paths
    original_cwd = os.getcwd()
    os.chdir(echoprime_path)
    
    try:
        # Set environment variable to help with relative paths
        os.environ['ECHOPRIME_ROOT'] = str(echoprime_path)
        
        model = EchoPrime(device=device, **kwargs)
        return model
    except Exception as e:
        # Restore working directory even on error
        os.chdir(original_cwd)
        raise e
    finally:
        os.chdir(original_cwd)


def get_model_info(model: EchoPrime) -> dict:
    """Get information about loaded model.
    
    Args:
        model: EchoPrime model instance
        
    Returns:
        Dictionary with model information
    """
    return {
        "model_type": "EchoPrime",
        "device": str(model.device),
        "frames_to_take": model.frames_to_take,
        "frame_stride": model.frame_stride,
        "video_size": model.video_size,
    }
