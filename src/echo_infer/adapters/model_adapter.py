"""Model adapter for EchoPrime submodule."""

import os
from pathlib import Path
from typing import Optional

import torch

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
    
    # Handle weights path resolution
    if weights_path:
        # Convert relative path to absolute path if needed
        if not Path(weights_path).is_absolute():
            # Get the EchoPrime submodule path
            echoprime_path = Path(__file__).parent.parent.parent.parent / "modules" / "EchoPrime"
            weights_path = str(echoprime_path / weights_path)
        
        # Verify the weights file exists
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {weights_path}")
        
        # Pass the absolute path to EchoPrime
        kwargs['weights_path'] = weights_path
    
    # Environment is already set up in __init__.py
    model = EchoPrime(device=device, **kwargs)
    return model


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
