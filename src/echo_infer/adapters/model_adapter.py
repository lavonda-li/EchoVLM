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
        # Try multiple path resolution strategies
        possible_paths = []
        
        # Strategy 1: Relative to project root (src/echo_infer/adapters/model_adapter.py -> project root)
        project_root = Path(__file__).parent.parent.parent.parent
        possible_paths.append(project_root / weights_path)
        
        # Strategy 2: Relative to current working directory
        possible_paths.append(Path.cwd() / weights_path)
        
        # Strategy 3: If weights_path already contains modules/EchoPrime, try from current directory
        if "modules/EchoPrime" in weights_path:
            # Remove the modules/EchoPrime prefix and try from current directory
            relative_path = weights_path.replace("modules/EchoPrime/", "")
            possible_paths.append(Path.cwd() / relative_path)
        
        # Find the first path that exists
        resolved_weights_path = None
        for path in possible_paths:
            if path.exists():
                resolved_weights_path = str(path)
                break
        
        if resolved_weights_path is None:
            raise FileNotFoundError(f"Model weights file not found. Tried: {[str(p) for p in possible_paths]}")
        
        # Pass the resolved path to EchoPrime
        kwargs['weights_path'] = resolved_weights_path
    
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
