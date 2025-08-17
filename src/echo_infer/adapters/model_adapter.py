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
    
    # Handle weights path resolution for verification only
    if weights_path:
        # Try multiple path resolution strategies to verify the weights exist
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
        
        # Don't pass weights_path to EchoPrime - it loads from relative paths
        # Just verify the file exists and log the path
        print(f"Verified weights file exists at: {resolved_weights_path}")
    
    # Environment is already set up in __init__.py, but we need to ensure working directory is correct for model initialization
    import os
    import torch as torch_module
    
    # Get the EchoPrime submodule path
    echoprime_path = Path(__file__).parent.parent.parent.parent / "modules" / "EchoPrime"
    
    # Change working directory to EchoPrime submodule for model initialization
    original_cwd = os.getcwd()
    os.chdir(echoprime_path)
    
    try:
        # Monkey patch torch.load to handle CPU-only environments
        original_torch_load = torch_module.load
        
        def safe_torch_load(*args, **kwargs):
            # Always use map_location=device for safety
            if 'map_location' not in kwargs:
                kwargs['map_location'] = device
            return original_torch_load(*args, **kwargs)
        
        # Apply the monkey patch
        torch_module.load = safe_torch_load
        
        try:
            model = EchoPrime(device=device, **kwargs)
            return model
        finally:
            # Restore original torch.load
            torch_module.load = original_torch_load
    finally:
        # Restore working directory
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
