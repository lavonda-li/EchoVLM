"""Dataset adapter for EchoPrime submodule."""

from typing import Dict, List, Optional

import torch

from echo_prime import EchoPrime


def process_dicoms(
    input_dir: str,
    model: Optional[EchoPrime] = None,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """Process DICOM files using EchoPrime's processing pipeline.
    
    Args:
        input_dir: Directory containing DICOM files
        model: EchoPrime model instance (optional, creates one if not provided)
        **kwargs: Additional arguments passed to model.process_dicoms
        
    Returns:
        Dictionary mapping file paths to processed video tensors
    """
    if model is None:
        model = EchoPrime()
    
    # Environment is already set up in __init__.py
    try:
        video_dict = model.process_dicoms(input_dir, **kwargs)
        return video_dict
    except Exception as e:
        # Log the error but don't crash
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in process_dicoms: {e}")
        logger.info("Returning empty video dict")
        return {}


def get_view_predictions(
    video_tensor: torch.Tensor,
    model: EchoPrime
) -> List[str]:
    """Get view predictions for processed video tensor.
    
    Args:
        video_tensor: Processed video tensor
        model: EchoPrime model instance
        
    Returns:
        List of predicted view names
    """
    # Use EchoPrime's view classification
    stack_of_first_frames = video_tensor[:, :, 0, :, :].to(model.device)
    
    with torch.no_grad():
        out_logits = model.view_classifier(stack_of_first_frames)
    
    out_views = torch.argmax(out_logits, dim=1)
    
    # Import COARSE_VIEWS from EchoPrime utils
    sys.path.insert(0, str(echoprime_path))
    from utils import COARSE_VIEWS
    
    view_list = [COARSE_VIEWS[v] for v in out_views]
    return view_list


def batch_process_dicoms(
    input_dirs: List[str],
    model: Optional[EchoPrime] = None,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """Process multiple DICOM directories.
    
    Args:
        input_dirs: List of directories containing DICOM files
        model: EchoPrime model instance (optional)
        **kwargs: Additional arguments passed to process_dicoms
        
    Returns:
        Combined dictionary mapping file paths to processed video tensors
    """
    if model is None:
        model = EchoPrime()
    
    all_videos = {}
    for input_dir in input_dirs:
        videos = process_dicoms(input_dir, model, **kwargs)
        all_videos.update(videos)
    
    return all_videos
