"""Main inference pipeline for EchoPrime."""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from .adapters.model_adapter import load_echoprime_model
from .adapters.dataset_adapter import process_dicoms, get_view_predictions
from .utils.io import ensure_output_dir, save_results, glob_files
from .utils.logging import setup_logging, get_logger


def _generate_output_filename(dicom_path: str, input_dir: str) -> str:
    """Generate output filename from DICOM path.
    
    Args:
        dicom_path: Full path to DICOM file
        input_dir: Input directory path
        
    Returns:
        Output filename in format: p10_p10002221_s94106955.json
    """
    # Convert to Path objects for easier manipulation
    dicom_path_obj = Path(dicom_path)
    input_dir_obj = Path(input_dir)
    
    try:
        # Get relative path from input directory
        relative_path = dicom_path_obj.relative_to(input_dir_obj)
        
        # Split path components
        parts = relative_path.parts
        
        # Extract the key components (p10, p10002221, s94106955)
        # Assuming structure: p10/p10002221/s94106955/file.dcm
        if len(parts) >= 3:
            # Get the directory levels (excluding the filename)
            key_parts = parts[:-1]  # Exclude the filename
            # Join with underscores and add .json extension
            output_name = "_".join(key_parts) + ".json"
            return output_name
        else:
            # Fallback: use the directory structure without filename
            dir_parts = parts[:-1] if len(parts) > 1 else parts
            return "_".join(dir_parts) + ".json" if dir_parts else "result.json"
            
    except ValueError:
        # If dicom_path is not relative to input_dir, use a different approach
        # Extract filename and create a safe name
        filename = dicom_path_obj.stem
        return f"{filename}.json"


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run EchoPrime inference pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Results dictionary
    """
    # Setup logging
    logger = setup_logging(
        level=config.get('logging', {}).get('level', 'INFO'),
        log_file=config.get('logging', {}).get('file')
    )
    
    logger.info("Starting EchoPrime inference pipeline")
    
        # Load model
    model_config = config.get('model', {})
    logger.info(f"Loading model: {model_config.get('name', 'echoprime')}")
    
    weights_path = model_config.get('weights_path')
    if weights_path:
        logger.info(f"Using weights path: {weights_path}")
        # Convert to absolute path for debugging
        if not Path(weights_path).is_absolute():
            echoprime_path = Path(__file__).parent.parent.parent.parent / "modules" / "EchoPrime"
            abs_weights_path = echoprime_path / weights_path
            logger.info(f"Absolute weights path: {abs_weights_path}")
            if not abs_weights_path.exists():
                logger.error(f"Weights file not found: {abs_weights_path}")
                raise FileNotFoundError(f"Model weights file not found: {abs_weights_path}")

    model = load_echoprime_model(
        weights_path=weights_path,
        device=model_config.get('device')
    )
    
    logger.info(f"Model loaded on device: {model.device}")
    
    # Process input data
    data_config = config.get('data', {})
    input_dir = data_config.get('input_dir', 'data/raw')
    pattern = data_config.get('pattern', '*.dcm')
    
    logger.info(f"Processing DICOM files from: {input_dir}")
    
    # Check if input directory exists
    if not Path(input_dir).exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Process DICOM files
    video_dict = process_dicoms(input_dir, model)
    
    if not video_dict:
        logger.warning("No valid DICOM files found")
        return {}
    
    logger.info(f"Processed {len(video_dict)} DICOM files")
    
    # Run inference and save individual results
    results = {}
    output_config = config.get('output', {})
    output_dir = output_config.get('dir', 'outputs/')
    ensure_output_dir(output_dir)
    
    for filename, video_tensor in video_dict.items():
        try:
            # Get view predictions
            view_list = get_view_predictions(video_tensor, model)
            
            # Create result data
            result_data = {
                'views': view_list,
                'video_shape': list(video_tensor.shape),
                'device': str(model.device),
                'source_file': filename
            }
            
            # Generate output filename from path
            output_filename = _generate_output_filename(filename, input_dir)
            output_file = Path(output_dir) / output_filename
            
            # Save individual result file
            save_results({filename: result_data}, output_file, format="json")
            
            # Store in results dict for return
            results[filename] = result_data
            
            logger.debug(f"Processed {filename}: {view_list} -> {output_filename}")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            results[filename] = {'error': str(e), 'source_file': filename}
    
    logger.info(f"Results saved to individual files in: {output_dir}")
    logger.info(f"Successfully processed {len([r for r in results.values() if 'error' not in r])} files")
    
    return results


def run_batch(
    manifest_path: str,
    config: Dict[str, Any],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Run batch inference using manifest file.
    
    Args:
        manifest_path: Path to CSV manifest file
        config: Configuration dictionary
        output_dir: Override output directory
        
    Returns:
        Combined results dictionary
    """
    from .utils.io import load_manifest
    
    logger = get_logger()
    logger.info(f"Starting batch inference with manifest: {manifest_path}")
    
    # Load manifest
    manifest = load_manifest(manifest_path)
    logger.info(f"Loaded {len(manifest)} items from manifest")
    
    # Override output directory if specified
    if output_dir:
        config = config.copy()
        config['output'] = config.get('output', {})
        config['output']['dir'] = output_dir
    
    # Process each item
    all_results = {}
    for i, item in enumerate(manifest):
        input_path = item.get('input', item.get('input_dir'))
        if not input_path:
            logger.warning(f"Missing input path in manifest row {i}")
            continue
        
        logger.info(f"Processing item {i+1}/{len(manifest)}: {input_path}")
        
        # Override input directory for this item
        item_config = config.copy()
        item_config['data'] = item_config.get('data', {})
        item_config['data']['input_dir'] = input_path
        
        try:
            results = run(item_config)
            all_results.update(results)
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            all_results[input_path] = {'error': str(e), 'source_file': input_path}
    
    logger.info(f"Batch processing complete. Total files processed: {len(all_results)}")
    return all_results
