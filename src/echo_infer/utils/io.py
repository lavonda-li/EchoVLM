"""IO utilities for EchoPrime inference."""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Union


def ensure_output_dir(output_dir: Union[str, Path]) -> Path:
    """Ensure output directory exists.
    
    Args:
        output_dir: Output directory path
        
    Returns:
        Path object for output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_results(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = "json"
) -> None:
    """Save results to file.
    
    Args:
        results: Results dictionary to save
        output_path: Output file path
        format: Output format ("json" or "csv")
    """
    output_path = Path(output_path)
    ensure_output_dir(output_path.parent)
    
    if format.lower() == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format.lower() == "csv":
        # Flatten nested results for CSV
        flattened = []
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened.append({
                        'file': key,
                        'field': subkey,
                        'value': subvalue
                    })
            else:
                flattened.append({
                    'file': key,
                    'field': 'result',
                    'value': value
                })
        
        with open(output_path, 'w', newline='') as f:
            if flattened:
                writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
                writer.writeheader()
                writer.writerows(flattened)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_manifest(manifest_path: Union[str, Path]) -> List[Dict[str, str]]:
    """Load batch processing manifest from CSV.
    
    Args:
        manifest_path: Path to CSV manifest file
        
    Returns:
        List of dictionaries with input/output paths
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    rows = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    return rows


def glob_files(
    input_dir: Union[str, Path],
    pattern: str = "*.dcm",
    recursive: bool = True
) -> List[Path]:
    """Glob files matching pattern.
    
    Args:
        input_dir: Input directory
        pattern: File pattern to match
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    input_path = Path(input_dir)
    if recursive:
        files = list(input_path.rglob(pattern))
    else:
        files = list(input_path.glob(pattern))
    
    return sorted(files)
