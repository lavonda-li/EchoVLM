"""I/O utilities for EchoQA."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def save_json(data: Any, file_path: Path, indent: int = 4) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to the output file
        indent: JSON indentation
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.debug(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise


def load_json(file_path: Path) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        Loaded data
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Data loaded from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def ensure_directory(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
    """
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Directory ensured: {path}")


def get_file_size(file_path: Path) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
    """
    try:
        return file_path.stat().st_size
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return 0
