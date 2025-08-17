"""Main caption processor for medical image data."""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .api_client import OpenAIClient
from ..utils.io import save_json, load_json

logger = logging.getLogger(__name__)


class CaptionProcessor:
    """Main processor for medical image captions."""
    
    def __init__(self, api_client: OpenAIClient, input_dir: str = "data", output_dir: str = "output_batches"):
        """
        Initialize the caption processor.
        
        Args:
            api_client: OpenAI API client instance
            input_dir: Directory containing input data
            output_dir: Directory for output files
        """
        self.api_client = api_client
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single data entry.
        
        Args:
            entry: Data entry containing conversations
            
        Returns:
            Processed entry with caption and answers
        """
        try:
            # Extract caption from conversations
            caption = entry["conversations"][-1]["value"].strip()
            
            # Process caption through API
            answers = self.api_client.process_caption(caption)
            
            # Create new entry structure
            processed_entry = entry.copy()
            processed_entry.pop("conversations", None)
            processed_entry["caption"] = caption
            processed_entry["answers"] = answers
            
            return processed_entry
            
        except Exception as e:
            logger.error(f"Error processing entry: {e}")
            # Return original entry with error flag
            entry["error"] = str(e)
            return entry
    
    def process_batch(self, data: List[Dict[str, Any]], start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """
        Process a batch of data entries.
        
        Args:
            data: List of data entries
            start_idx: Starting index for batch
            end_idx: Ending index for batch
            
        Returns:
            List of processed entries
        """
        processed_entries = []
        
        for i in range(start_idx, end_idx):
            if i >= len(data):
                break
                
            entry = data[i]
            processed_entry = self.process_entry(entry)
            processed_entries.append(processed_entry)
            
            logger.info(f"Processed {i}th entry")
            
        return processed_entries
    
    def process_dataset(self, data_str: str, batch_size: int = 100, start_idx: int = 0, 
                       process_all: bool = True, num_entries: int = 10) -> str:
        """
        Process an entire dataset.
        
        Args:
            data_str: Dataset type (train, val, test)
            batch_size: Number of entries to process per batch
            start_idx: Starting index for processing
            process_all: Whether to process all entries
            num_entries: Number of entries to process if not processing all
            
        Returns:
            Path to the final combined output file
        """
        # Input and output file paths
        input_file = self.input_dir / f"CV_images_tinyllava-6-24-24-{data_str}.json"
        output_dir = self.output_dir / f"output_{data_str}"
        final_output_file = output_dir / f"combined_output_{data_str}.json"
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info(f"Loading data from {input_file}")
        data = load_json(input_file)
        logger.info(f"Loaded {len(data)} entries")
        
        # Determine processing range
        total_entries = len(data) if process_all else min(num_entries, len(data))
        
        # Process in batches
        for batch_start in range(start_idx, total_entries, batch_size):
            batch_end = min(batch_start + batch_size, total_entries)
            logger.info(f"Processing batch {batch_start} to {batch_end - 1}")
            
            batch_data = self.process_batch(data, batch_start, batch_end)
            
            # Save batch to file
            batch_file = output_dir / f"batch_{batch_start}_{batch_end - 1}.json"
            save_json(batch_data, batch_file)
            logger.info(f"Batch saved to {batch_file}")
        
        # Combine all batches
        self._combine_batches(output_dir, final_output_file)
        logger.info(f"All batches combined into {final_output_file}")
        
        return str(final_output_file)
    
    def _combine_batches(self, output_dir: Path, final_output_file: Path):
        """Combine all batch files into a single output file."""
        combined_data = []
        
        # Get all batch files and sort them
        batch_files = sorted([f for f in output_dir.iterdir() if f.name.endswith(".json")])
        
        for batch_file in batch_files:
            batch_data = load_json(batch_file)
            combined_data.extend(batch_data)
        
        save_json(combined_data, final_output_file)
