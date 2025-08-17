"""Command-line interface for EchoQA."""

import argparse
import sys
from pathlib import Path

from .core.processor import CaptionProcessor
from .core.api_client import OpenAIClient
from .utils.config import Config
from .utils.logging import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Process medical image captions using OpenAI API for Q&A generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        "--data_str", 
        type=str, 
        choices=["train", "val", "test"], 
        required=True,
        help="Dataset type to process"
    )
    
    # Directory arguments
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="data",
        help="Input directory containing data files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output_batches",
        help="Output directory for processed files"
    )
    
    # Processing arguments
    parser.add_argument(
        "--process_all", 
        action="store_true", 
        default=True,
        help="Process all entries in the dataset"
    )
    parser.add_argument(
        "--num_entries_to_process", 
        type=int, 
        default=10,
        help="Number of entries to process if not processing all"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=100,
        help="Number of entries to process per batch"
    )
    parser.add_argument(
        "--start_idx", 
        type=int, 
        default=0,
        help="Start index for processing"
    )
    
    # API arguments
    parser.add_argument(
        "--openai_model", 
        type=str, 
        default="gpt-4",
        help="OpenAI model to use for processing"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=300,
        help="Maximum tokens for API response"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--log_file", 
        type=str, 
        default=None,
        help="Log file path (optional)"
    )
    
    return parser


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = Config.from_args(args)
        
        # Setup logging
        setup_logging(
            level=config.log_level,
            log_file=config.log_file
        )
        
        # Validate configuration
        config.validate()
        
        # Print configuration
        print(f"Configuration:")
        print(f"  Dataset: {config.data_str}")
        print(f"  Input directory: {config.input_dir}")
        print(f"  Output directory: {config.output_dir}")
        print(f"  Process all: {config.process_all}")
        print(f"  Number of entries: {config.num_entries_to_process}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Start index: {config.start_idx}")
        print(f"  OpenAI model: {config.openai_model}")
        print(f"  Max tokens: {config.max_tokens}")
        print()
        
        # Initialize API client
        api_client = OpenAIClient(
            api_key=config.openai_api_key,
            model=config.openai_model,
            max_tokens=config.max_tokens
        )
        
        # Initialize processor
        processor = CaptionProcessor(
            api_client=api_client,
            input_dir=str(config.input_dir),
            output_dir=str(config.output_dir)
        )
        
        # Process dataset
        output_file = processor.process_dataset(
            data_str=config.data_str,
            batch_size=config.batch_size,
            start_idx=config.start_idx,
            process_all=config.process_all,
            num_entries=config.num_entries_to_process
        )
        
        print(f"\nProcessing completed successfully!")
        print(f"Output file: {output_file}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
