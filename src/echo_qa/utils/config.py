"""Configuration management for EchoQA."""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class Config:
    """Configuration for EchoQA processing."""
    
    # Dataset configuration
    data_str: str = field(default="train")
    input_dir: str = field(default="data")
    output_dir: str = field(default="output_batches")
    
    # Processing configuration
    process_all: bool = field(default=True)
    num_entries_to_process: int = field(default=10)
    batch_size: int = field(default=100)
    start_idx: int = field(default=0)
    
    # API configuration
    openai_api_key: Optional[str] = field(default=None)
    openai_model: str = field(default="gpt-4")
    max_tokens: int = field(default=300)
    
    # Logging configuration
    log_level: str = field(default="INFO")
    log_file: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Set API key from environment if not provided
        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        # Validate data_str
        if self.data_str not in ["train", "val", "test"]:
            raise ValueError("data_str must be one of: train, val, test")
        
        # Ensure paths are Path objects
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
    
    @classmethod
    def from_args(cls, args):
        """Create config from argparse arguments."""
        return cls(
            data_str=args.data_str,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            process_all=args.process_all,
            num_entries_to_process=args.num_entries_to_process,
            batch_size=args.batch_size,
            start_idx=args.start_idx,
            openai_model=getattr(args, 'openai_model', 'gpt-4'),
            max_tokens=getattr(args, 'max_tokens', 300),
            log_level=getattr(args, 'log_level', 'INFO'),
            log_file=getattr(args, 'log_file', None)
        )
    
    def validate(self):
        """Validate configuration."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.start_idx < 0:
            raise ValueError("start_idx must be non-negative")
    
    def get_input_file(self) -> Path:
        """Get the input file path for the current dataset."""
        return self.input_dir / f"CV_images_tinyllava-6-24-24-{self.data_str}.json"
    
    def get_output_dir(self) -> Path:
        """Get the output directory for the current dataset."""
        return self.output_dir / f"output_{self.data_str}"
    
    def get_final_output_file(self) -> Path:
        """Get the final output file path."""
        return self.get_output_dir() / f"combined_output_{self.data_str}.json"
