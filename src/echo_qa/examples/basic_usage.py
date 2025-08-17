#!/usr/bin/env python3
"""
Basic usage example for EchoQA package.

This script demonstrates how to use the EchoQA package to process
medical image captions programmatically.
"""

import os
import tempfile
from pathlib import Path

from echo_qa import CaptionProcessor, OpenAIClient, Config


def create_sample_data():
    """Create sample data for demonstration."""
    return [
        {
            "id": "sample_1",
            "conversations": [
                {
                    "value": "Echocardiogram showing normal left ventricular function with ejection fraction of 65%"
                }
            ]
        },
        {
            "id": "sample_2", 
            "conversations": [
                {
                    "value": "Chest X-ray revealing cardiomegaly with pulmonary congestion"
                }
            ]
        }
    ]


def main():
    """Main example function."""
    print("EchoQA Basic Usage Example")
    print("=" * 40)
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Using mock mode.")
        api_key = "mock-key"
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        
        # Create directories
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create sample data file
        sample_data = create_sample_data()
        input_file = input_dir / "CV_images_tinyllava-6-24-24-test.json"
        
        import json
        with open(input_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"Created sample data file: {input_file}")
        
        try:
            # Initialize components
            print("\nInitializing EchoQA components...")
            api_client = OpenAIClient(
                api_key=api_key,
                model="gpt-4",
                max_tokens=300
            )
            
            processor = CaptionProcessor(
                api_client=api_client,
                input_dir=str(input_dir),
                output_dir=str(output_dir)
            )
            
            print("Components initialized successfully!")
            
            # Process the sample data
            print("\nProcessing sample data...")
            output_file = processor.process_dataset(
                data_str="test",
                batch_size=2,
                start_idx=0,
                process_all=True
            )
            
            print(f"Processing completed!")
            print(f"Output file: {output_file}")
            
            # Display results
            print("\nResults:")
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            for i, result in enumerate(results, 1):
                print(f"\nSample {i}:")
                print(f"  ID: {result['id']}")
                print(f"  Caption: {result['caption']}")
                print(f"  Answers: {len(result['answers'])} answers generated")
                for answer in result['answers']:
                    print(f"    {answer}")
            
        except Exception as e:
            print(f"Error during processing: {e}")
            print("This is expected if OPENAI_API_KEY is not set or invalid.")


if __name__ == "__main__":
    main()
