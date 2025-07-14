#!/usr/bin/env python3
"""
Script to clean "Pixel Data" entries from results.json files.
Iterates through all results.json files in the specified directory and removes
the "Pixel Data" key from the metadata of each object.
"""

import os
import json
import glob
import argparse
from pathlib import Path


def clean_pixel_data_from_json(json_file_path):
    """
    Remove "Pixel Data" entries from metadata in a results.json file.

    Args:
        json_file_path (str): Path to the JSON file to clean

    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        modified = False

        # Iterate through each entry in the JSON
        for key, value in data.items():
            if isinstance(value, dict) and 'metadata' in value:
                metadata = value['metadata']
                if 'Pixel Data' in metadata:
                    del metadata['Pixel Data']
                    modified = True
                    print(f"Removed 'Pixel Data' from {key}")

        # Write back the modified data if changes were made
        if modified:
            with open(json_file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Updated {json_file_path}")
            return True
        else:
            print(f"- No 'Pixel Data' entries found in {json_file_path}")
            return False

    except Exception as e:
        print(f"✗ Error processing {json_file_path}: {str(e)}")
        return False


def find_and_clean_results_files(input_directory):
    """
    Find all results.json files in the directory tree and clean them.

    Args:
        input_directory (str): Root directory to search for results.json files
    """
    # Find all results.json files recursively
    pattern = os.path.join(input_directory, "**/results.json")
    json_files = glob.glob(pattern, recursive=True)

    if not json_files:
        print(f"No results.json files found in {input_directory}")
        return

    print(f"Found {len(json_files)} results.json files")
    print("=" * 50)

    modified_count = 0

    for json_file in json_files:
        print(f"\nProcessing: {json_file}")
        if clean_pixel_data_from_json(json_file):
            modified_count += 1

    print("=" * 50)
    print(f"Summary: {modified_count}/{len(json_files)} files were modified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean 'Pixel Data' entries from results.json files"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input directory to search for results.json files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    # Verify the input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        exit(1)

    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print("=" * 50)

    find_and_clean_results_files(args.input)