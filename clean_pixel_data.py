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
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time


def clean_pixel_data_from_json(json_file_path):
    """
    Remove "Pixel Data" entries from metadata in a results.json file.

    Args:
        json_file_path (str): Path to the JSON file to clean

    Returns:
        tuple: (file_path, success, modified, message)
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        modified = False
        entries_processed = 0

        # Iterate through each entry in the JSON
        for key, value in data.items():
            if isinstance(value, dict) and 'metadata' in value:
                metadata = value['metadata']
                entries_processed += 1

                if 'Pixel Data' in metadata:
                    del metadata['Pixel Data']
                    modified = True

                if "Sequence of Ultrasound Regions" in metadata:
                    del metadata["Sequence of Ultrasound Regions"]
                    modified = True

        # Write back the modified data if changes were made
        if modified:
            with open(json_file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return (json_file_path, True, True, f"Updated {entries_processed} entries")
        else:
            return (json_file_path, True, False, f"No changes needed ({entries_processed} entries)")

    except Exception as e:
        return (json_file_path, False, False, f"Error: {str(e)}")


def process_files_batch(file_paths):
    """Process a batch of files sequentially to reduce overhead."""
    results = []
    for file_path in file_paths:
        results.append(clean_pixel_data_from_json(file_path))
    return results


def find_and_clean_results_files(input_directory, max_workers=None, batch_size=10):
    """
    Find all results.json files in the directory tree and clean them.

    Args:
        input_directory (str): Root directory to search for results.json files
        max_workers (int): Number of worker processes (default: CPU count)
        batch_size (int): Number of files to process per batch
    """
    print("Searching for results.json files...")
    start_time = time.time()

    # Find all results.json files recursively
    pattern = os.path.join(input_directory, "**/results.json")
    json_files = glob.glob(pattern, recursive=True)

    if not json_files:
        print(f"No results.json files found in {input_directory}")
        return

    search_time = time.time() - start_time
    print(f"Found {len(json_files)} results.json files in {search_time:.2f}s")

    if max_workers is None:
        max_workers = min(cpu_count(), 8)  # Cap at 8 to avoid overwhelming I/O

    print(f"Processing with {max_workers} workers in batches of {batch_size}")
    print("=" * 70)

    # Split files into batches
    batches = [json_files[i:i + batch_size] for i in range(0, len(json_files), batch_size)]

    modified_count = 0
    error_count = 0
    processed_count = 0

    start_processing = time.time()

    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_batch = {executor.submit(process_files_batch, batch): batch for batch in batches}

        # Process completed batches
        for future in as_completed(future_to_batch):
            batch_results = future.result()

            for file_path, success, modified, message in batch_results:
                processed_count += 1

                if success:
                    if modified:
                        modified_count += 1
                        status = "✓"
                    else:
                        status = "-"
                else:
                    error_count += 1
                    status = "✗"

                # Show progress every 10 files or for errors
                if processed_count % 10 == 0 or not success:
                    elapsed = time.time() - start_processing
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    progress = (processed_count / len(json_files)) * 100

                    print(f"{status} [{processed_count:4d}/{len(json_files):4d}] "
                          f"({progress:5.1f}%) {rate:5.1f} files/s - {Path(file_path).name}")

                    if not success:
                        print(f"    {message}")

    processing_time = time.time() - start_processing
    total_time = time.time() - start_time

    print("=" * 70)
    print(f"Summary:")
    print(f"  Total files: {len(json_files)}")
    print(f"  Modified: {modified_count}")
    print(f"  Unchanged: {processed_count - modified_count - error_count}")
    print(f"  Errors: {error_count}")
    print(f"  Processing time: {processing_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average rate: {len(json_files) / processing_time:.1f} files/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean 'Pixel Data' entries from results.json files"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory to search for results.json files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count, max 8)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of files to process per batch (default: 10)"
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
        print("=" * 70)

    find_and_clean_results_files(args.input, args.workers, args.batch_size)