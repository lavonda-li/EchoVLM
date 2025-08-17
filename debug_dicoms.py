#!/usr/bin/env python3
"""Debug script to check DICOM files in input directory."""

import glob
from pathlib import Path

def debug_dicom_files(input_dir: str):
    """Debug DICOM files in the input directory."""
    print(f"=== DICOM Files Debug ===")
    print(f"Input directory: {input_dir}")
    
    # Check if directory exists
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        return False
    
    print(f"✅ Input directory exists")
    
    # Find all DICOM files
    pattern = f"{input_dir}/**/*.dcm"
    dicom_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(dicom_files)} DICOM files")
    
    if len(dicom_files) == 0:
        print("❌ No DICOM files found")
        print("Trying alternative patterns...")
        
        # Try alternative patterns
        patterns = [
            f"{input_dir}/**/*.DCM",
            f"{input_dir}/**/*.dicom",
            f"{input_dir}/**/*.DICOM",
            f"{input_dir}/*.dcm",
            f"{input_dir}/*.DCM"
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern, recursive=True)
            if files:
                print(f"✅ Found {len(files)} files with pattern: {pattern}")
                print(f"First few files: {files[:5]}")
                break
        else:
            print("❌ No files found with any pattern")
            return False
    else:
        print(f"✅ Found {len(dicom_files)} DICOM files")
        print(f"First few files: {dicom_files[:5]}")
    
    # Check file sizes
    print("\nFile sizes:")
    for i, file_path in enumerate(dicom_files[:10]):  # Check first 10 files
        size = Path(file_path).stat().st_size
        print(f"  {i+1}. {Path(file_path).name}: {size / (1024*1024):.1f} MB")
    
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = "/home/lavonda/mount-folder/inference_output/p10"
    
    success = debug_dicom_files(input_dir)
    if success:
        print("\n✅ DICOM files found!")
    else:
        print("\n❌ No DICOM files found!")
