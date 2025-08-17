#!/usr/bin/env python3
"""Test script to verify the new output format."""

from pathlib import Path

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

def test_filename_generation():
    """Test the filename generation function."""
    
    # Test cases
    test_cases = [
        {
            'dicom_path': '/Users/lavonda/EchoPrime/p10/p10002221/s94106955/94106955_0001.dcm',
            'input_dir': '/Users/lavonda/EchoPrime',
            'expected': 'p10_p10002221_s94106955.json'
        },
        {
            'dicom_path': '/Users/lavonda/EchoPrime/p10/p10002430/s92290733/92290733_0001.dcm',
            'input_dir': '/Users/lavonda/EchoPrime',
            'expected': 'p10_p10002430_s92290733.json'
        },
        {
            'dicom_path': '/Users/lavonda/EchoPrime/p10/p10002430/s98667422/98667422_0001.dcm',
            'input_dir': '/Users/lavonda/EchoPrime',
            'expected': 'p10_p10002430_s98667422.json'
        }
    ]
    
    print("Testing filename generation...")
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        result = _generate_output_filename(test_case['dicom_path'], test_case['input_dir'])
        expected = test_case['expected']
        passed = result == expected
        
        print(f"Test {i}:")
        print(f"  Input: {test_case['dicom_path']}")
        print(f"  Expected: {expected}")
        print(f"  Result: {result}")
        print(f"  Pass: {passed}")
        print()
        
        if not passed:
            all_passed = False
    
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    test_filename_generation()
