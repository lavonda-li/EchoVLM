#!/usr/bin/env python3
"""Debug script to check weights path and EchoPrime setup."""

import os
import sys
from pathlib import Path

def debug_weights_path():
    """Debug the weights path issue."""
    print("=== EchoPrime Weights Path Debug ===")
    
    # Current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if we're in the right place
    if not Path("modules/EchoPrime").exists():
        print("❌ modules/EchoPrime not found in current directory")
        return False
    
    # Check EchoPrime structure
    echoprime_path = Path("modules/EchoPrime")
    print(f"EchoPrime path: {echoprime_path.absolute()}")
    
    # Check model_data directory
    model_data_path = echoprime_path / "model_data"
    if not model_data_path.exists():
        print("❌ model_data directory not found")
        return False
    
    print(f"✅ model_data directory found: {model_data_path}")
    
    # Check weights directory
    weights_path = model_data_path / "weights"
    if not weights_path.exists():
        print("❌ weights directory not found")
        return False
    
    print(f"✅ weights directory found: {weights_path}")
    
    # List all files in weights directory
    print("\nFiles in weights directory:")
    for file in weights_path.iterdir():
        if file.is_file():
            print(f"  - {file.name} ({file.stat().st_size / (1024*1024):.1f} MB)")
    
    # Check specific weights file
    weights_file = weights_path / "echo_prime_encoder.pt"
    if not weights_file.exists():
        print(f"❌ echo_prime_encoder.pt not found at {weights_file}")
        return False
    
    print(f"✅ echo_prime_encoder.pt found: {weights_file}")
    print(f"   Size: {weights_file.stat().st_size / (1024*1024):.1f} MB")
    
    # Test relative path resolution
    relative_path = "modules/EchoPrime/model_data/weights/echo_prime_encoder.pt"
    if Path(relative_path).exists():
        print(f"✅ Relative path works: {relative_path}")
    else:
        print(f"❌ Relative path doesn't work: {relative_path}")
    
    # Test absolute path
    abs_path = Path.cwd() / relative_path
    if abs_path.exists():
        print(f"✅ Absolute path works: {abs_path}")
    else:
        print(f"❌ Absolute path doesn't work: {abs_path}")
    
    return True

def debug_echoprime_import():
    """Debug EchoPrime import."""
    print("\n=== EchoPrime Import Debug ===")
    
    try:
        # Add EchoPrime to path
        echoprime_path = Path("modules/EchoPrime")
        sys.path.insert(0, str(echoprime_path))
        
        # Change working directory
        original_cwd = os.getcwd()
        os.chdir(echoprime_path)
        
        print(f"Changed working directory to: {os.getcwd()}")
        
        # Try to import EchoPrime
        from echo_prime import EchoPrime
        print("✅ EchoPrime import successful")
        
        # Try to create model instance
        model = EchoPrime()
        print("✅ EchoPrime model creation successful")
        
        # Restore working directory
        os.chdir(original_cwd)
        
    except Exception as e:
        print(f"❌ EchoPrime import failed: {e}")
        # Restore working directory
        os.chdir(original_cwd)
        return False
    
    return True

if __name__ == "__main__":
    print("Running EchoPrime weights path debug...")
    
    weights_ok = debug_weights_path()
    import_ok = debug_echoprime_import()
    
    if weights_ok and import_ok:
        print("\n✅ All checks passed!")
    else:
        print("\n❌ Some checks failed. Please review the output above.")
