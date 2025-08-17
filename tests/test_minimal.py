#!/usr/bin/env python3
"""Minimal test to check EchoPrime model loading."""

import os
import sys
import torch
from pathlib import Path

def test_minimal_loading():
    """Test minimal EchoPrime loading."""
    print("=== Minimal EchoPrime Loading Test ===")
    
    # Set up environment
    echoprime_path = Path("modules/EchoPrime")
    sys.path.insert(0, str(echoprime_path))
    
    # Change working directory
    original_cwd = os.getcwd()
    os.chdir(echoprime_path)
    
    try:
        # Try to load just the echo encoder
        print("Loading echo encoder...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load("model_data/weights/echo_prime_encoder.pt", map_location=device)
        
        print("✅ Echo encoder weights loaded successfully")
        print(f"Checkpoint keys: {list(checkpoint.keys())[:5]}...")  # Show first 5 keys
        
        # Try to create the model architecture
        import torchvision
        echo_encoder = torchvision.models.video.mvit_v2_s()
        echo_encoder.head[-1] = torch.nn.Linear(echo_encoder.head[-1].in_features, 512)
        
        print("✅ Echo encoder architecture created")
        
        # Try to load state dict
        echo_encoder.load_state_dict(checkpoint)
        echo_encoder.eval()
        echo_encoder.to(device)
        
        print("✅ Echo encoder loaded and ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = test_minimal_loading()
    if success:
        print("\n✅ Minimal test passed!")
    else:
        print("\n❌ Minimal test failed!")
