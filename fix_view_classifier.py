#!/usr/bin/env python3
"""Fix view classifier checkpoint by extracting state dict from PyTorch Lightning checkpoint."""

import os
import torch
from pathlib import Path

def fix_view_classifier_checkpoint():
    """Extract state dict from PyTorch Lightning checkpoint."""
    print("=== Fixing View Classifier Checkpoint ===")
    
    # Paths
    weights_dir = Path("modules/EchoPrime/model_data/weights")
    checkpoint_path = weights_dir / "view_classifier.pt"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Load the PyTorch Lightning checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Extract the state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"✅ Found state_dict with {len(state_dict)} keys")
            
            # Remove the 'model.' prefix if it exists
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    cleaned_key = key[6:]  # Remove 'model.' prefix
                else:
                    cleaned_key = key
                cleaned_state_dict[cleaned_key] = value
            
            print(f"✅ Cleaned state_dict with {len(cleaned_state_dict)} keys")
            
            # Save the cleaned state dict
            output_path = weights_dir / "view_classifier_fixed.pt"
            torch.save(cleaned_state_dict, output_path)
            print(f"✅ Saved fixed state dict to: {output_path}")
            
            # Create a backup and replace the original
            backup_path = weights_dir / "view_classifier_backup.pt"
            os.rename(checkpoint_path, backup_path)
            os.rename(output_path, checkpoint_path)
            
            print(f"✅ Replaced original with fixed version")
            print(f"✅ Original backed up to: {backup_path}")
            
            return True
            
        else:
            print("❌ No 'state_dict' found in checkpoint")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = fix_view_classifier_checkpoint()
    if success:
        print("\n✅ View classifier checkpoint fixed!")
    else:
        print("\n❌ Failed to fix view classifier checkpoint!")
