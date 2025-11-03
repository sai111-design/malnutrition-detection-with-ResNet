#!/usr/bin/env python3
"""
Mistral 7B GGUF Model Downloader
Downloads the Q4_K_M quantized version (best balance of quality and speed)
"""

import os
import sys
from pathlib import Path

def download_mistral():
    """Download Mistral 7B using huggingface-cli"""

    print("="*70)
    print("Mistral 7B GGUF Model Downloader")
    print("="*70)

    # Check if huggingface-cli is installed
    print("\nStep 1: Checking dependencies...")

    try:
        import huggingface_hub
        print("✓ huggingface-hub is installed")
    except ImportError:
        print("✗ huggingface-hub not found")
        print("\nInstalling required package...")
        os.system("pip install huggingface-hub")

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"\nStep 2: Creating models directory...")
    print(f"✓ Using: {models_dir.absolute()}")

    # Download model
    print("\nStep 3: Downloading Mistral 7B (Q4_K_M - 4.37 GB)...")
    print("This will take 5-20 minutes depending on your internet speed...")

    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    try:
        os.system(f'huggingface-cli download {model_name} {model_file} --local-dir {models_dir} --local-dir-use-symlinks False')

        # Verify download
        model_path = models_dir / model_file
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024**3)  # Size in GB
            print(f"\n✓ Download successful!")
            print(f"✓ File size: {file_size:.2f} GB")
            print(f"✓ Location: {model_path}")
            return True
        else:
            print("✗ Download failed - file not found")
            return False
    except Exception as e:
        print(f"✗ Error during download: {e}")
        return False

if __name__ == "__main__":
    success = download_mistral()
    sys.exit(0 if success else 1)
