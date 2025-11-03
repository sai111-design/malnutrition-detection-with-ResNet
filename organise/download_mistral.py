#!/usr/bin/env python3
"""
Mistral 7B GGUF Model Downloader
No need for huggingface-cli - downloads directly in Python
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_mistral():
    """Download Mistral 7B using Python API"""

    print("="*70)
    print("Mistral 7B GGUF Model Downloader (Python-Based)")
    print("="*70)

    # Step 1: Check huggingface_hub
    print("\nStep 1: Checking dependencies...")
    try:
        import huggingface_hub
        print(f"✓ huggingface_hub is installed (version: {huggingface_hub.__version__})")
    except ImportError:
        print("✗ huggingface_hub not found - installing...")
        os.system("pip install huggingface-hub")
        import huggingface_hub

    # Step 2: Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"\nStep 2: Using models directory")
    print(f"✓ Location: {models_dir.absolute()}")

    # Step 3: Download model
    print("\nStep 3: Downloading Mistral 7B (Q4_K_M - 4.37 GB)...")
    print("This will take 5-20 minutes depending on internet speed...")
    print("Progress will show below:\n")

    try:
        model_repo = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
        model_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

        print(f"Downloading from: {model_repo}/{model_file}")

        model_path = hf_hub_download(
            repo_id=model_repo,
            filename=model_file,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False
        )

        # Verify download
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024**3)  # Size in GB
            print(f"\n✓ Download successful!")
            print(f"✓ File size: {file_size:.2f} GB")
            print(f"✓ Location: {model_path}")

            print("\n" + "="*70)
            print("NEXT STEPS:")
            print("="*70)
            print("\n1. Create src/llm_handler.py with code from MISTRAL-7B-SETUP-GUIDE.md")
            print("2. Run: python test_mistral.py")
            print("3. Integrate into your web app")

            return True
        else:
            print(f"✗ Download failed - file not found at {model_path}")
            return False

    except Exception as e:
        print(f"✗ Download error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check internet connection")
        print("2. Make sure you have enough disk space (5+ GB)")
        print("3. Try again in a few minutes")
        return False

if __name__ == "__main__":
    success = download_mistral()
    sys.exit(0 if success else 1)
