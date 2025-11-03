#!/usr/bin/env python3
"""
Download Roboflow Malnutrition Detection Dataset
Usage: python scripts/download_dataset.py
"""

import os
import sys
from pathlib import Path

def download_with_roboflow():
    """Download using Roboflow Python package"""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Roboflow not installed. Installing...")
        os.system("pip install roboflow")
        from roboflow import Roboflow

    print("Downloading malnutrition detection dataset from Roboflow...")
    print("This may take a few minutes depending on internet speed...\n")

    rf = Roboflow(api_key="your_api_key_here")
    project = rf.workspace("workspacedavdev").project("detect-malnutrition-zaiof")

    dataset = project.download("folder", location="data/", overwrite=True)

    print(f"\n✓ Dataset downloaded successfully!")
    print(f"Location: {dataset.location}")

    return dataset.location

def verify_dataset_structure(data_dir="data/"):
    """Verify downloaded dataset structure"""
    print("\nVerifying dataset structure...")

    required_dirs = ['train', 'val', 'test']
    for dir_name in required_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if os.path.exists(dir_path):
            images_path = os.path.join(dir_path, 'images')
            if os.path.exists(images_path):
                img_count = len([f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            else:
                img_count = len([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            print(f"✓ {dir_name.upper()}: {img_count} images found")
        else:
            print(f"✗ {dir_name.upper()}: NOT FOUND")
            return False

    if os.path.exists(os.path.join(data_dir, 'data.yaml')):
        print("✓ data.yaml found")
    else:
        print("✗ data.yaml NOT FOUND (will create)")

    return True

def create_data_yaml(data_dir="data/"):
    """Create data.yaml if it doesn't exist"""
    yaml_path = os.path.join(data_dir, 'data.yaml')

    if os.path.exists(yaml_path):
        print("data.yaml already exists")
        return

    yaml_content = """path: {}
train: train/images
val: val/images
test: test/images

nc: 2
names: ['healthy', 'malnourished']
""".format(os.path.abspath(data_dir))

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"✓ Created data.yaml")

def main():
    """Main function"""
    print("="*70)
    print("Roboflow Malnutrition Detection Dataset Downloader")
    print("="*70 + "\n")

    os.makedirs("data/", exist_ok=True)

    print("Step 1: Downloading dataset from Roboflow...")
    print("-" * 70)
    try:
        data_location = download_with_roboflow()
        print("\nStep 2: Verifying dataset structure...")
        print("-" * 70)
        if verify_dataset_structure():
            print("\nStep 3: Creating configuration file...")
            print("-" * 70)
            create_data_yaml()
            print("\n" + "="*70)
            print("✓ Dataset setup complete!")
            print("="*70)
            print("\nNext steps:")
            print("1. Train model: python scripts/train.py")
            print("2. Run interface: python ui/gradio_app.py")
        else:
            print("\n✗ Dataset structure verification failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nManual download:")
        print("1. Visit: https://universe.roboflow.com/workspacedavdev/detect-malnutrition-zaiof")
        print("2. Click Export → Select 'Folder Structure'")
        print("3. Download and extract to data/ folder")
        sys.exit(1)

if __name__ == "__main__":
    main()
