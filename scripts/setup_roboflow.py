#!/usr/bin/env python3
"""
Setup Roboflow Dataset - Organize images by class
Usage: python scripts/setup_roboflow.py
"""

import os
import shutil
import json
from pathlib import Path

def create_class_folders(data_dir="data/"):
    """Create class-based folder structure"""
    print("Creating class-based folder structure...\n")

    splits = ['train', 'val', 'test']
    classes = ['healthy', 'malnourished']

    for split in splits:
        split_dir = os.path.join(data_dir, split)
        for cls in classes:
            class_dir = os.path.join(split_dir, cls)
            os.makedirs(class_dir, exist_ok=True)
            print(f"Created: {class_dir}")

def read_roboflow_labels(data_dir="data/"):
    """Read labels from Roboflow metadata files"""
    print("\nReading image labels...")

    labels = {}

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_dir, split)
        csv_path = os.path.join(split_path, '_annotations.csv')
        if os.path.exists(csv_path):
            print(f"Found annotations in {split}_annotations.csv")
            with open(csv_path, 'r') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        filename = parts[0]
                        class_name = parts[1].lower()
                        labels[filename] = class_name

    return labels

def organize_images_by_class(data_dir="data/", labels=None):
    """Organize images into class folders"""
    if labels is None:
        labels = read_roboflow_labels(data_dir)

    print("\nOrganizing images by class...\n")

    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(data_dir, split, 'images')

        if not os.path.exists(images_dir):
            print(f"Skipping {split} - images directory not found")
            continue

        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                src_path = os.path.join(images_dir, filename)

                if filename in labels:
                    class_name = labels[filename]
                else:
                    if 'healthy' in filename.lower():
                        class_name = 'healthy'
                    elif 'malnourished' in filename.lower():
                        class_name = 'malnourished'
                    else:
                        class_name = 'unknown'

                dst_dir = os.path.join(data_dir, split, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, filename)

                if os.path.isfile(src_path):
                    shutil.move(src_path, dst_path)
                    print(f"Moved: {filename} → {class_name}/")

def verify_organization(data_dir="data/"):
    """Verify image organization"""
    print("\n" + "="*70)
    print("Dataset Organization Summary")
    print("="*70 + "\n")

    for split in ['train', 'val', 'test']:
        print(f"{split.upper()}:")
        split_dir = os.path.join(data_dir, split)

        for cls in ['healthy', 'malnourished']:
            class_dir = os.path.join(split_dir, cls)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                print(f"  {cls}: {count} images")
            else:
                print(f"  {cls}: Not found")
        print()

def main():
    """Main function"""
    print("="*70)
    print("Roboflow Dataset Setup")
    print("="*70 + "\n")

    data_dir = "data/"

    if not os.path.exists(data_dir):
        print("✗ data/ directory not found")
        print("Please download dataset first: python scripts/download_dataset.py")
        return

    create_class_folders(data_dir)
    labels = read_roboflow_labels(data_dir)
    organize_images_by_class(data_dir, labels)
    verify_organization(data_dir)

    print("="*70)
    print("✓ Dataset setup complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Train model: python scripts/train.py")
    print("2. Run interface: python ui/gradio_app.py")

if __name__ == "__main__":
    main()
