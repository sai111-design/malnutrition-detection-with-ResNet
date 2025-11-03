#!/usr/bin/env python3
"""
Diagnostic Script - Helps identify dataset folder location
"""

import os
from pathlib import Path

print("="*70)
print("Dataset Location Diagnostic")
print("="*70)

print(f"\nCurrent directory: {Path.cwd()}")

print("\nDirectory contents:")
for item in sorted(Path(".").iterdir()):
    if item.is_dir():
        size = len(list(item.glob("*")))
        print(f"  📁 {item.name}/ ({size} items)")
    else:
        print(f"  📄 {item.name}")

print("\n" + "="*70)
print("Searching for Roboflow dataset folder...")
print("="*70)

found_folders = []

# Search in current directory and parent
for base in [Path("."), Path("..")]:
    if not base.exists():
        continue

    for item in base.iterdir():
        name = item.name.lower()
        if item.is_dir() and any(x in name for x in ["tensorflow", "malnutrition", "detect", "dataset"]):
            print(f"\nFound: {item.name}")
            print(f"Path: {item.absolute()}")

            # Check for train/val/test
            for split in ["train", "val", "test"]:
                split_path = item / split
                if split_path.exists():
                    images_path = split_path / "images"
                    if images_path.exists():
                        count = len(list(images_path.glob("*")))
                        print(f"  ✓ {split}/images/ ({count} files)")
                    else:
                        print(f"  ✗ {split}/images/ (NOT FOUND)")

            found_folders.append(item)

if not found_folders:
    print("\n✗ No Roboflow dataset folder found!")
    print("\nTroubleshooting:")
    print("1. Check if 'Detect Malnutrition' folder exists")
    print("2. Use: dir (Windows) or ls (Mac/Linux)")
    print("3. Make sure you're in the correct directory")

print("\n" + "="*70)
print("To fix the issue:")
print("="*70)
print("\n1. Navigate to project root:")
print("   cd C:\Users\saini\OneDrive\Desktop\detect")
print("\n2. Check folder name:")
print("   dir")
print("\n3. Run reorganizer:")
print("   python reorganize_dataset.py")
