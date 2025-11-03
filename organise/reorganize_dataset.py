
#!/usr/bin/env python3
"""
Corrected Roboflow Dataset Reorganizer
Handles different folder naming conventions:
- train/valid (TensorFlow format)
- train/val/test (Standard format)
"""

import os
import shutil
from pathlib import Path
import sys

def find_roboflow_folder():
    """Find the Roboflow dataset folder"""

    for base_path in [Path("."), Path("..")]:
        if not base_path.exists():
            continue

        for item in base_path.glob("*tensorflow*"):
            if item.is_dir():
                return item

    return None

def reorganize_roboflow_dataset():
    """Main reorganization function"""

    print("="*70)
    print("Roboflow Dataset Reorganizer (Corrected)")
    print("="*70)

    # Find Roboflow folder
    roboflow_dir = find_roboflow_folder()

    if not roboflow_dir:
        print("\n✗ Could not find Roboflow dataset folder!")
        return False

    # Make sure we're at project root
    data_dir = Path("data")
    if not data_dir.exists():
        print("\n✗ 'data' folder not found")
        return False

    print(f"\n✓ Source: {roboflow_dir.name}")
    print(f"✓ Working in: {Path.cwd()}")

    # Create destination folders
    train_img_dir = data_dir / "train" / "images"
    val_img_dir = data_dir / "val" / "images"
    test_img_dir = data_dir / "test" / "images"

    for folder in [train_img_dir, val_img_dir, test_img_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    print("\n✓ Created destination folders")

    # Image counts
    image_counts = {"train": 0, "val": 0, "test": 0}

    print("\nCopying images...")

    # Handle different naming conventions
    # Roboflow TensorFlow format uses: train, valid, test
    # Standard format uses: train, val, test

    split_mappings = {
        "train": "train",      # train → train
        "valid": "val",        # valid → val (TensorFlow format)
        "val": "val",          # val → val (standard format)
        "test": "test"         # test → test
    }

    for src_split, dst_split in split_mappings.items():
        # Look for images in different possible locations
        possible_paths = [
            roboflow_dir / src_split / "images",
            roboflow_dir / src_split
        ]

        src_path = None
        for p in possible_paths:
            if p.exists() and any(p.glob("*.jpg")) or any(p.glob("*.png")):
                src_path = p
                break

        if src_path:
            dst_path = data_dir / dst_split / "images"
            dst_path.mkdir(parents=True, exist_ok=True)

            # Count and copy images
            for image_file in src_path.glob("*"):
                if image_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    dest_file = dst_path / image_file.name

                    if not dest_file.exists():
                        try:
                            shutil.copy2(str(image_file), str(dest_file))
                            image_counts[dst_split] += 1
                        except Exception as e:
                            print(f"    Error: {e}")

            if image_counts[dst_split] > 0:
                print(f"  ✓ {src_split} → {dst_split}: {image_counts[dst_split]} images")

    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    total = sum(image_counts.values())

    print(f"✓ Train images:  {image_counts['train']}")
    print(f"✓ Val images:    {image_counts['val']}")
    print(f"✓ Test images:   {image_counts['test']}")
    print(f"✓ Total images:  {total}")

    if total > 0:
        print("\n✓ Dataset reorganized successfully!")
        print("\nNext steps:")
        print("1. Create train.py with training code")
        print("2. Run: python train.py")
        return True
    else:
        print("\n✗ No images copied!")
        print("\nDebugging info:")
        print(f"Source folder: {roboflow_dir}")
        print("Contents:")
        for item in roboflow_dir.rglob("*"):
            if item.is_file() and item.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                print(f"  Found image: {item}")
        return False

if __name__ == "__main__":
    success = reorganize_roboflow_dataset()
    sys.exit(0 if success else 1)