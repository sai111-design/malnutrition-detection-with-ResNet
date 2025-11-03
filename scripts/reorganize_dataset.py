#!/usr/bin/env python3
"""
Reorganize Roboflow Dataset Structure
Converts the Roboflow tensorflow format to our project structure
"""

import os
import shutil
from pathlib import Path
import sys

def reorganize_roboflow_dataset():
    """
    Reorganize Roboflow dataset from Detect Malnutrition.v1i.tensorflow 
    to data/train/images, data/val/images, data/test/images structure
    """

    print("="*70)
    print("Roboflow Dataset Reorganizer")
    print("="*70)

    # Paths
    roboflow_dir = Path("Detect Malnutrition.v1i.tensorflow (1)")
    data_dir = Path("data")

    # Check if source exists
    if not roboflow_dir.exists():
        print(f"✗ Source folder not found: {roboflow_dir}")
        print("\nLooking for similar folders...")
        for item in Path(".").glob("*tensorflow*"):
            print(f"  Found: {item}")
        return False

    print(f"\n✓ Found source: {roboflow_dir}")

    # Create destination folders
    train_img_dir = data_dir / "train" / "images"
    val_img_dir = data_dir / "val" / "images"
    test_img_dir = data_dir / "test" / "images"

    for folder in [train_img_dir, val_img_dir, test_img_dir]:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {folder}")

    # Find and organize images
    image_counts = {"train": 0, "val": 0, "test": 0}

    print("\nSearching for images in Roboflow structure...")

    # Method 1: Look for split folders (train, val, test)
    for split in ["train", "val", "test"]:
        split_paths = list(roboflow_dir.glob(f"*/{split}/images"))
        split_paths.extend(roboflow_dir.glob(f"{split}/images"))
        split_paths.extend(roboflow_dir.glob(f"{split}"))

        if split_paths:
            src_path = split_paths[0]
            print(f"  Found {split}: {src_path}")

            # Find images
            image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

            if src_path.is_dir():
                for image_file in src_path.glob('*'):
                    if image_file.suffix.lower() in image_extensions:
                        dest_path = Path(data_dir) / split / "images" / image_file.name

                        # Copy image
                        if dest_path.exists():
                            print(f"    Already exists: {image_file.name}")
                        else:
                            shutil.copy2(str(image_file), str(dest_path))
                            image_counts[split] += 1

                print(f"  → Copied {image_counts[split]} images to data/{split}/images")

    # Method 2: If no split folders found, look for images recursively
    if sum(image_counts.values()) == 0:
        print("\n  No split structure found, searching for images...")

        for image_file in roboflow_dir.glob("**/images/*.jpg"):
            # Determine split from path
            path_str = str(image_file)
            if "train" in path_str.lower():
                dest = train_img_dir / image_file.name
                split = "train"
            elif "val" in path_str.lower():
                dest = val_img_dir / image_file.name
                split = "val"
            elif "test" in path_str.lower():
                dest = test_img_dir / image_file.name
                split = "test"
            else:
                dest = train_img_dir / image_file.name
                split = "train"

            if not dest.exists():
                shutil.copy2(str(image_file), str(dest))
                image_counts[split] += 1

        for image_file in roboflow_dir.glob("**/images/*.png"):
            # Similar logic for PNG
            path_str = str(image_file)
            if "train" in path_str.lower():
                dest = train_img_dir / image_file.name
                split = "train"
            elif "val" in path_str.lower():
                dest = val_img_dir / image_file.name
                split = "val"
            elif "test" in path_str.lower():
                dest = test_img_dir / image_file.name
                split = "test"
            else:
                dest = train_img_dir / image_file.name
                split = "train"

            if not dest.exists():
                shutil.copy2(str(image_file), str(dest))
                image_counts[split] += 1

    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"✓ Train images: {image_counts['train']}")
    print(f"✓ Val images: {image_counts['val']}")
    print(f"✓ Test images: {image_counts['test']}")
    print(f"✓ Total images: {sum(image_counts.values())}")

    if sum(image_counts.values()) > 0:
        print("\n✓ Dataset reorganized successfully!")
        return True
    else:
        print("\n✗ No images found. Please check the source folder structure.")
        return False

if __name__ == "__main__":
    success = reorganize_roboflow_dataset()
    sys.exit(0 if success else 1)
