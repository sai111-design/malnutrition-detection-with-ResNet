#!/usr/bin/env python3
"""
Automated Image Labeling Tool with Multiple Strategies
Separates images into labeled datasets for malnutrition detection
"""

import os
import csv
import random
from pathlib import Path
from PIL import Image
import shutil

print("\n" + "="*70)
print("Automated Image Labeling Tool")
print("="*70)

# ============================================================================
# 1. INTERACTIVE LABELING MODE
# ============================================================================

def interactive_labeling():
    """
    Interactive mode: Show images and ask user to label each one
    """

    print("\n" + "-"*70)
    print("Interactive Labeling Mode")
    print("-"*70)
    print("\nYou will be shown images one by one.")
    print("For each image, enter: 0 (Healthy) or 1 (Malnourished)")
    print("Press 'q' to quit, 's' to skip, 'b' to go back\n")

    train_dir = 'data/train/images'
    val_dir = 'data/val/images'

    all_labels = {}

    for split, directory in [('train', train_dir), ('val', val_dir)]:
        if not os.path.exists(directory):
            print(f"✗ Directory not found: {directory}")
            continue

        print(f"\nLabeling {split} images...")
        images = sorted(os.listdir(directory))

        if not images:
            print(f"No images found in {directory}")
            continue

        for idx, img_file in enumerate(images):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(directory, img_file)
            full_path = f"{split}/{img_file}"

            # Show image info
            try:
                img = Image.open(img_path)
                size = img.size
                print(f"\n[{idx+1}/{len(images)}] {img_file} ({size[0]}x{size[1]})")
            except:
                print(f"\n[{idx+1}/{len(images)}] {img_file} (couldn't load)")

            # Get user input
            while True:
                try:
                    label = input("Label (0=Healthy, 1=Malnourished, q=quit, s=skip, b=back): ").strip()

                    if label == 'q':
                        return all_labels
                    elif label == 's':
                        break
                    elif label == 'b' and idx > 0:
                        idx -= 2
                        break
                    elif label in ['0', '1']:
                        all_labels[full_path] = int(label)
                        print(f"  ✓ Labeled as {'Healthy' if label == '0' else 'Malnourished'}")
                        break
                    else:
                        print("  Invalid input. Please enter 0, 1, q, s, or b")
                except KeyboardInterrupt:
                    return all_labels

        print(f"\n✓ Completed {split} set: {len([l for k,l in all_labels.items() if k.startswith(split)])} labeled")

    return all_labels

# ============================================================================
# 2. STRATEGY-BASED LABELING (Automatic)
# ============================================================================

def strategy_based_labeling():
    """
    Automatic labeling based on image characteristics:
    - Small file size → Malnourished
    - Large file size → Healthy
    - (Can be customized with your own rules)
    """

    print("\n" + "-"*70)
    print("Strategy-Based Automatic Labeling")
    print("-"*70)

    all_labels = {}

    for split_name, directory in [('train', 'data/train/images'), ('val', 'data/val/images')]:
        if not os.path.exists(directory):
            print(f"✗ Directory not found: {directory}")
            continue

        print(f"\nAnalyzing {split_name} images...")
        images = sorted(os.listdir(directory))
        file_sizes = []

        # Analyze file sizes
        for img_file in images:
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(directory, img_file)
            file_size = os.path.getsize(img_path)
            file_sizes.append((img_file, file_size))

        if not file_sizes:
            continue

        # Sort by file size
        file_sizes.sort(key=lambda x: x[1])

        # Split: bottom 50% = Malnourished, top 50% = Healthy
        threshold_idx = len(file_sizes) // 2

        for img_file, size in file_sizes:
            full_path = f"{split_name}/{img_file}"
            file_idx = file_sizes.index((img_file, size))

            # Label based on file size threshold
            label = 1 if file_idx < threshold_idx else 0
            all_labels[full_path] = label
            print(f"  {img_file}: {size:,} bytes → {'Malnourished' if label == 1 else 'Healthy'}")

        print(f"\n✓ {split_name}: {len(file_sizes)} images labeled")

    return all_labels

# ============================================================================
# 3. BALANCED RANDOM LABELING
# ============================================================================

def balanced_random_labeling():
    """
    Generate balanced random labels (50% healthy, 50% malnourished)
    """

    print("\n" + "-"*70)
    print("Balanced Random Labeling")
    print("-"*70)

    all_labels = {}

    for split_name, directory in [('train', 'data/train/images'), ('val', 'data/val/images')]:
        if not os.path.exists(directory):
            print(f"✗ Directory not found: {directory}")
            continue

        print(f"\nLabeling {split_name} images (random balanced)...")
        images = sorted([f for f in os.listdir(directory) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        if not images:
            print(f"  No images found in {directory}")
            continue

        # Create balanced labels
        num_images = len(images)
        num_malnourished = num_images // 2

        labels_list = [1] * num_malnourished + [0] * (num_images - num_malnourished)
        random.shuffle(labels_list)

        for img_file, label in zip(images, labels_list):
            full_path = f"{split_name}/{img_file}"
            all_labels[full_path] = label

        healthy = sum(1 for l in labels_list if l == 0)
        malnourished = num_images - healthy

        print(f"  Healthy: {healthy}, Malnourished: {malnourished}")
        print(f"✓ {split_name}: {num_images} images labeled")

    return all_labels

# ============================================================================
# 4. SAVE LABELS TO CSV
# ============================================================================

def save_labels_to_csv(labels_dict, filename='data/labels.csv'):
    """
    Save labels to CSV file
    """

    print(f"\nSaving labels to {filename}...")

    # Create data directory if needed
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])  # Header

        for filepath, label in sorted(labels_dict.items()):
            writer.writerow([filepath, label])

    print(f"✓ Saved {len(labels_dict)} labels to {filename}")

    # Print statistics
    healthy_count = sum(1 for l in labels_dict.values() if l == 0)
    malnourished_count = len(labels_dict) - healthy_count

    print(f"\nLabel Statistics:")
    print(f"  Healthy: {healthy_count} ({100*healthy_count/len(labels_dict):.1f}%)")
    print(f"  Malnourished: {malnourished_count} ({100*malnourished_count/len(labels_dict):.1f}%)")

# ============================================================================
# 5. ORGANIZE INTO FOLDERS
# ============================================================================

def organize_by_labels(labels_dict):
    """
    Create separate folders for healthy and malnourished images
    """

    print("\nOrganizing images into labeled folders...")

    # Create output directories
    healthy_dir = 'data/labeled/healthy'
    malnourished_dir = 'data/labeled/malnourished'

    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(malnourished_dir, exist_ok=True)

    for filepath, label in labels_dict.items():
        # Parse the path
        parts = filepath.split('/')
        split_type = parts[0]  # 'train' or 'val'
        filename = parts[1]

        src_path = os.path.join(f'data/{split_type}/images', filename)

        if label == 0:
            dst_path = os.path.join(healthy_dir, f"{split_type}_{filename}")
        else:
            dst_path = os.path.join(malnourished_dir, f"{split_type}_{filename}")

        try:
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"  Warning: Could not copy {filename}: {e}")

    healthy_count = len(os.listdir(healthy_dir))
    malnourished_count = len(os.listdir(malnourished_dir))

    print(f"\n✓ Organized images:")
    print(f"  Healthy folder: {healthy_count} images")
    print(f"  Malnourished folder: {malnourished_count} images")

# ============================================================================
# 6. MAIN MENU
# ============================================================================

def main():
    """Main menu"""

    print("\n" + "="*70)
    print("Choose Labeling Method")
    print("="*70)
    print("""
1. Interactive Labeling (Manual - Recommended for accuracy)
   - Shows each image and asks you to label it
   - Best for small datasets (under 100 images)

2. Strategy-Based Labeling (Automatic - File size based)
   - Analyzes image file sizes
   - Smaller files → Malnourished
   - Larger files → Healthy
   - Fast but may be inaccurate

3. Balanced Random Labeling (Quick testing)
   - Randomly assigns 50% healthy, 50% malnourished
   - Good for testing the pipeline
   - Use real labels later

4. Quit
    """)

    choice = input("Enter your choice (1-4): ").strip()

    if choice == '1':
        labels = interactive_labeling()
    elif choice == '2':
        labels = strategy_based_labeling()
    elif choice == '3':
        labels = balanced_random_labeling()
    elif choice == '4':
        print("\nExiting...")
        return
    else:
        print("Invalid choice. Using balanced random labeling...")
        labels = balanced_random_labeling()

    if not labels:
        print("\n✗ No labels created!")
        return

    # Save to CSV
    save_labels_to_csv(labels)

    # Ask if user wants to organize into folders
    organize = input("\nOrganize images into separate folders? (y/n): ").strip().lower()
    if organize == 'y':
        organize_by_labels(labels)

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Review the generated labels.csv file")
    print("2. If needed, edit labels manually in data/labels.csv")
    print("3. Update train.py with the new dataset class")
    print("4. Run: python train.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
