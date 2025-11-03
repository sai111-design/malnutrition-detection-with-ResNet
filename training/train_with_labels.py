#!/usr/bin/env python3
"""
Updated Training Script with Label Support
Trains on labeled dataset (healthy vs malnourished)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from tqdm import tqdm
import json
import csv
from pathlib import Path
import pandas as pd

print("\n" + "="*70)
print("Malnutrition Detection - Training with Labels")
print("="*70)

# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Using device: {device}")

os.makedirs('models', exist_ok=True)
os.makedirs('outputs/logs', exist_ok=True)

BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

# ============================================================================
# 2. LOAD LABELS
# ============================================================================

def load_labels(labels_file='data/labels.csv'):
    """Load labels from CSV file"""

    if not os.path.exists(labels_file):
        print(f"\n✗ Labels file not found: {labels_file}")
        print("Please run: python auto_label_images.py")
        return None

    try:
        df = pd.read_csv(labels_file)
        labels_dict = dict(zip(df['filename'], df['label']))
        print(f"✓ Loaded {len(labels_dict)} labels from {labels_file}")
        return labels_dict
    except Exception as e:
        print(f"✗ Error loading labels: {e}")
        return None

labels = load_labels()
if labels is None:
    print("\n✗ Cannot proceed without labels!")
    exit(1)

# ============================================================================
# 3. UPDATED DATASET CLASS WITH LABELS
# ============================================================================

class MalnutritionDataset(Dataset):
    """Dataset with proper label support"""

    def __init__(self, img_dir, split_type, labels_dict=None, transform=None):
        self.img_dir = img_dir
        self.split_type = split_type
        self.labels_dict = labels_dict or {}
        self.transform = transform
        self.images = []

        for filename in sorted(os.listdir(img_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.images.append(filename)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.img_dir, img_file)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load {img_path}")
            return None, None

        if self.transform:
            image = self.transform(image)

        # Get label from dictionary
        key = f"{self.split_type}/{img_file}"
        label = self.labels_dict.get(key, 0)

        return image, label

# ============================================================================
# 4. DATA TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("\n✓ Data transforms created")

# ============================================================================
# 5. LOAD DATASETS
# ============================================================================

print("\n" + "-"*70)
print("Loading Datasets")
print("-"*70)

train_dataset = MalnutritionDataset('data/train/images', 'train', labels, train_transform)
val_dataset = MalnutritionDataset('data/val/images', 'val', labels, val_transform)

print(f"✓ Train dataset: {len(train_dataset)} images")
print(f"✓ Val dataset: {len(val_dataset)} images")

# Print label distribution
train_labels = [train_dataset.labels_dict.get(f'train/{img}', 0) for img in train_dataset.images]
val_labels = [val_dataset.labels_dict.get(f'val/{img}', 0) for img in val_dataset.images]

train_healthy = sum(1 for l in train_labels if l == 0)
train_malnourished = len(train_labels) - train_healthy
val_healthy = sum(1 for l in val_labels if l == 0)
val_malnourished = len(val_labels) - val_healthy

print(f"\nTrain Label Distribution:")
print(f"  Healthy: {train_healthy}, Malnourished: {train_malnourished}")
print(f"\nVal Label Distribution:")
print(f"  Healthy: {val_healthy}, Malnourished: {val_malnourished}")

# ============================================================================
# 6. CREATE DATALOADERS
# ============================================================================

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\n✓ Train loader: {len(train_loader)} batches")
print(f"✓ Val loader: {len(val_loader)} batches")

# ============================================================================
# 7. BUILD MODEL
# ============================================================================

print("\n" + "-"*70)
print("Building Model")
print("-"*70)

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

print("✓ ResNet50 model created")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# ============================================================================
# 8. LOSS AND OPTIMIZER
# ============================================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print("\n✓ Loss: CrossEntropyLoss")
print("✓ Optimizer: Adam")

# ============================================================================
# 9. TRAINING LOOP
# ============================================================================

print("\n" + "="*70)
print("Starting Training with Labeled Data")
print("="*70 + "\n")

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    # Train phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False):
        images, batch_labels = images.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += batch_labels.size(0)
        train_correct += (predicted == batch_labels).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100 * train_correct / train_total

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, batch_labels in tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]', leave=False):
            images, batch_labels = images.to(device), batch_labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, batch_labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_labels.size(0)
            val_correct += (predicted == batch_labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total

    print(f'Epoch {epoch+1:2d}/{NUM_EPOCHS} | '
          f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | '
          f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}%')

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'models/best_model.pth')
        print(f'  ↳ Best model saved! Val Loss: {val_loss:.4f}')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'\n⚠ Early stopping triggered')
        break

    scheduler.step()

# ============================================================================
# 10. SAVE MODELS AND HISTORY
# ============================================================================

print("\n" + "="*70)
print("Training Complete!")
print("="*70)

if os.path.exists('models/best_model.pth'):
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))

torch.save(model.state_dict(), 'models/malnutrition_model.pth')
print(f"\n✓ Model saved to: models/malnutrition_model.pth")

with open('outputs/logs/training_history.json', 'w') as f:
    json.dump(history, f, indent=4)

print(f"✓ Training history saved")

# ============================================================================
# 11. SUMMARY
# ============================================================================

print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"\nBest Validation Loss: {best_val_loss:.4f}")
print(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
print(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%")
print(f"\nLabel Distribution Used:")
print(f"  Train: {train_healthy} healthy, {train_malnourished} malnourished")
print(f"  Val: {val_healthy} healthy, {val_malnourished} malnourished")
print(f"\n✓ Model ready for deployment!")
print("="*70 + "\n")
