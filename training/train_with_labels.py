#!/usr/bin/env python3
"""
Updated Training Script with Label Support
Trains on labeled dataset (healthy vs malnourished)

Key anti-overfitting techniques applied:
  - Transfer learning with layer freezing (freeze conv1–layer2 initially)
  - Aggressive data augmentation (affine, perspective, blur, erasing)
  - Dropout (0.3) on classifier head
  - Label smoothing (0.1)
  - Higher weight_decay (1e-4)
  - Cosine annealing LR schedule
  - Reproducibility seeds
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
import numpy as np
from tqdm import tqdm
import json
import csv
from pathlib import Path
import pandas as pd

# ============================================================================
# 0. REPRODUCIBILITY
# ============================================================================

SEED = 42

def set_seed(seed=SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

print("\n" + "="*70)
print("Malnutrition Detection - Training with Labels")
print("="*70)
print(f"  Random seed: {SEED} (deterministic mode ON)")

# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Using device: {device}")

os.makedirs('models', exist_ok=True)
os.makedirs('outputs/logs', exist_ok=True)

# --- Hyper-parameters (documented for transparency) ---
BATCH_SIZE = 8
NUM_EPOCHS = 25          # Increased from 10 for better convergence
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4      # Increased from 1e-5 for stronger regularization
LABEL_SMOOTHING = 0.1    # Prevents overconfident predictions
PATIENCE = 10            # Allow enough room after unfreeze
UNFREEZE_EPOCH = 3       # Unfreeze early — frozen phase converges fast
DROPOUT_RATE = 0.3       # Dropout on classifier head
FINETUNE_LR_MULT = 0.05  # Multiplier for LR after unfreezing backbone

config = {
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "label_smoothing": LABEL_SMOOTHING,
    "patience": PATIENCE,
    "unfreeze_epoch": UNFREEZE_EPOCH,
    "dropout_rate": DROPOUT_RATE,
    "optimizer": "Adam",
    "scheduler": "CosineAnnealingLR",
    "model": "ResNet-50",
    "pretrained_weights": "IMAGENET1K_V1",
    "input_size": 224,
    "num_classes": 2,
    "device": str(device),
}

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
# 4. DATA TRANSFORMS — strengthened augmentation
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),                        # Resize slightly larger for random crop
    transforms.RandomCrop(224),                           # Random 224×224 crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),                 # Mild vertical flip
    transforms.RandomRotation(degrees=15),                # Increased from 10
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # Cutout-style regularization
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("\n✓ Data transforms created (strengthened augmentation)")

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

config["data"] = {
    "train_total": len(train_dataset),
    "train_healthy": train_healthy,
    "train_malnourished": train_malnourished,
    "val_total": len(val_dataset),
    "val_healthy": val_healthy,
    "val_malnourished": val_malnourished,
}

# ============================================================================
# 6. CREATE DATALOADERS
# ============================================================================

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\n✓ Train loader: {len(train_loader)} batches")
print(f"✓ Val loader: {len(val_loader)} batches")

# ============================================================================
# 7. BUILD MODEL — transfer learning with layer freezing + dropout
# ============================================================================

print("\n" + "-"*70)
print("Building Model")
print("-"*70)

# Load ResNet-50 with ImageNet-1K V1 pretrained weights
# (torchvision ≥0.13: models.resnet50(weights='IMAGENET1K_V1'))
model = models.resnet50(pretrained=True)

# Replace the final FC layer with a custom head including dropout
model.fc = nn.Sequential(
    nn.Dropout(p=DROPOUT_RATE),
    nn.Linear(model.fc.in_features, 2)       # 2048 → 2 (binary classification)
)
model = model.to(device)

print("✓ ResNet-50 model created")
print(f"  Pretrained weights: IMAGENET1K_V1 (ImageNet)")
print(f"  Classifier head: Dropout({DROPOUT_RATE}) → Linear(2048, 2)")

# --- Freeze early layers (conv1, bn1, layer1, layer2) ---
def freeze_backbone(model):
    """Freeze conv1 through layer2, keep layer3/layer4/fc trainable."""
    frozen_layers = ['conv1', 'bn1', 'layer1', 'layer2']
    for name, param in model.named_parameters():
        if any(name.startswith(layer) for layer in frozen_layers):
            param.requires_grad = False

def unfreeze_all(model):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True

freeze_backbone(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Frozen parameters: {frozen_params:,}")
print(f"  (Backbone frozen until epoch {UNFREEZE_EPOCH})")

config["model_params"] = {
    "total": total_params,
    "trainable_initial": trainable_params,
    "frozen_initial": frozen_params,
}

# ============================================================================
# 8. LOSS AND OPTIMIZER
# ============================================================================

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

print(f"\n✓ Loss: CrossEntropyLoss (label_smoothing={LABEL_SMOOTHING})")
print(f"✓ Optimizer: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
print(f"✓ Scheduler: CosineAnnealingLR (T_max={NUM_EPOCHS})")

# ============================================================================
# 9. TRAINING LOOP
# ============================================================================

print("\n" + "="*70)
print("Starting Training with Labeled Data")
print("="*70 + "\n")

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': [], 'overfit_gap': []}
best_val_loss = float('inf')
best_val_acc = 0.0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    # --- Unfreeze backbone at the scheduled epoch ---
    if epoch == UNFREEZE_EPOCH:
        print(f"\n  ⚡ Unfreezing full backbone at epoch {epoch + 1}")
        unfreeze_all(model)
        # Re-create optimizer with a much lower LR for pretrained layers
        finetune_lr = LEARNING_RATE * FINETUNE_LR_MULT
        optimizer = optim.Adam(model.parameters(), lr=finetune_lr, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS - UNFREEZE_EPOCH, eta_min=1e-7
        )
        unfrozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters now: {unfrozen_trainable:,}\n")

    current_lr = optimizer.param_groups[0]['lr']

    # --- Train phase ---
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

    # --- Validation phase ---
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
    overfit_gap = train_acc - val_acc

    print(f'Epoch {epoch+1:2d}/{NUM_EPOCHS} | '
          f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | '
          f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | '
          f'Gap: {overfit_gap:+.1f}% | LR: {current_lr:.2e}')

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    history['overfit_gap'].append(overfit_gap)

    # Save best model (by val loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'models/best_model.pth')
        print(f'  ↳ Best model saved (loss)! Val Loss: {val_loss:.4f}')
    else:
        patience_counter += 1

    # Also save checkpoint with best val accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/best_model_acc.pth')
        print(f'  ↳ Best accuracy checkpoint! Val Acc: {val_acc:.2f}%')

    if patience_counter >= PATIENCE:
        print(f'\n⚠ Early stopping triggered at epoch {epoch + 1}')
        break

    scheduler.step()

# ============================================================================
# 10. SAVE MODELS AND HISTORY
# ============================================================================

print("\n" + "="*70)
print("Training Complete!")
print("="*70)

# Load the best model — prefer best-accuracy if it outperforms best-loss
if os.path.exists('models/best_model_acc.pth') and best_val_acc >= 88.0:
    print(f"  Using best-accuracy checkpoint (Val Acc: {best_val_acc:.2f}%)")
    model.load_state_dict(torch.load('models/best_model_acc.pth', map_location=device))
elif os.path.exists('models/best_model.pth'):
    print(f"  Using best-loss checkpoint (Val Loss: {best_val_loss:.4f})")
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))

torch.save(model.state_dict(), 'models/malnutrition_model.pth')
print(f"\n✓ Model saved to: models/malnutrition_model.pth")

with open('outputs/logs/training_history.json', 'w') as f:
    json.dump(history, f, indent=4)

print(f"✓ Training history saved")

# Save full training config for reproducibility
with open('outputs/logs/training_config.json', 'w') as f:
    json.dump(config, f, indent=4)

print(f"✓ Training config saved to: outputs/logs/training_config.json")

# ============================================================================
# 11. SUMMARY
# ============================================================================

print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"\nBest Validation Loss: {best_val_loss:.4f}")
print(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
print(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%")
final_gap = history['train_acc'][-1] - history['val_acc'][-1]
print(f"Final Overfit Gap: {final_gap:+.2f}%")

if final_gap > 15:
    print("  ⚠ HIGH overfitting — consider more data or regularisation")
elif final_gap > 8:
    print("  ⚠ Moderate overfitting — acceptable for small dataset")
else:
    print("  ✓ Good generalisation")

print(f"\nAnti-Overfitting Techniques Applied:")
print(f"  ✓ Layer freezing (conv1–layer2 frozen for first {UNFREEZE_EPOCH} epochs)")
print(f"  ✓ Dropout ({DROPOUT_RATE}) on classifier head")
print(f"  ✓ Label smoothing ({LABEL_SMOOTHING})")
print(f"  ✓ Weight decay ({WEIGHT_DECAY})")
print(f"  ✓ Strong data augmentation (13 transforms)")
print(f"  ✓ Cosine annealing LR schedule")
print(f"  ✓ Early stopping (patience={PATIENCE})")

print(f"\nLabel Distribution Used:")
print(f"  Train: {train_healthy} healthy, {train_malnourished} malnourished")
print(f"  Val: {val_healthy} healthy, {val_malnourished} malnourished")
print(f"\n✓ Model ready for evaluation & deployment!")
print("="*70 + "\n")
