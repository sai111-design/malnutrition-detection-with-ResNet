#!/usr/bin/env python3
"""
Training script for Roboflow malnutrition detection dataset
Usage: python scripts/train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from src.model import MalnutritionDetector
from src.utils import get_device
from src.data_loader import get_roboflow_dataloaders

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def train_model(data_dir="data/", num_epochs=20, batch_size=32, learning_rate=1e-4):
    """Train the malnutrition detection model"""

    print("="*70)
    print("Training Malnutrition Detection Model")
    print("Dataset: Roboflow malnutrition-detection")
    print("="*70 + "\n")

    if not os.path.exists(os.path.join(data_dir, 'train')):
        print(f"✗ Dataset not found in {data_dir}")
        print("\nDownload dataset first:")
        print("  python scripts/download_dataset.py")
        return

    device = get_device()

    print("\nLoading Roboflow dataset...")
    print("-" * 70)
    train_loader, val_loader, test_loader = get_roboflow_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4
    )

    print("\nInitializing model...")
    print("-" * 70)
    model = MalnutritionDetector(pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print("\nTraining started...")
    print("-" * 70)

    best_val_acc = 0
    best_model_path = "models/detection_model.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Best model saved (Acc: {val_acc:.4f})")

    print("\n" + "="*70)
    print("Testing on test set...")
    print("-" * 70)
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    print("\n" + "="*70)
    print("✓ Training complete!")
    print("="*70)
    print(f"\nBest model saved: {best_model_path}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nNext steps:")
    print("  python ui/gradio_app.py")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train malnutrition detection model")
    parser.add_argument("--data-dir", type=str, default="data/", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
