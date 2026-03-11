#!/usr/bin/env python3
"""
Training History Visualization Script
Generates professional accuracy & loss curves from training history JSON
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from pathlib import Path

print("\n" + "="*70)
print("Training History Visualization")
print("="*70)

# ============================================================================
# 1. LOAD TRAINING HISTORY
# ============================================================================

print("\n✓ Loading training history...")

history_file = 'outputs/logs/training_history.json'

if not os.path.exists(history_file):
    print(f"\n✗ Training history file not found: {history_file}")
    print("\nTo generate training history:")
    print("  1. Run: python train_with_labels.py")
    print("  2. This will create: outputs/logs/training_history.json")
    exit(1)

try:
    with open(history_file, 'r') as f:
        history = json.load(f)
    print(f"✓ Loaded training history from: {history_file}")
except json.JSONDecodeError:
    print(f"✗ Error: Invalid JSON file")
    exit(1)

# Extract data
train_loss = history.get('train_loss', [])
val_loss = history.get('val_loss', [])
train_acc = history.get('train_acc', [])
val_acc = history.get('val_acc', [])

num_epochs = len(train_loss)
epochs = np.arange(1, num_epochs + 1)

print(f"✓ Epochs: {num_epochs}")
print(f"✓ Training samples: {len(train_loss)}")
print(f"✓ Validation samples: {len(val_loss)}")

# ============================================================================
# 2. CREATE OUTPUT DIRECTORY
# ============================================================================

os.makedirs('outputs/images', exist_ok=True)
print("\n✓ Output directory ready: outputs/images/")

# ============================================================================
# 3. PLOT 1: ACCURACY CURVES
# ============================================================================

print("\n" + "-"*70)
print("Generating Accuracy Curves")
print("-"*70)

plt.figure(figsize=(14, 6))

# Plot training accuracy
plt.plot(epochs, train_acc, 'b-', linewidth=2.5, marker='o', 
         markersize=4, label='Training Accuracy', alpha=0.8)

# Plot validation accuracy
plt.plot(epochs, val_acc, 'g-', linewidth=2.5, marker='s', 
         markersize=4, label='Validation Accuracy', alpha=0.8)

# Styling
plt.xlabel('Epoch', fontsize=13, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
plt.title('Training & Validation Accuracy Across Epochs', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim([min(min(train_acc), min(val_acc)) - 5, 105])

# Add epoch markers at 0, 25, 50, 75, etc
plt.xticks(np.arange(0, num_epochs + 1, max(5, num_epochs // 10)))

# Add value annotations at key epochs
for idx in [0, num_epochs // 4, num_epochs // 2, 3 * num_epochs // 4, num_epochs - 1]:
    if idx < len(train_acc):
        plt.annotate(f'{train_acc[idx]:.1f}%', 
                    xy=(idx+1, train_acc[idx]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, color='blue', alpha=0.7)
        plt.annotate(f'{val_acc[idx]:.1f}%', 
                    xy=(idx+1, val_acc[idx]),
                    xytext=(0, -15), textcoords='offset points',
                    ha='center', fontsize=9, color='green', alpha=0.7)

plt.tight_layout()
plt.savefig('outputs/images/training_accuracy_curves.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: outputs/images/training_accuracy_curves.png")
plt.close()

# ============================================================================
# 4. PLOT 2: LOSS CURVES
# ============================================================================

print("\n" + "-"*70)
print("Generating Loss Curves")
print("-"*70)

plt.figure(figsize=(14, 6))

# Plot training loss
plt.plot(epochs, train_loss, 'r-', linewidth=2.5, marker='o', 
         markersize=4, label='Training Loss', alpha=0.8)

# Plot validation loss
plt.plot(epochs, val_loss, 'orange', linewidth=2.5, marker='s', 
         markersize=4, label='Validation Loss', alpha=0.8)

# Styling
plt.xlabel('Epoch', fontsize=13, fontweight='bold')
plt.ylabel('Loss', fontsize=13, fontweight='bold')
plt.title('Training & Validation Loss Across Epochs', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')

# Add epoch markers
plt.xticks(np.arange(0, num_epochs + 1, max(5, num_epochs // 10)))

# Add value annotations at key epochs
for idx in [0, num_epochs // 4, num_epochs // 2, 3 * num_epochs // 4, num_epochs - 1]:
    if idx < len(train_loss):
        plt.annotate(f'{train_loss[idx]:.3f}', 
                    xy=(idx+1, train_loss[idx]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, color='red', alpha=0.7)
        plt.annotate(f'{val_loss[idx]:.3f}', 
                    xy=(idx+1, val_loss[idx]),
                    xytext=(0, -15), textcoords='offset points',
                    ha='center', fontsize=9, color='orange', alpha=0.7)

plt.tight_layout()
plt.savefig('outputs/images/training_loss_curves.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: outputs/images/training_loss_curves.png")
plt.close()

# ============================================================================
# 5. PLOT 3: COMBINED SUBPLOT (PROFESSIONAL LAYOUT)
# ============================================================================

print("\n" + "-"*70)
print("Generating Combined Subplot")
print("-"*70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left subplot: Accuracy
ax1.plot(epochs, train_acc, 'b-', linewidth=2.5, marker='o', 
         markersize=4, label='Training Accuracy', alpha=0.8)
ax1.plot(epochs, val_acc, 'g-', linewidth=2.5, marker='s', 
         markersize=4, label='Validation Accuracy', alpha=0.8)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim([min(min(train_acc), min(val_acc)) - 5, 105])

# Right subplot: Loss
ax2.plot(epochs, train_loss, 'r-', linewidth=2.5, marker='o', 
         markersize=4, label='Training Loss', alpha=0.8)
ax2.plot(epochs, val_loss, 'orange', linewidth=2.5, marker='s', 
         markersize=4, label='Validation Loss', alpha=0.8)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Loss Curves', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--')

# Overall title
fig.suptitle('Training History: Accuracy & Loss Curves', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('outputs/images/training_history_combined.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: outputs/images/training_history_combined.png")
plt.close()

# ============================================================================
# 6. PLOT 4: COMPREHENSIVE ANALYSIS
# ============================================================================

print("\n" + "-"*70)
print("Generating Comprehensive Analysis")
print("-"*70)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Top left: Accuracy curves
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, train_acc, 'b-', linewidth=2, marker='o', markersize=3, label='Training', alpha=0.8)
ax1.plot(epochs, val_acc, 'g-', linewidth=2, marker='s', markersize=3, label='Validation', alpha=0.8)
ax1.set_title('Accuracy Curves', fontweight='bold', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([min(min(train_acc), min(val_acc)) - 5, 105])

# Top right: Loss curves
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, train_loss, 'r-', linewidth=2, marker='o', markersize=3, label='Training', alpha=0.8)
ax2.plot(epochs, val_loss, 'orange', linewidth=2, marker='s', markersize=3, label='Validation', alpha=0.8)
ax2.set_title('Loss Curves', fontweight='bold', fontsize=12)
ax2.set_ylabel('Loss', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Middle left: Training accuracy improvement
ax3 = fig.add_subplot(gs[1, 0])
acc_improvement = np.array(train_acc) - train_acc[0]
ax3.bar(epochs, acc_improvement, color='skyblue', alpha=0.7, edgecolor='blue', linewidth=1.5)
ax3.set_title('Training Accuracy Improvement', fontweight='bold', fontsize=12)
ax3.set_ylabel('Improvement (%)', fontsize=11)
ax3.set_xlabel('Epoch', fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

# Middle right: Loss reduction
ax4 = fig.add_subplot(gs[1, 1])
loss_reduction = np.array(train_loss[0]) - np.array(train_loss)
ax4.bar(epochs, loss_reduction, color='lightcoral', alpha=0.7, edgecolor='red', linewidth=1.5)
ax4.set_title('Training Loss Reduction', fontweight='bold', fontsize=12)
ax4.set_ylabel('Loss Reduction', fontsize=11)
ax4.set_xlabel('Epoch', fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')

# Bottom left: Gap between train/val accuracy
ax5 = fig.add_subplot(gs[2, 0])
gap_acc = np.abs(np.array(train_acc) - np.array(val_acc))
ax5.plot(epochs, gap_acc, 'purple', linewidth=2, marker='D', markersize=4, label='Train-Val Gap')
ax5.fill_between(epochs, gap_acc, alpha=0.3, color='purple')
ax5.set_title('Overfitting Indicator (Accuracy Gap)', fontweight='bold', fontsize=12)
ax5.set_ylabel('Gap (%)', fontsize=11)
ax5.set_xlabel('Epoch', fontsize=11)
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=10)

# Bottom right: Statistics table
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')

# Calculate statistics
final_train_acc = train_acc[-1]
final_val_acc = val_acc[-1]
final_train_loss = train_loss[-1]
final_val_loss = val_loss[-1]
best_train_acc = max(train_acc)
best_val_acc = max(val_acc)
best_epoch = val_acc.index(best_val_acc) + 1

stats_text = f"""
TRAINING STATISTICS

Final Epoch: {num_epochs}

Accuracy:
  Train:       {final_train_acc:.2f}%
  Validation:  {final_val_acc:.2f}%
  Best Val:    {best_val_acc:.2f}% (Epoch {best_epoch})

Loss:
  Train:       {final_train_loss:.4f}
  Validation:  {final_val_loss:.4f}

Improvement:
  Accuracy:    {final_train_acc - train_acc[0]:.2f}%
  Loss:        {train_loss[0] - final_train_loss:.4f}

Overfitting:
  Gap:         {abs(final_train_acc - final_val_acc):.2f}%
  Status:      {"Minimal" if abs(final_train_acc - final_val_acc) < 5 else "Moderate" if abs(final_train_acc - final_val_acc) < 10 else "Significant"}
"""

ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('Training History - Comprehensive Analysis', fontsize=16, fontweight='bold')
plt.savefig('outputs/images/training_history_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: outputs/images/training_history_analysis.png")
plt.close()

# ============================================================================
# 7. GENERATE SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("TRAINING SUMMARY STATISTICS")
print("="*70)

print(f"""
Total Epochs: {num_epochs}

ACCURACY:
  Initial Train:     {train_acc[0]:.2f}%
  Final Train:       {train_acc[-1]:.2f}%
  Improvement:       {train_acc[-1] - train_acc[0]:.2f}%

  Initial Val:       {val_acc[0]:.2f}%
  Final Val:         {val_acc[-1]:.2f}%
  Improvement:       {val_acc[-1] - val_acc[0]:.2f}%

  Best Val Acc:      {max(val_acc):.2f}% (Epoch {val_acc.index(max(val_acc)) + 1})

LOSS:
  Initial Train:     {train_loss[0]:.4f}
  Final Train:       {train_loss[-1]:.4f}
  Reduction:         {train_loss[0] - train_loss[-1]:.4f}

  Initial Val:       {val_loss[0]:.4f}
  Final Val:         {val_loss[-1]:.4f}
  Reduction:         {val_loss[0] - val_loss[-1]:.4f}

  Best Val Loss:     {min(val_loss):.4f} (Epoch {val_loss.index(min(val_loss)) + 1})

OVERFITTING ANALYSIS:
  Final Gap (Acc):   {abs(train_acc[-1] - val_acc[-1]):.2f}%
  Final Gap (Loss):  {abs(train_loss[-1] - val_loss[-1]):.4f}
  Status:            {"✓ NO OVERFITTING" if abs(train_acc[-1] - val_acc[-1]) < 5 else "⚠ MILD OVERFITTING" if abs(train_acc[-1] - val_acc[-1]) < 10 else "✗ SIGNIFICANT OVERFITTING"}

RECOMMENDATIONS:
""")

# Recommendations based on analysis
gap = abs(train_acc[-1] - val_acc[-1])
if gap > 15:
    print("  ⚠ High overfitting detected - Consider:")
    print("    • Use more data augmentation")
    print("    • Increase regularization (dropout, L2)")
    print("    • Reduce model complexity")
elif gap > 5:
    print("  ⚠ Mild overfitting - Consider:")
    print("    • Increase data augmentation slightly")
    print("    • Fine-tune learning rate")
else:
    print("  ✓ Good generalization - Model is well-balanced")

if val_acc[-1] < 80:
    print("  ⚠ Validation accuracy < 80% - Consider:")
    print("    • Train for more epochs")
    print("    • Check data labeling")
    print("    • Verify data quality")
elif val_acc[-1] < 90:
    print("  • Good performance - Ready for production")
else:
    print("  ✓ Excellent performance - Ready for deployment")

print("\n" + "="*70)
print("GENERATED IMAGES")
print("="*70)
print("""
✓ outputs/images/training_accuracy_curves.png
  └─ Accuracy curves only (professional format)

✓ outputs/images/training_loss_curves.png
  └─ Loss curves only (professional format)

✓ outputs/images/training_history_combined.png
  └─ Side-by-side accuracy & loss (2 subplots)

✓ outputs/images/training_history_analysis.png
  └─ Comprehensive analysis (6 subplots + statistics)
""")

print("="*70 + "\n")
