#!/usr/bin/env python3
"""
Model Evaluation Script - Generate Confusion Matrix & Accuracy Metrics
Evaluates on ALL data splits: train, validation, AND test

Addresses reviewer concern:
  "Evaluation done on only 25 images instead of 725 test images"
  → Now evaluates every available split and reports per-split metrics.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

print("\n" + "="*70)
print("Model Evaluation - Full Multi-Split Evaluation")
print("="*70)

# ============================================================================
# 1. SETUP
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Using device: {device}")

os.makedirs('outputs/metrics', exist_ok=True)
os.makedirs('outputs/images', exist_ok=True)

# ============================================================================
# 2. LOAD MODEL
# ============================================================================

print("\n✓ Loading trained model...")

model_path = 'models/malnutrition_model.pth'
if not os.path.exists(model_path):
    print(f"✗ Model not found at {model_path}")
    print("Please train the model first: python training/train_with_labels.py")
    exit(1)

# ResNet-50 with custom classifier head (Dropout + Linear)
# Matches the architecture used in train_with_labels.py
model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, 2)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

print(f"✓ Model loaded from: {model_path}")
print(f"  Architecture: ResNet-50 → Dropout(0.3) → Linear(2048, 2)")
print(f"  Pretrained backbone: IMAGENET1K_V1 (fine-tuned)")

# ============================================================================
# 3. LOAD LABELS
# ============================================================================

print("\n✓ Loading labels...")

labels_df = pd.read_csv('data/labels.csv')
labels_dict = dict(zip(labels_df['filename'], labels_df['label']))
print(f"✓ Loaded {len(labels_dict)} labels")

# ============================================================================
# 4. EVALUATION DATASET
# ============================================================================

class EvaluationDataset(Dataset):
    def __init__(self, img_dir, split_type, labels_dict, transform=None):
        self.img_dir = img_dir
        self.split_type = split_type
        self.labels_dict = labels_dict
        self.transform = transform
        self.images = []

        if not os.path.exists(img_dir):
            print(f"  ⚠ Directory not found: {img_dir}")
            return

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
        except Exception:
            return None, None

        if self.transform:
            image = self.transform(image)

        key = f"{self.split_type}/{img_file}"
        label = self.labels_dict.get(key, 0)

        return image, label

# ============================================================================
# 5. TRANSFORMS
# ============================================================================

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("✓ Data transforms created")

# ============================================================================
# 6. HELPER: evaluate one split
# ============================================================================

def evaluate_split(split_name, dataset, batch_size=1):
    """Evaluate model on a single data split and return detailed metrics."""

    if len(dataset) == 0:
        print(f"\n  ⚠ {split_name} split is empty — skipping")
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_labels = []
    all_confidences = []
    all_probs = []

    with torch.no_grad():
        for idx, (images, batch_labels) in enumerate(loader):
            if images is None:
                continue
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            confidences = torch.max(probs, dim=1).values.cpu().numpy() * 100
            probs_np = probs.cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(batch_labels.numpy())
            all_confidences.extend(confidences)
            all_probs.extend(probs_np)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # --- Metrics ---
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    accuracy = accuracy_score(all_labels, all_predictions)

    # Handle edge cases where a class may be missing
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc_score = roc_auc_score(all_labels, all_probs[:, 1])
    except Exception:
        auc_score = 0.0

    class_report = classification_report(
        all_labels, all_predictions,
        target_names=['Healthy', 'Malnourished'],
        output_dict=True,
        zero_division=0
    )

    result = {
        'split': split_name,
        'total_samples': len(all_labels),
        'healthy_count': int((all_labels == 0).sum()),
        'malnourished_count': int((all_labels == 1).sum()),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
        },
        'accuracy_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'auc_roc': float(auc_score),
        },
        'classification_report': class_report,
        'raw': {
            'predictions': all_predictions,
            'labels': all_labels,
            'probs': all_probs,
            'cm': cm,
        }
    }

    return result


def print_split_metrics(result):
    """Pretty-print metrics for a single split."""
    m = result['accuracy_metrics']
    cm_vals = result['confusion_matrix']
    print(f"\n  Samples: {result['total_samples']} "
          f"(Healthy: {result['healthy_count']}, Malnourished: {result['malnourished_count']})")
    print(f"  TN={cm_vals['true_negatives']}  FP={cm_vals['false_positives']}  "
          f"FN={cm_vals['false_negatives']}  TP={cm_vals['true_positives']}")
    print(f"  Accuracy:    {m['accuracy']*100:6.2f}%")
    print(f"  Precision:   {m['precision']*100:6.2f}%")
    print(f"  Recall:      {m['recall']*100:6.2f}%")
    print(f"  Specificity: {m['specificity']*100:6.2f}%")
    print(f"  F1-Score:    {m['f1_score']*100:6.2f}%")
    print(f"  AUC-ROC:     {m['auc_roc']*100:6.2f}%")


# ============================================================================
# 7. EVALUATE ALL SPLITS
# ============================================================================

print("\n" + "="*70)
print("Evaluating on ALL Data Splits")
print("="*70)

splits_to_evaluate = [
    ('Validation', 'data/val/images', 'val'),
    ('Test', 'data/test/images', 'test'),
    ('Train', 'data/train/images', 'train'),
]

all_results = {}

for split_label, img_dir, split_key in splits_to_evaluate:
    print(f"\n{'─'*70}")
    print(f"  Split: {split_label.upper()} ({img_dir})")
    print(f"{'─'*70}")

    dataset = EvaluationDataset(img_dir, split_key, labels_dict, val_transform)
    result = evaluate_split(split_label, dataset)
    if result is not None:
        all_results[split_key] = result
        print_split_metrics(result)

# ============================================================================
# 8. PRIMARY EVALUATION — VALIDATION SET (for backward compat)
# ============================================================================
# The validation split is the primary evaluation split.
# If a test split with images exists, report it prominently too.

primary_key = 'val' if 'val' in all_results else ('test' if 'test' in all_results else None)

if primary_key is None:
    print("\n✗ No evaluation data available!")
    exit(1)

primary = all_results[primary_key]
m = primary['accuracy_metrics']
cm = primary['raw']['cm']
all_labels = primary['raw']['labels']
all_predictions = primary['raw']['predictions']
all_probs = primary['raw']['probs']

# ============================================================================
# 9. GENERATE CONFUSION MATRIX VISUALIZATION (primary split)
# ============================================================================

print("\n" + "-"*70)
print("Generating Visualizations")
print("-"*70)

# --- Per-split confusion matrices ---
for split_key, result in all_results.items():
    if result is None:
        continue
    split_cm = result['raw']['cm']
    plt.figure(figsize=(10, 8))
    sns.heatmap(split_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Malnourished'],
                yticklabels=['Healthy', 'Malnourished'],
                cbar_kws={'label': 'Count'},
                square=True, linewidths=2)
    plt.title(f'Confusion Matrix — {split_key.upper()} Split '
              f'(n={result["total_samples"]})', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    fname = f'outputs/images/confusion_matrix_{split_key}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fname}")
    plt.close()

# Also save the primary one as the default name
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Malnourished'],
            yticklabels=['Healthy', 'Malnourished'],
            cbar_kws={'label': 'Count'},
            square=True, linewidths=2)
plt.title('Confusion Matrix - Malnutrition Detection', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/images/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/images/confusion_matrix.png")
plt.close()

# ============================================================================
# 10. GENERATE ACCURACY METRICS TABLE
# ============================================================================

cm_vals = primary['confusion_matrix']
metrics_data = {
    'Metric': [
        'Accuracy',
        'Precision',
        'Recall (Sensitivity)',
        'Specificity',
        'F1-Score',
        'AUC-ROC'
    ],
    'Score': [
        f'{m["accuracy"]:.4f}',
        f'{m["precision"]:.4f}',
        f'{m["recall"]:.4f}',
        f'{m["specificity"]:.4f}',
        f'{m["f1_score"]:.4f}',
        f'{m["auc_roc"]:.4f}'
    ],
    'Percentage': [
        f'{m["accuracy"]*100:.2f}%',
        f'{m["precision"]*100:.2f}%',
        f'{m["recall"]*100:.2f}%',
        f'{m["specificity"]*100:.2f}%',
        f'{m["f1_score"]*100:.2f}%',
        f'{m["auc_roc"]*100:.2f}%'
    ]
}

metrics_df = pd.DataFrame(metrics_data)
print("\n" + "="*70)
print(f"ACCURACY METRICS TABLE (Primary: {primary_key.upper()} split)")
print("="*70)
print(metrics_df.to_string(index=False))

metrics_df.to_csv('outputs/metrics/accuracy_metrics.csv', index=False)
print("\n✓ Saved: outputs/metrics/accuracy_metrics.csv")

# ============================================================================
# 11. CONFUSION MATRIX TABLE
# ============================================================================

cm_df = pd.DataFrame(
    cm,
    index=['Actual Healthy', 'Actual Malnourished'],
    columns=['Predicted Healthy', 'Predicted Malnourished']
)

print("\n" + "="*70)
print("CONFUSION MATRIX TABLE")
print("="*70)
print(cm_df)
print()

cm_df_with_totals = cm_df.copy()
cm_df_with_totals['Total'] = cm_df_with_totals.sum(axis=1)
cm_df_with_totals.loc['Total'] = cm_df_with_totals.sum()

print("WITH TOTALS:")
print(cm_df_with_totals)

cm_df.to_csv('outputs/metrics/confusion_matrix.csv')
print("\n✓ Saved: outputs/metrics/confusion_matrix.csv")

# ============================================================================
# 12. DETAILED CLASSIFICATION REPORT
# ============================================================================

print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)

report_df = pd.DataFrame(primary['classification_report']).transpose()
print(report_df)

report_df.to_csv('outputs/metrics/classification_report.csv')
print("\n✓ Saved: outputs/metrics/classification_report.csv")

# ============================================================================
# 13. ROC CURVE
# ============================================================================

plt.figure(figsize=(10, 8))
fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC-AUC Curve - Malnutrition Detection', fontsize=16, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/images/roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/images/roc_curve.png")
plt.close()

# ============================================================================
# 14. SAVE ALL METRICS AS JSON — comprehensive multi-split report
# ============================================================================

all_metrics_json = {}
for split_key, result in all_results.items():
    all_metrics_json[split_key] = {
        'total_samples': result['total_samples'],
        'healthy_count': result['healthy_count'],
        'malnourished_count': result['malnourished_count'],
        'confusion_matrix': result['confusion_matrix'],
        'accuracy_metrics': result['accuracy_metrics'],
    }

# Keep backward-compatible top-level keys from the primary split
all_metrics_json['primary_split'] = primary_key
all_metrics_json['confusion_matrix'] = primary['confusion_matrix']
all_metrics_json['accuracy_metrics'] = primary['accuracy_metrics']
all_metrics_json['total_samples'] = primary['total_samples']
all_metrics_json['healthy_count'] = primary['healthy_count']
all_metrics_json['malnourished_count'] = primary['malnourished_count']

with open('outputs/metrics/all_metrics.json', 'w') as f:
    json.dump(all_metrics_json, f, indent=4)

print("\n✓ Saved: outputs/metrics/all_metrics.json")

# ============================================================================
# 15. SUMMARY
# ============================================================================

print("\n" + "="*70)
print("EVALUATION COMPLETE — ALL SPLITS")
print("="*70)

for split_key, result in all_results.items():
    rm = result['accuracy_metrics']
    print(f"\n  {split_key.upper()} ({result['total_samples']} images):")
    print(f"    Accuracy={rm['accuracy']*100:.1f}%  F1={rm['f1_score']*100:.1f}%  AUC={rm['auc_roc']*100:.1f}%")

print(f"""
📊 PRIMARY EVALUATION ({primary_key.upper()} split, n={primary['total_samples']}):
├─ Accuracy:  {m['accuracy']*100:.2f}%
├─ Precision: {m['precision']*100:.2f}%
├─ Recall:    {m['recall']*100:.2f}%
├─ F1-Score:  {m['f1_score']*100:.2f}%
└─ AUC-ROC:   {m['auc_roc']*100:.2f}%

📁 OUTPUT FILES:
├─ Confusion Matrix: outputs/images/confusion_matrix.png
├─ Per-Split CMs:    outputs/images/confusion_matrix_{{split}}.png
├─ ROC Curve:        outputs/images/roc_curve.png
├─ Metrics CSV:      outputs/metrics/accuracy_metrics.csv
├─ CM CSV:           outputs/metrics/confusion_matrix.csv
├─ Report CSV:       outputs/metrics/classification_report.csv
└─ All Metrics JSON: outputs/metrics/all_metrics.json

✓ All files saved successfully!
""")

print("="*70 + "\n")
