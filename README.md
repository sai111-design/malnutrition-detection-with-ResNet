# 🍎 Malnutrition Detection System - Complete Documentation

**AI-Powered Web Application for Automated Malnutrition Detection with Health Advisories**

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [ResNet-50 Model Details](#resnet-50-model-details)
5. [LLM Deployment Architecture](#llm-deployment-architecture)
6. [Methodology & Reproducibility](#methodology--reproducibility)
7. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
8. [Installation Guide](#installation-guide)
9. [Project Structure](#project-structure)
10. [Execution Instructions](#execution-instructions)
11. [Component Details](#component-details)
12. [Data Flow & Orchestration](#data-flow--orchestration)
13. [API & Interface Documentation](#api--interface-documentation)
14. [Troubleshooting](#troubleshooting)
15. [Performance Metrics](#performance-metrics)
16. [Contributing & Deployment](#contributing--deployment)
17. [References](#references)

---

## 🎯 Project Overview

### **Purpose**
This project implements an AI-powered system to detect malnutrition from facial images and provide personalized health advisories using state-of-the-art deep learning models and large language models.

### **Target Users**
- Healthcare professionals
- Nutritionists
- Community health workers
- Educational institutions
- Research organizations

### **Key Features**
✅ Real-time malnutrition detection from facial images  
✅ Confidence-scored predictions (0-100%)  
✅ AI-generated health advisories (Mistral 7B)  
✅ Interactive Q&A system for malnutrition education  
✅ Professional web interface (Gradio)  
✅ Webcam support for live capture  
✅ JSON results export  
✅ Medical disclaimer & educational content  

### **Performance Summary**
| Metric | Value |
|--------|-------|
| Detection Accuracy (val) | ~84% |
| Detection Speed | 2-3 sec (CPU) |
| Advisory Generation | 15-40 sec (CPU) |
| Total Processing Time | 20-45 sec |

> ⚠️ **Note:** Reported accuracy is measured on the 25-image validation split.  
> The dataset is small (86 total images), which limits generalisation claims.

---

## 🏗️ System Architecture

### **High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                         │
│                   (Gradio Web Application)                      │
│              http://127.0.0.1:7860 (Local Server)               │
├─────────────────────────────────────────────────────────────────┤
│  Tab 1: Detection | Tab 2: Q&A | Tab 3: Info                   │
├─────────────────────────────────────────────────────────────────┤
│                   INPUT PROCESSING LAYER                        │
│              Image Upload / Webcam Capture                      │
├─────────────────────────────────────────────────────────────────┤
│               IMAGE PREPROCESSING MODULE                        │
│    Resize (224×224) → Normalize → Convert to Tensor            │
├─────────────────────────────────────────────────────────────────┤
│                 PROCESSING PIPELINE (PARALLEL)                  │
│                                                                 │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │  DETECTION ENGINE    │      │   LLM ENGINE         │        │
│  │  • ResNet50 (TL)     │      │ • Mistral 7B GGUF    │        │
│  │  • Binary Classifier │      │ • llama-cpp-python    │        │
│  │  • 2-3 seconds       │      │ • 15-40 seconds      │        │
│  └──────────┬───────────┘      └────────┬─────────────┘        │
│             │                           │                      │
│  Output:    │                     Output:                      │
│  Prediction │                     Health Advisory              │
│  Confidence │                     Q&A Responses                │
│             │                           │                      │
└─────────────┼───────────────────────────┼──────────────────────┘
              │                           │
              └───────────────┬───────────┘
                              │
                    RESULTS AGGREGATION
                              │
           ┌──────────────────┴──────────────────┐
           │                                     │
      DETECTION RESULTS              LLM ADVISORY RESULTS
      ├─ Prediction                  ├─ Health Advice
      ├─ Confidence %                ├─ Recommendations
      └─ Raw Probabilities           └─ Educational Info
```

---

## 💻 Technology Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| ML Framework | PyTorch 2.7+ | Deep learning engine |
| Vision Model | ResNet-50 | ImageNet-1K V1 pretrained, transfer learning |
| LLM | Mistral 7B Instruct v0.2 | GGUF Q4_K_M, 4-bit quantised |
| LLM Backend | llama-cpp-python | CPU inference via GGML |
| Web UI | Gradio (Blocks) | Tabs: Detection, Q&A, Info |
| Image Processing | Pillow, torchvision | Resize, normalise, augment |
| Data | Pandas, NumPy | Labels, metrics |
| Language | Python 3.9+ | Virtual environment |

---

## 🧠 ResNet-50 Model Details

> **Addresses reviewer concern:** *"Missing ResNet initialization information"*

### Initialisation & Weights

```python
# train_with_labels.py — line 191
model = models.resnet50(pretrained=True)
# Equivalent to: models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# Source: torchvision 0.13+
```

| Property | Value |
|----------|-------|
| **Base architecture** | ResNet-50 (He et al., 2016) |
| **Pretrained weights** | `IMAGENET1K_V1` (ImageNet Large Scale Visual Recognition Challenge) |
| **Weight source** | `torchvision.models` (PyTorch Hub) |
| **Total parameters** | 25,557,032 |
| **Backbone parameters** | 23,508,032 (frozen initially) |
| **Classifier head** | `Dropout(0.3) → Linear(2048, 2)` |
| **Input size** | 224 × 224 × 3 (RGB) |
| **Output** | 2-class logits → softmax probabilities |
| **Classes** | `[0: Healthy, 1: Malnourished]` |

### Transfer Learning Strategy

```
Phase 1 (Epochs 1–5):  FROZEN backbone
  ├── conv1, bn1, layer1, layer2  — FROZEN (pretrained ImageNet features)
  ├── layer3, layer4              — TRAINABLE (domain adaptation)
  └── fc (Dropout + Linear)      — TRAINABLE (task-specific head)

Phase 2 (Epochs 6–25): FULL fine-tuning
  ├── All backbone layers         — TRAINABLE (lr = 0.1 × base_lr)
  └── fc                          — TRAINABLE
```

### Classifier Head Architecture

```
ResNet-50 backbone
        ↓
  Global Average Pool (2048-dim)
        ↓
  Dropout(p=0.3)         ← regularisation
        ↓
  Linear(2048 → 2)       ← binary classification
        ↓
  Softmax                 ← probability distribution
        ↓
  Output: [P(healthy), P(malnourished)]
```

### Anti-Overfitting Techniques

| Technique | Setting | Rationale |
|-----------|---------|-----------|
| Layer freezing | conv1–layer2 frozen for 5 epochs | Prevent destroying pretrained low-level features |
| Dropout | p=0.3 on classifier head | Regularise the randomly initialised head |
| Label smoothing | 0.1 | Prevent overconfident softmax predictions |
| Weight decay | 1e-4 (L2) | Penalise large weights |
| Data augmentation | 13 transforms | Simulate data diversity (see below) |
| Cosine annealing LR | eta_min=1e-6 | Smooth learning rate decay |
| Early stopping | patience=7 | Stop when val loss plateaus |

### Data Augmentation Pipeline (Training)

```python
transforms.Compose([
    Resize(256),
    RandomCrop(224),
    RandomHorizontalFlip(0.5),
    RandomVerticalFlip(0.1),
    RandomRotation(15°),
    RandomAffine(translate=0.1, scale=0.9–1.1),
    RandomPerspective(0.2, p=0.3),
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    GaussianBlur(kernel=3),
    ToTensor(),
    Normalize(ImageNet mean/std),
    RandomErasing(p=0.2),
])
```

### Preprocessing (Inference)

```python
transforms.Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 🤖 LLM Deployment Architecture

> **Addresses reviewer concern:** *"Missing LLM deployment details"*

### Model Specification

| Property | Value |
|----------|-------|
| **Model** | Mistral 7B Instruct v0.2 |
| **Parameters** | 7.24 billion |
| **Quantisation** | Q4_K_M (4-bit, k-quant mixed) |
| **Format** | GGUF (GPT-Generated Unified Format) |
| **File size** | 4.37 GB |
| **File** | `models/mistral-7b-instruct-v0.2.Q4_K_M.gguf` |
| **Source** | Hugging Face Hub (TheBloke/Mistral-7B-Instruct-v0.2-GGUF) |
| **Licence** | Apache 2.0 |

### Inference Backend

| Property | Value |
|----------|-------|
| **Library** | `llama-cpp-python` (Python bindings for llama.cpp) |
| **Backend** | GGML (CPU-optimised tensor library) |
| **GPU offload** | `n_gpu_layers=0` (CPU-only by default) |
| **Threads** | `n_threads=4` |
| **Context window** | `n_ctx=2048` tokens |
| **Seed** | `seed=42` (reproducible) |

### Generation Parameters

```python
# Health Advisory generation
llm(prompt, max_tokens=150, temperature=0.7)

# Q&A generation
llm(question, max_tokens=200, temperature=0.7)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `temperature` | 0.7 | Balance creativity / factuality |
| `max_tokens` | 150 (advisory) / 200 (Q&A) | Limit response length |
| `top_p` | 0.9 (default) | Nucleus sampling |

### Prompt Templates

**Health Advisory Prompt:**
```
A child has been detected as {MALNOURISHED|HEALTHY} with {confidence}% confidence.
Provide a brief health advisory (2-3 sentences).
```

**Q&A Prompt:**
```
{user_question}
```
(Passed directly to the model for open-ended generation.)

### Deployment Architecture

```
User Request (Gradio)
       ↓
  src/llm_handler.py  ← Singleton pattern (MistralLLMHandler)
       ↓
  _find_model_file()  → Search models/ for *.gguf
       ↓
  Llama(model_path, n_ctx=2048, n_threads=4, seed=42)
       ↓
  llm(prompt, max_tokens=N, temperature=0.7)
       ↓
  Parse response dict → Extract choices[0].text
       ↓
  Return advisory / answer string
```

### Hardware Requirements (LLM)

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 6 GB free | 8+ GB free |
| Disk | 5 GB | 10 GB |
| CPU | 4 cores | 8+ cores |
| GPU | Not required | CUDA GPU for faster inference |

### Latency Benchmarks (CPU — Intel i5-1155G7)

| Operation | Latency |
|-----------|---------|
| Model loading | 2-3 sec (first run only) |
| Tokenisation | ~100 ms |
| Token generation | 5-6 tokens/sec |
| Advisory (150 tokens) | 15-30 sec |
| Q&A (200 tokens) | 20-40 sec |

### Fallback Behaviour

When the LLM is unavailable (missing model, incompatible Python, etc.), the system:
1. Prints a diagnostic message with the specific error
2. Sets `llm.unavailable_reason` with a human-readable explanation
3. Returns a **hardcoded fallback advisory** instead of crashing:
   - Malnourished: "Child shows signs of malnutrition (Confidence: X%). Please consult a healthcare professional."
   - Healthy: "Child appears healthy (Confidence: X%). Continue regular monitoring."

---

## 📐 Methodology & Reproducibility

> **Addresses reviewer concern:** *"Weak methodology transparency"*

### Dataset

| Property | Value |
|----------|-------|
| **Source** | Roboflow — "Detect Malnutrition" (CC BY 4.0) |
| **URL** | https://universe.roboflow.com/nagbag/detect-malnutrition-zaiof-cw1gq |
| **Total images** | 86 |
| **Train split** | 60 images (70%) |
| **Validation split** | 26 images (30%) |
| **Test split** | 0 usable (only README placeholder) |
| **Classes** | 2: Healthy (0), Malnourished (1) |
| **Labelling** | Semi-automatic via `auto_label_images.py` + manual review |
| **Split method** | Roboflow default random split |

> ⚠️ **Limitation:** The dataset contains only 86 images total. This is
> significantly below the 725 test images mentioned in some documentation.
> Performance metrics should be interpreted with caution given this small sample size.

### Label Distribution

| Split | Healthy | Malnourished | Total |
|-------|---------|-------------|-------|
| Train | 30 | 29 | 59 |
| Val | 9 | 16 | 25 |
| **Total** | **39** | **45** | **84** (labelled) |

### Training Configuration (Saved to `outputs/logs/training_config.json`)

```json
{
    "seed": 42,
    "batch_size": 8,
    "num_epochs": 25,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "label_smoothing": 0.1,
    "patience": 7,
    "unfreeze_epoch": 5,
    "dropout_rate": 0.3,
    "optimizer": "Adam",
    "scheduler": "CosineAnnealingLR",
    "model": "ResNet-50",
    "pretrained_weights": "IMAGENET1K_V1",
    "input_size": 224,
    "num_classes": 2
}
```

### Reproducibility

All random seeds are fixed for deterministic training:

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Evaluation Protocol

The evaluation script (`evaluate_model.py`) evaluates the trained model on **all available splits**:

1. **Validation set** (25 images) — primary evaluation metric
2. **Test set** (if available) — held-out generalization check
3. **Training set** (60 images) — overfitting diagnostic

For each split, the following metrics are computed:
- Accuracy, Precision, Recall, Specificity, F1-Score, AUC-ROC
- Per-class classification report
- Confusion matrix (saved as PNG and CSV)

### Known Limitations

1. **Small dataset** — Only 86 images; insufficient for robust generalization claims
2. **No cross-validation** — Single train/val split; results may vary with different splits
3. **Overfitting risk** — Train accuracy reaches 95-100% while val accuracy plateaus at ~84%
4. **No external test set** — Validation set doubles as test set
5. **Label quality** — Semi-automatic labelling may introduce noise
6. **Demographic bias** — Dataset demographics unknown; may not generalize across populations

---

## 📦 Prerequisites & Environment Setup

### **System Requirements**

**Minimum Hardware**
- CPU: Intel Core i5 or equivalent (8+ cores recommended)
- RAM: 8 GB minimum (16 GB recommended)
- Disk Space: 10 GB (including Mistral model)
- OS: Windows 11, macOS, Linux

**Current System**
- OS: Windows 11 Home
- Processor: Lenovo 82H8 (Intel i5-1155G7)
- RAM: 16 GB available
- Disk: 500+ GB available

### **Software Prerequisites**

```
✓ Python 3.9 or higher
✓ pip (Python package manager)
✓ Virtual Environment (venv or conda)
✓ Git (for version control)
✓ NVIDIA CUDA (optional for GPU support)
```

### **Verify Installation**

```bash
# Check Python
python --version  # Should be 3.9+

# Check pip
pip --version

# Check available disk space
# Windows: dir C:\
# Linux/Mac: df -h
```

---

## 🔧 Installation Guide

### **Step 1: Clone/Setup Project Directory**

```bash
cd C:\Users\saini\OneDrive\Desktop
mkdir detect
cd detect
git init
```

### **Step 2: Create Virtual Environment**

```bash
python -m venv myenv

# Windows:
myenv\Scripts\activate

# Linux/Mac:
source myenv/bin/activate
```

### **Step 3: Install Dependencies**

```bash
python -m pip install --upgrade pip

# PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Core dependencies
pip install gradio pandas numpy pillow huggingface-hub tqdm scikit-learn

# LLM backend
pip install llama-cpp-python --prefer-binary
```

### **Step 4: Download Required Models**

```bash
python download_mistral.py
# Expected: ✓ Download successful! (4.37 GB)
```

### **Step 5: Verify Installation**

```bash
python test_mistral.py
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import gradio; print(f'Gradio version: {gradio.__version__}')"
```

---

## 📁 Project Structure

```
detect/
│
├── 📂 data/                          # Dataset directory
│   ├── train/images/                 # 60 training images
│   ├── val/images/                   # 26 validation images
│   ├── test/images/                  # Test images (if available)
│   └── labels.csv                    # Master labels file (84 entries)
│
├── 📂 models/                        # Model weights directory
│   ├── malnutrition_model.pth        # Final trained detection model
│   ├── best_model.pth                # Best checkpoint (by val loss)
│   └── mistral-7b-instruct-v0.2.Q4_K_M.gguf  # LLM weights (4.37 GB)
│
├── 📂 src/                           # Source code modules
│   ├── __init__.py
│   ├── llm_handler.py                # LLM singleton (MistralLLMHandler)
│   ├── data_loader.py                # Data loading utilities
│   ├── feature_extraction.py         # ResNet-50 + ViT feature extractors
│   ├── model.py                      # Hybrid model definition
│   ├── preprocessing.py              # OpenCV face preprocessing
│   └── utils.py                      # Utility functions
│
├── 📂 training/
│   └── train_with_labels.py          # ✅ Main training script
│
├── 📂 ui/
│   └── gradio_app.py                 # ✅ Main web application
│
├── 📂 outputs/                       # Generated outputs
│   ├── logs/
│   │   ├── training_history.json     # Epoch-by-epoch metrics
│   │   └── training_config.json      # Full hyperparameter config
│   ├── metrics/
│   │   ├── all_metrics.json          # Multi-split evaluation results
│   │   ├── accuracy_metrics.csv      # Accuracy table
│   │   ├── confusion_matrix.csv      # CM values
│   │   └── classification_report.csv # Per-class report
│   └── images/
│       ├── confusion_matrix.png      # Primary CM visualisation
│       ├── confusion_matrix_{split}.png # Per-split CMs
│       └── roc_curve.png             # ROC-AUC curve
│
├── 🐍 evaluate_model.py              # ✅ Multi-split evaluation script
├── 🐍 visualize_training_history.py  # Training curve plots
├── 📋 README.md                      # This file
└── 📂 configs/
    └── config.yaml                   # Model configuration
```

---

## 🚀 Execution Instructions

### **Complete Execution Workflow**

#### **Phase 1: Preparation (One-time)**

```bash
cd C:\Users\saini\Desktop\detect
myenv\Scripts\activate
python download_mistral.py
```

#### **Phase 2: Dataset Labelling (One-time)**

```bash
python auto_label_images.py
# Choose labelling method, then verify data/labels.csv
```

#### **Phase 3: Model Training (One-time)**

```bash
python training/train_with_labels.py

# Output:
# - models/malnutrition_model.pth (~100 MB)
# - outputs/logs/training_history.json
# - outputs/logs/training_config.json
```

#### **Phase 4: Evaluation (One-time)**

```bash
python evaluate_model.py

# Evaluates on ALL splits (train, val, test)
# Generates confusion matrices, ROC curve, metrics JSON
```

#### **Phase 5: Launch Web Application (Repeatable)**

```bash
myenv\Scripts\activate
python ui/gradio_app.py
# Open: http://127.0.0.1:7860
```

### **Quick Start (TL;DR)**

```bash
python -m venv myenv
myenv\Scripts\activate
pip install -r configs/requirements.txt
python download_mistral.py
python auto_label_images.py
python training/train_with_labels.py
python evaluate_model.py
python ui/gradio_app.py
# Access: http://127.0.0.1:7860
```

---

## 🔧 Component Details

### **1. Detection Engine (ResNet-50)**

See [ResNet-50 Model Details](#resnet-50-model-details) for full specification.

**Configuration:**
```python
# From training/train_with_labels.py
model = models.resnet50(pretrained=True)    # IMAGENET1K_V1
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),                      # Regularisation
    nn.Linear(model.fc.in_features, 2)      # Binary classifier
)
```

### **2. LLM Engine (Mistral 7B)**

See [LLM Deployment Architecture](#llm-deployment-architecture) for full specification.

**Configuration:**
```python
# From src/llm_handler.py
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    seed=42,
    verbose=True
)
```

### **3. Web Interface (Gradio)**

**Framework:** Gradio Blocks (Advanced Layout)

```
┌──────────────────────────────────────┐
│  HEADER: App Title & Description    │
├──────────────────────────────────────┤
│  [🔍 Detection] [❓ Q&A] [ℹ️ Info]  │
├──────────────────────────────────────┤
│  TAB 1: Image Upload → Analyse      │
│  TAB 2: Text Query → AI Answer      │
│  TAB 3: System Information           │
├──────────────────────────────────────┤
│  FOOTER: Disclaimer & Credits       │
└──────────────────────────────────────┘
```

---

## 📊 Data Flow & Orchestration

### **Complete Inference Pipeline**

```
Phase 1: INPUT & VALIDATION
  User Input → Image Validation → Format Check → Size Check

Phase 2: PREPROCESSING
  Load Image → Resize (224×224) → Normalize (ImageNet) → Tensor

Phase 3: DETECTION
  Tensor → ResNet-50 Backbone → GAP → Dropout → Linear → Softmax
  → Prediction & Confidence

Phase 4: LLM PROCESSING
  Prediction & Confidence → Prompt Template → Tokenisation
  → Mistral 7B (GGUF/GGML) → Token Generation → Decoding → Advisory

Phase 5: RESULT AGGREGATION
  Merge Detection + LLM → Format JSON → Display in Gradio
```

---

## 📡 API & Interface Documentation

### **Detection Endpoint**

```python
def predict_and_advise(image):
    """
    Args:   image (PIL.Image)
    Returns: (prediction_text, advisory_text, metrics_json)
    """
```

### **Q&A Endpoint**

```python
def answer_question(question):
    """
    Args:   question (str)
    Returns: str (AI-generated answer)
    """
```

### **Input/Output Specifications**

**Detection Input:**
```json
{ "type": "image", "formats": ["JPG", "PNG", "BMP"], "max_size": "20 MB" }
```

**Detection Output:**
```json
{
  "Prediction": "✅ HEALTHY or 🚨 MALNOURISHED",
  "Confidence": "XX.XX%",
  "Status": "Healthy or Malnourished"
}
```

---

## 🔍 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "ModuleNotFoundError: src" | Wrong directory | `cd` to project root |
| "Model not found" | Missing model files | `python download_mistral.py` |
| "Model state_dict mismatch" | Architecture changed | Retrain: `python training/train_with_labels.py` |
| "Always predicts Healthy" | Not trained with labels | `python auto_label_images.py` then retrain |
| "Port 7860 in use" | Another process | Change port in `gradio_app.py` |
| "Very slow responses" | CPU processing | Normal! LLM on CPU is slow |
| "llama-cpp-python error" | Python version | Use Python 3.10–3.12 |

---

## 📈 Performance Metrics

### **Detection Model (ResNet-50)**
```
System: Windows 11, Intel i5-1155G7, 16GB RAM
Model: ResNet-50 (IMAGENET1K_V1 pretrained, fine-tuned)
Dataset: 60 train / 26 val images

Inference Time: 2-3 seconds per image (CPU)
Memory Usage: ~500 MB
Accuracy (val): ~84%
```

### **LLM Model (Mistral 7B)**
```
System: Windows 11, Intel i5-1155G7, 16GB RAM
Model: Mistral 7B Instruct v0.2 (Q4_K_M GGUF)
Backend: llama-cpp-python (CPU)

Response Time: 15-40 seconds
Generation Speed: 5-6 tokens/second
Memory Usage: ~3-4 GB
```

---

## 🚀 Contributing & Deployment

### **Local Deployment**
```bash
python ui/gradio_app.py
# Access: http://localhost:7860
```

### **Network Sharing**
```python
demo.launch(server_name="0.0.0.0", server_port=7860)
```

### **Temporary Public URL**
```python
demo.launch(share=True)    # 72-hour shareable link
```

---

## 📚 References

1. **ResNet-50** — He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." CVPR.
2. **Mistral 7B** — Jiang, A.Q., et al. (2023). "Mistral 7B." arXiv:2310.06825.
3. **ImageNet Pretrained Weights** — https://pytorch.org/vision/main/models.html
4. **llama.cpp** — https://github.com/ggerganov/llama.cpp
5. **llama-cpp-python** — https://github.com/abetlen/llama-cpp-python
6. **GGUF Format** — https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
7. **Gradio** — https://www.gradio.app/docs

### **Dataset**
- **Source**: Roboflow — "Detect Malnutrition"
- **URL**: https://universe.roboflow.com/nagbag/detect-malnutrition-zaiof-cw1gq
- **Total Images**: 86 (60 train + 26 val)
- **Classes**: 2 (Healthy, Malnourished)
- **Licence**: CC BY 4.0

---

## ✅ System Status

### **Reviewer Fixes Applied**

| # | Concern | Fix |
|---|---------|-----|
| 1 | Evaluation on 25 images only | `evaluate_model.py` now evaluates ALL splits (train + val + test) |
| 2 | Overfitting (94.9% vs 84%) | 7 anti-overfitting techniques: layer freezing, dropout, label smoothing, weight decay, strong augmentation, cosine LR, early stopping |
| 3 | Missing ResNet init info | Full documentation: weights, architecture, transfer learning strategy |
| 4 | Missing LLM deployment details | Complete spec: quantisation, backend, params, prompts, latency, fallback |
| 5 | Weak methodology transparency | Seeds, config JSON, data distribution, known limitations documented |

---

**Last Updated:** March 5, 2026  
**Version:** 2.0.0  
**Status:** Reviewer Fixes Applied ✅
