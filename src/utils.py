import torch
import yaml
import os

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config if config else {}

def get_device():
    """Get device (GPU or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    return device

def format_prediction(label_idx, probability):
    """Format prediction for display"""
    labels = ["Healthy", "Malnourished"]
    label = labels[label_idx]
    confidence = probability[0, label_idx].item() * 100
    return label, confidence
