import torch
import torch.nn as nn
from .feature_extraction import HybridFeatureExtractor

class MalnutritionDetector(nn.Module):
    """Hybrid ResNet50 + ViT model for malnutrition detection"""

    def __init__(self, pretrained=True):
        super().__init__()
        self.feature_extractor = HybridFeatureExtractor(pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(2816, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

    def predict(self, x):
        """Returns prediction with probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1)
        return pred_label, probs
