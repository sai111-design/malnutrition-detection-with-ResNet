import torch
import torch.nn as nn
import torchvision.models as models
import timm

class ResNet50Extractor(nn.Module):
    """ResNet-50 feature extractor (2048-dim)"""

    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return x

class ViTExtractor(nn.Module):
    """Vision Transformer feature extractor (768-dim)"""

    def __init__(self, pretrained=True):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.vit.head = nn.Identity()

    def forward(self, x):
        x = self.vit(x)
        return x

class HybridFeatureExtractor(nn.Module):
    """Combines ResNet50 and ViT features"""

    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = ResNet50Extractor(pretrained=pretrained)
        self.vit = ViTExtractor(pretrained=pretrained)

    def forward(self, x):
        resnet_feat = self.resnet(x)
        vit_feat = self.vit(x)
        combined_feat = torch.cat([resnet_feat, vit_feat], dim=1)
        return combined_feat
