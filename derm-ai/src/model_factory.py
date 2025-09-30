# src/model_factory.py
import timm
import torch.nn as nn

# Gợi ý backbone mạnh & gọn
DEFAULT_ARCH = "convnext_tiny"  # 224x224, nhanh & tốt

def create_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Tạo model từ timm và đặt num_classes cho classifier.
    """
    model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
    return model
