from __future__ import annotations

from typing import Tuple

import torch
from torchvision import models, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
)


AVAILABLE_MODELS = ("resnet50", "efficientnet_b0", "vit_b_16")


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> Tuple[torch.nn.Module, int]:
    """Create a transfer-learning model and return (model, input_size)."""
    model_name = model_name.lower()

    if model_name == "resnet50":
        weights = _safe_weights(ResNet50_Weights.DEFAULT if pretrained else None)
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        return model, 224

    if model_name == "efficientnet_b0":
        weights = _safe_weights(EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        return model, 224

    if model_name == "vit_b_16":
        weights = _safe_weights(ViT_B_16_Weights.DEFAULT if pretrained else None)
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)
        return model, 224

    raise ValueError(f"Unsupported model: {model_name}. Available: {AVAILABLE_MODELS}")


def get_transforms(input_size: int, is_train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((input_size + 32, input_size + 32)),
                transforms.RandomResizedCrop(input_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def _safe_weights(weights):
    """Use pretrained weights when available; fallback to random init offline."""
    if weights is None:
        return None
    try:
        _ = weights.meta
        return weights
    except Exception:
        return None
