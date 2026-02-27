from __future__ import annotations

import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if name.startswith("heads.head"):
            param.requires_grad = True
        else:
            param.requires_grad = trainable
