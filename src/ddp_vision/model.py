from __future__ import annotations

import torch.nn as nn
from torchvision import models

from ddp_vision.config import TrainConfig

def build_model(config: TrainConfig) -> nn.Module:
    if config.model_name != "resnet50":
        raise ValueError(f"Unsupported model_name={config.model_name!r}. Currently supported: resnet50.")
    
    weights = models.ResNet50_Weights.DEFAULT if config.pretrained else None
    model = models.resnet50(weights=weights)
    if config.num_classes != model.fc.out_features:
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    return model