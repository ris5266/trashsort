import torch.nn as nn
from torchvision import models

# build the efficientnet material classifier
def build_model(num_classes, pretrained=True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def freeze_backbone(model, freeze):
    for parameter in model.features.parameters():
        parameter.requires_grad = not freeze
