import torch.nn as nn
from torchvision import models


# build a pretrained model and swap the last layer for the 6 classes
def build_model(arch, num_classes, pretrained=True):
    if arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError("unknown arch: " + arch)
    return m


def freeze_backbone(model, arch, freeze):
    # freeze everything except the final layer
    if arch == "efficientnet_b0":
        for p in model.features.parameters():
            p.requires_grad = not freeze
    else:
        head = set(id(p) for p in model.fc.parameters())
        for p in model.parameters():
            if id(p) not in head:
                p.requires_grad = not freeze
