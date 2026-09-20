import torch
import torch.nn as nn

def create_detector_model(backbone: str = "resnet18", num_classes: int = 2, pretrained: bool = True):
    """
    Creates an AI Image Detector backbone (ResNet-18 or EfficientNet-B0)
    optimized for fast CUDA inference and training.
    """
    try:
        import timm
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
        return model
    except Exception as e:
        from torchvision import models
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            return model
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
