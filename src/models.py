"""
Model definitions for optimizer comparison experiments.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet18(num_classes=10):
    """
    Get ResNet-18 model with custom number of classes.
    
    Args:
        num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
    
    Returns:
        ResNet-18 model
    """
    model = models.resnet18(weights=None)
    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(model_name, num_classes=10):
    """
    Get model by name.
    
    Args:
        model_name: Name of the model ('resnet18', 'resnet34', 'resnet50')
        num_classes: Number of output classes
    
    Returns:
        PyTorch model
    """
    if model_name == 'resnet18':
        return get_resnet18(num_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

