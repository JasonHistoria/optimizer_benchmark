"""
Optimized model architectures specifically designed for CIFAR-100.

These models are:
1. Faster to train than ResNet-18
2. Better for comparing optimizers (moderate capacity)
3. Designed for 32x32 images (no wasted downsampling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for CIFAR."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    """Pre-activation residual block (better for deep networks)."""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        if self.dropout:
            out = self.dropout(out)
        out = self.conv2(out)
        out += shortcut
        return out


class WideResNet_CIFAR(nn.Module):
    """
    Wide ResNet optimized for CIFAR-100.
    
    This is a smaller version than the famous WRN-28-10.
    WRN-16-4: 16 layers, widening factor 4
    - Parameters: ~2.7M (vs ResNet-18's 11M)
    - Faster training but maintains good capacity
    - Better for comparing optimizers
    """
    def __init__(self, depth=16, widen_factor=4, dropout=0.3, num_classes=100):
        super().__init__()
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        
        nStages = [16, 16*k, 32*k, 64*k]
        
        self.conv1 = nn.Conv2d(3, nStages[0], 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(PreActBlock, nStages[0], nStages[1], n, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(PreActBlock, nStages[1], nStages[2], n, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(PreActBlock, nStages[2], nStages[3], n, stride=2, dropout=dropout)
        self.bn = nn.BatchNorm2d(nStages[3])
        self.fc = nn.Linear(nStages[3], num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride, dropout):
        layers = []
        layers.append(block(in_channels, out_channels, stride, dropout))
        for _ in range(num_blocks - 1):
            layers.append(block(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet_CIFAR(nn.Module):
    """
    ResNet optimized for CIFAR (no wasted downsampling).
    
    ResNet-20: 20 layers, ~0.27M parameters
    - Much faster than ResNet-18
    - Good for quick experiments
    - Still shows optimizer differences
    """
    def __init__(self, depth=20, num_classes=100):
        super().__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(16, 16, n, stride=1)
        self.layer2 = self._make_layer(16, 32, n, stride=2)
        self.layer3 = self._make_layer(32, 64, n, stride=2)
        
        self.fc = nn.Linear(64, num_classes)
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_model_for_cifar100(model_name='wrn-16-4', num_classes=100):
    """
    Get model optimized for CIFAR-100.
    
    Args:
        model_name: 'wrn-16-4', 'wrn-16-8', 'resnet20', 'resnet32'
        num_classes: Number of classes (default 100)
    
    Returns:
        PyTorch model
    """
    if model_name == 'wrn-16-4':
        # Recommended: Good balance of speed and accuracy
        # ~2.7M params, ~70-73% accuracy
        model = WideResNet_CIFAR(depth=16, widen_factor=4, dropout=0.3, num_classes=num_classes)
    
    elif model_name == 'wrn-16-8':
        # Higher capacity version
        # ~11M params, ~72-75% accuracy
        model = WideResNet_CIFAR(depth=16, widen_factor=8, dropout=0.3, num_classes=num_classes)
    
    elif model_name == 'wrn-28-4':
        # Deeper version (slower but better)
        # ~5.8M params, ~73-76% accuracy
        model = WideResNet_CIFAR(depth=28, widen_factor=4, dropout=0.3, num_classes=num_classes)
    
    elif model_name == 'resnet20':
        # Very fast, for quick experiments
        # ~0.27M params, ~65-68% accuracy
        model = ResNet_CIFAR(depth=20, num_classes=num_classes)
    
    elif model_name == 'resnet32':
        # Good baseline
        # ~0.46M params, ~68-70% accuracy
        model = ResNet_CIFAR(depth=32, num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    print("=" * 60)
    print("Model Architectures for CIFAR-100")
    print("=" * 60)
    
    models = {
        'WRN-16-4 (Recommended)': 'wrn-16-4',
        'WRN-16-8 (High Capacity)': 'wrn-16-8',
        'WRN-28-4 (Deeper)': 'wrn-28-4',
        'ResNet-20 (Fast)': 'resnet20',
        'ResNet-32 (Baseline)': 'resnet32',
    }
    
    for name, model_name in models.items():
        model = get_model_for_cifar100(model_name)
        params = count_parameters(model)
        
        # Test forward pass
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            y = model(x)
        
        print(f"\n{name}:")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {y.shape}")
        print(f"  Memory: ~{params * 4 / 1024 / 1024:.1f} MB")



