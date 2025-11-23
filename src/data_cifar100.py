"""
Optimized data loading and augmentation for CIFAR-100.
Includes strong regularization techniques.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import platform


class Cutout:
    """
    Cutout augmentation: randomly mask out square regions.
    Reference: https://arxiv.org/abs/1708.04552
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img: Tensor of shape [C, H, W]
        Returns:
            Tensor with cutout applied
        """
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def get_cifar100_transforms(augmentation_level='strong'):
    """
    Get optimized transforms for CIFAR-100.
    
    Args:
        augmentation_level: 'basic', 'medium', 'strong'
    
    Returns:
        train_transform, test_transform
    """
    # CIFAR-100 normalization
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    if augmentation_level == 'basic':
        # Basic augmentation (similar to original)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    elif augmentation_level == 'medium':
        # Medium: Add color jitter and rotation
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    elif augmentation_level == 'strong':
        # Strong: Add AutoAugment or RandAugment + Cutout
        try:
            # Try to use AutoAugment (PyTorch >= 1.7)
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                Cutout(n_holes=1, length=16)
            ])
        except AttributeError:
            # Fallback to medium + cutout
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                Cutout(n_holes=1, length=16)
            ])
    
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    return train_transform, test_transform


def get_cifar100_loaders(batch_size=128, num_workers=4, augmentation='strong', 
                         pin_memory=True, validation_split=0.0):
    """
    Get CIFAR-100 data loaders with optimized settings.
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        augmentation: 'basic', 'medium', or 'strong'
        pin_memory: Pin memory for faster GPU transfer
        validation_split: Fraction of training data for validation (0.0-0.2)
    
    Returns:
        train_loader, test_loader, (val_loader if validation_split > 0)
    """
    train_transform, test_transform = get_cifar100_transforms(augmentation)
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Optional validation split
    if validation_split > 0:
        n_train = len(train_dataset)
        n_val = int(n_train * validation_split)
        n_train = n_train - n_val
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
    else:
        val_loader = None
    
    # Adjust num_workers for Windows
    if platform.system() == 'Windows' and num_workers > 0:
        print(f"Warning: Windows detected. Setting num_workers to 0.")
        num_workers = 0
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    if val_loader:
        return train_loader, test_loader, val_loader
    else:
        return train_loader, test_loader


if __name__ == '__main__':
    # Test data loading
    print("=" * 60)
    print("Testing CIFAR-100 Data Loading")
    print("=" * 60)
    
    for aug_level in ['basic', 'medium', 'strong']:
        print(f"\nAugmentation: {aug_level}")
        train_loader, test_loader = get_cifar100_loaders(
            batch_size=128, augmentation=aug_level
        )
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Check batch
        images, labels = next(iter(train_loader))
        print(f"  Batch shape: {images.shape}")
        print(f"  Label range: [{labels.min()}, {labels.max()}]")



