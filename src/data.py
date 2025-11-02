"""
Data loading utilities for CIFAR-10 and CIFAR-100.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import platform


def get_cifar_transforms(dataset='cifar10', augment=True):
    """
    Get data transforms for CIFAR datasets.
    
    Args:
        dataset: 'cifar10' or 'cifar100'
        augment: Whether to use data augmentation for training
    
    Returns:
        train_transform, test_transform
    """
    # Normalization values for CIFAR
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    else:  # cifar100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    return train_transform, test_transform


def add_label_noise(dataset, noise_rate=0.2):
    """
    Add label noise to dataset for robustness testing.
    
    Args:
        dataset: PyTorch dataset
        noise_rate: Fraction of labels to corrupt (0.0 to 1.0)
    
    Returns:
        Modified dataset with noisy labels
    """
    if noise_rate == 0:
        return dataset
    
    n_samples = len(dataset)
    n_classes = len(dataset.classes)
    n_noisy = int(noise_rate * n_samples)
    
    # Get all targets
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        targets = np.array([dataset[i][1] for i in range(n_samples)])
    
    # Randomly select indices to corrupt
    noisy_idx = np.random.choice(n_samples, n_noisy, replace=False)
    
    # Replace with random labels (different from original)
    for idx in noisy_idx:
        original_label = targets[idx]
        new_label = np.random.choice([i for i in range(n_classes) if i != original_label])
        targets[idx] = new_label
    
    # Update dataset targets
    dataset.targets = targets.tolist()
    
    print(f"Added {noise_rate*100:.1f}% label noise ({n_noisy}/{n_samples} samples)")
    return dataset


def get_dataloader(dataset='cifar10', batch_size=128, augment=True, 
                   num_workers=4, data_fraction=1.0, label_noise=0.0):
    """
    Get train and test dataloaders for CIFAR datasets.
    
    Args:
        dataset: 'cifar10' or 'cifar100'
        batch_size: Batch size for training
        augment: Whether to use data augmentation
        num_workers: Number of workers for data loading
        data_fraction: Fraction of training data to use (for limited data experiments)
        label_noise: Fraction of labels to corrupt (for robustness experiments)
    
    Returns:
        train_loader, test_loader, num_classes
    """
    train_transform, test_transform = get_cifar_transforms(dataset, augment)
    
    # Load datasets
    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )
        num_classes = 10
    elif dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=test_transform
        )
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Add label noise if specified
    if label_noise > 0:
        train_dataset = add_label_noise(train_dataset, label_noise)
    
    # Use subset of data if specified
    if data_fraction < 1.0:
        n_samples = len(train_dataset)
        n_subset = int(data_fraction * n_samples)
        indices = np.random.choice(n_samples, n_subset, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f"Using {data_fraction*100:.1f}% of training data ({n_subset}/{n_samples} samples)")
    
    # Create dataloaders
    # Note: pin_memory is not supported on MPS devices
    use_pin_memory = torch.cuda.is_available()
    
    # On Windows, multiprocessing can cause memory issues, so use 0 workers
    if platform.system() == 'Windows' and num_workers > 0:
        print(f"Warning: Windows detected. Setting num_workers to 0 to avoid memory issues.")
        num_workers = 0
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
        persistent_workers=False  # Disable persistent workers to help with memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
        persistent_workers=False  # Disable persistent workers to help with memory
    )
    
    return train_loader, test_loader, num_classes

