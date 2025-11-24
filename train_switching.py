#!/usr/bin/env python
"""
Training script with Dynamic Optimizer Switching.
"""

import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models_cifar100 import get_model_for_cifar100, count_parameters
from data_cifar100 import get_cifar100_loaders
from optimizer_switching import get_switching_strategy
from optimizers import get_scheduler
from utils import set_seed, AverageMeter, accuracy, save_checkpoint, format_time


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch:3d} [Train]', ncols=100)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        # Measure accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.3f}',
            'acc': f'{top1.avg:.1f}%'
        })
    
    return losses.avg, top1.avg, top5.avg


@torch.no_grad()
def validate(model, test_loader, criterion, device):
    """Validate the model."""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for inputs, targets in tqdm(test_loader, desc='[Val]', ncols=100):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Measure accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
    
    return losses.avg, top1.avg, top5.avg


def main(args):
    """Main training function."""
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory
    exp_name = f"switching_{args.strategy}_{args.model}_seed{args.seed}"
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
    # Save configuration
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    # Load data
    print(f"\nLoading {args.dataset.upper()}...")
    if args.dataset == 'cifar100':
        train_loader, test_loader = get_cifar100_loaders(
            batch_size=args.batch_size,
            num_workers=args.workers,
            augmentation='strong'
        )
        num_classes = 100
    else:
        from data import get_dataloader
        train_loader, test_loader, num_classes = get_dataloader(
            dataset='cifar10',
            batch_size=args.batch_size,
            num_workers=args.workers,
            augment=True
        )

    # Create model
    print(f"\nCreating {args.model} model...")
    if args.dataset == 'cifar100':
        model = get_model_for_cifar100(args.model, num_classes=100)
    else:
        from models import get_model
        # Map cifar100 model names to cifar10 equivalents if needed, or just use get_model
        # For now, simple fallback
        if 'wrn' in args.model:
             print("Warning: WRN models might not be standard in src/models.py for CIFAR-10. Trying anyway.")
             # You might need to import WRN for CIFAR10 if not present, or force resnet18
        
        # Actually, for simplicity in this script, let's stick to the requested model
        # But src/models.py usually has ResNet18.
        if args.model == 'resnet18':
             model = get_model('resnet18', num_classes=10)
        else:
             # Fallback to cifar100 model function but with 10 classes if compatible
             # Or just raise error if mixing model types.
             # Let's assume users use consistent models.
             model = get_model_for_cifar100(args.model, num_classes=10)

    model = model.to(device)
    
    # Initialize Switching Strategy
    print(f"\nInitializing Switching Strategy: {args.strategy}")
    switcher = get_switching_strategy(args.strategy, model, args.epochs)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting Training with Strategy: {args.strategy}")
    print(f"{'='*70}\n")
    
    best_acc = 0.0
    metrics = {'train_loss': [], 'test_acc': [], 'switch_epoch': None}
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # 1. Check for Adaptive Switch based on previous metrics (before getting optimizer)
        force_switch = False
        if switcher.is_adaptive and metrics['test_acc']:
            # Update monitor with last known test_acc
            force_switch = switcher.update_metrics(epoch, metrics['test_acc'][-1])
        
        # 2. Get Optimizer (handles switching)
        optimizer = switcher.get_optimizer(epoch, force_switch=force_switch)
        current_opt_name = type(optimizer).__name__
        
        # Record switch epoch if it just happened
        if switcher.has_switched and metrics['switch_epoch'] is None:
             metrics['switch_epoch'] = epoch
        
        # Train
        train_loss, train_acc, _ = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args
        )
        
        # Validate
        test_loss, test_acc, _ = validate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch}/{args.epochs} [{current_opt_name}] - {format_time(epoch_time)}")
        print(f"  Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        metrics['train_loss'].append(train_loss)
        metrics['test_acc'].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, optimizer, epoch, metrics, os.path.join(save_dir, 'best_model.pth'))
            print(f"  >>> New Best: {best_acc:.2f}%")

    print(f"\nFinal Best Accuracy: {best_acc:.2f}%")
    
    # Save Final Summary
    import json
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimizer Switching Experiments')
    parser.add_argument('--strategy', type=str, required=True,
                        help='Switching strategy name (e.g., adam_to_sgd, adaptive_adam_to_sgd, adam_to_sgd_at_50)')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, default='wrn-16-4',
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--workers', type=int, default=4,
                        help='Data workers')
    parser.add_argument('--save-dir', type=str, default='./results_switching',
                        help='Save directory')
    
    args = parser.parse_args()
    main(args)

