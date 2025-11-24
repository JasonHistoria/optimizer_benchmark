#!/usr/bin/env python
"""
Optimized training script for CIFAR-100.

Key improvements:
1. Lighter, faster models (WRN-16-4 instead of ResNet-18)
2. Strong regularization (label smoothing, dropout, weight decay)
3. Better data augmentation (AutoAugment + Cutout)
4. Optimized hyperparameters for each optimizer
5. Fast training with early stopping option

Expected performance:
- WRN-16-4: 70-73% accuracy in ~100 epochs
- 2-3x faster than ResNet-18
- Clear optimizer differences
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
from optimizers import get_optimizer, get_scheduler
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
        
        # Gradient clipping (optional, helps stability)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
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
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print(f"Using CPU")
    
    # Set random seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")
    
    # Create save directory
    # Support custom experiment name suffix (e.g., for radam-v2)
    exp_suffix = getattr(args, 'exp_suffix', '')
    if exp_suffix:
        exp_name = f"cifar100_{args.model}_{args.optimizer}{exp_suffix}_seed{args.seed}"
    else:
        exp_name = f"cifar100_{args.model}_{args.optimizer}_seed{args.seed}"
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
    # Save configuration
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    # Load data
    print(f"\nLoading CIFAR-100 with '{args.augmentation}' augmentation...")
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        augmentation=args.augmentation,
        pin_memory=(device.type == 'cuda')
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = get_model_for_cifar100(args.model, num_classes=100)
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")
    print(f"Memory: ~{n_params * 4 / 1024 / 1024:.1f} MB")
    
    # Create optimizer
    print(f"\nOptimizer: {args.optimizer.upper()}")
    optimizer = get_optimizer(
        args.optimizer,
        model,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    
    # Learning rate scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        epochs=args.epochs
    )
    if scheduler:
        print(f"LR Scheduler: {args.scheduler}")
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.label_smoothing > 0:
        print(f"Label smoothing: {args.label_smoothing}")
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"Training for {args.epochs} epochs")
    print(f"{'='*70}\n")
    
    best_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    # Metrics tracking
    metrics = {
        'train_loss': [], 'train_acc': [], 'train_acc5': [],
        'test_loss': [], 'test_acc': [], 'test_acc5': [],
        'lr': [], 'epoch_time': []
    }
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        train_loss, train_acc, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args
        )
        
        # Validate
        test_loss, test_acc, test_acc5 = validate(model, test_loader, criterion, device)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['train_acc5'].append(train_acc5)
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)
        metrics['test_acc5'].append(test_acc5)
        metrics['lr'].append(current_lr)
        metrics['epoch_time'].append(epoch_time)
        
        # Print epoch summary
        print(f"\nEpoch {epoch:3d}/{args.epochs} - {format_time(epoch_time)}")
        print(f"  Train: Loss={train_loss:.4f}, Top1={train_acc:.2f}%, Top5={train_acc5:.2f}%")
        print(f"  Test:  Loss={test_loss:.4f}, Top1={test_acc:.2f}%, Top5={test_acc5:.2f}%")
        print(f"  Gap: {train_acc - test_acc:.2f}%  |  LR: {current_lr:.6f}")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, metrics, best_model_path)
            print(f"  >>> New best: {best_acc:.2f}% (epoch {best_epoch})")
        
        # Early stopping check
        if args.early_stop > 0 and epoch - best_epoch >= args.early_stop:
            print(f"\nEarly stopping: no improvement for {args.early_stop} epochs")
            break
        
        # Save metrics periodically
        if epoch % 10 == 0 or epoch == args.epochs:
            import json
            with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
        
        print()
    
    # Training complete
    total_time = time.time() - start_time
    print(f"{'='*70}")
    print(f"Training complete!")
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Best accuracy: {best_acc:.2f}% (epoch {best_epoch})")
    print(f"  Final accuracy: {test_acc:.2f}%")
    print(f"  Avg epoch time: {format_time(total_time / epoch)}")
    print(f"{'='*70}\n")
    
    # Save final metrics
    import json
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save summary
    summary = {
        'optimizer': args.optimizer,
        'model': args.model,
        'seed': args.seed,
        'best_accuracy': best_acc,
        'best_epoch': best_epoch,
        'final_accuracy': test_acc,
        'total_time': total_time,
        'epochs_trained': epoch,
        'parameters': n_params,
        'augmentation': args.augmentation,
        'label_smoothing': args.label_smoothing
    }
    
    with open(os.path.join(save_dir, 'summary.yaml'), 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, epoch, metrics, final_model_path)
    
    return best_acc, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized CIFAR-100 Training')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='wrn-16-4',
                        choices=['wrn-16-4', 'wrn-16-8', 'wrn-28-4', 'resnet20', 'resnet32'],
                        help='Model architecture (default: wrn-16-4)')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adam', 'adamw', 'radam', 'lion', 'muon'],
                        help='Optimizer (default: adamw)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (uses optimizer default if not specified)')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay (uses optimizer default if not specified)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'step', 'multistep'],
                        help='LR scheduler (default: cosine)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Regularization arguments
    parser.add_argument('--augmentation', type=str, default='strong',
                        choices=['basic', 'medium', 'strong'],
                        help='Data augmentation level (default: strong)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--grad-clip', type=float, default=0,
                        help='Gradient clipping (0 = disabled)')
    
    # Output arguments
    parser.add_argument('--save-dir', type=str, default='./results_cifar100',
                        help='Directory to save results')
    parser.add_argument('--early-stop', type=int, default=0,
                        help='Early stopping patience (0 = disabled)')
    parser.add_argument('--exp-suffix', type=str, default='',
                        help='Suffix to add to experiment name (e.g., "-v2")')
    
    args = parser.parse_args()
    
    # Set default hyperparameters based on optimizer
    if args.lr is None or args.weight_decay is None:
        from optimizers import get_default_config
        default_config = get_default_config(args.optimizer)
        
        if args.lr is None:
            # Adjust learning rates for faster models
            if args.optimizer == 'sgd':
                args.lr = 0.1
            elif args.optimizer in ['adam', 'adamw', 'radam']:
                args.lr = 0.001
            elif args.optimizer == 'lion':
                args.lr = 0.0001
            elif args.optimizer == 'muon':
                args.lr = 0.001 # Adam part LR
        
        if args.weight_decay is None:
            args.weight_decay = default_config['weight_decay']
    
    # Run training
    main(args)


