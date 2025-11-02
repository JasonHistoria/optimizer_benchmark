"""
Main training script for optimizer comparison experiments.
"""

import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from models import get_model, count_parameters
from data import get_dataloader
from optimizers import get_optimizer, get_scheduler, get_default_config
from utils import (
    MetricsLogger, set_seed, AverageMeter, accuracy, 
    save_checkpoint, format_time
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
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
        acc = accuracy(outputs, targets)[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc.item(), inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{top1.avg:.2f}%'
        })
    
    return losses.avg, top1.avg


def validate(model, test_loader, criterion, device):
    """Validate the model."""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='[Val]'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Measure accuracy
            acc = accuracy(outputs, targets)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc.item(), inputs.size(0))
    
    return losses.avg, top1.avg


def main(args):
    """Main training function."""
    
    # Set device (support CUDA, MPS, and CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA - {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")
    print(f"Device: {device}")
    
    # Set random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to {args.seed}")
    
    # Create save directory
    exp_name = f"{args.dataset}_{args.optimizer}_lr{args.lr}_wd{args.weight_decay}_seed{args.seed}"
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    
    # Save configuration
    config_path = os.path.join(save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    # Load data
    print(f"\nLoading {args.dataset.upper()} dataset...")
    train_loader, test_loader, num_classes = get_dataloader(
        dataset=args.dataset,
        batch_size=args.batch_size,
        augment=args.augment,
        num_workers=args.workers,
        data_fraction=args.data_fraction,
        label_noise=args.label_noise
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = get_model(args.model, num_classes=num_classes)
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    
    # Create optimizer
    print(f"\nCreating {args.optimizer.upper()} optimizer...")
    optimizer = get_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"Learning rate: {args.lr}, Weight decay: {args.weight_decay}")
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        epochs=args.epochs
    )
    if scheduler:
        print(f"Using {args.scheduler} learning rate scheduler")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Metrics logger
    logger = MetricsLogger(save_dir)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logger.log(epoch, train_loss, train_acc, test_loss, test_acc, current_lr, epoch_time)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} - {format_time(epoch_time)}")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, logger.metrics, best_model_path)
            print(f"  ✓ New best accuracy: {best_acc:.2f}%")
        
        # Save metrics periodically
        if epoch % 10 == 0 or epoch == args.epochs:
            logger.save()
        
        # Clear cache periodically to help with memory
        if epoch % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print()
    
    # Training complete
    total_time = time.time() - start_time
    print(f"{'='*60}")
    print(f"Training complete!")
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Best accuracy: {best_acc:.2f}%")
    print(f"  Final accuracy: {test_acc:.2f}%")
    print(f"{'='*60}\n")
    
    # Save final metrics and model
    logger.save()
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs, logger.metrics, final_model_path)
    
    # Save summary
    summary = {
        'optimizer': args.optimizer,
        'dataset': args.dataset,
        'model': args.model,
        'seed': args.seed,
        'best_accuracy': best_acc,
        'final_accuracy': test_acc,
        'total_time': total_time,
        'epochs': args.epochs,
        'parameters': n_params
    }
    summary_path = os.path.join(save_dir, 'summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    return best_acc, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimizer Comparison Training')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to use')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                        help='Fraction of training data to use (0.0-1.0)')
    parser.add_argument('--label-noise', type=float, default=0.0,
                        help='Fraction of labels to corrupt (0.0-1.0)')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation')
    parser.add_argument('--no-augment', action='store_false', dest='augment',
                        help='Disable data augmentation')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Model architecture')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'adamw', 'radam', 'lion'],
                        help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (uses default if not specified)')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay (uses default if not specified)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'step', 'multistep'],
                        help='Learning rate scheduler')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--save-dir', type=str, default='./results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Use default hyperparameters if not specified
    if args.lr is None or args.weight_decay is None:
        default_config = get_default_config(args.optimizer)
        if args.lr is None:
            args.lr = default_config['lr']
        if args.weight_decay is None:
            args.weight_decay = default_config['weight_decay']
    
    # Run training
    main(args)

