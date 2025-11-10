"""
Training script with optimizer switching support.

This script extends the standard training to support dynamic optimizer switching.
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
from optimizer_switching import OptimizerSwitchingStrategy, get_switching_strategy, list_strategies
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
    """Main training function with optimizer switching."""
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA - {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")
    
    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to {args.seed}")
    
    # Create save directory
    exp_name = f"{args.dataset}_switching_{args.strategy}_seed{args.seed}"
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    
    # Load data
    print(f"\nLoading {args.dataset.upper()} dataset...")
    train_loader, test_loader, num_classes = get_dataloader(
        dataset=args.dataset,
        batch_size=args.batch_size,
        augment=args.augment,
        num_workers=args.workers
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = get_model(args.model, num_classes=num_classes)
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    
    # Get switching strategy configuration
    strategy_config = get_switching_strategy(args.strategy)
    print(f"\n🔄 Using switching strategy: {strategy_config['name']}")
    
    # Initialize optimizer switching manager
    switching_manager = OptimizerSwitchingStrategy(
        model=model,
        strategy_config=strategy_config,
        device=device
    )
    
    # Get initial optimizer and scheduler
    optimizer = switching_manager.get_optimizer()
    scheduler = switching_manager.get_scheduler()
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Metrics logger
    logger = MetricsLogger(save_dir)
    
    # Add switching info to logger
    logger.metrics['switching_strategy'] = args.strategy
    logger.metrics['optimizer_switches'] = []
    
    # Save configuration
    config = {
        **vars(args),
        'switching_strategy': strategy_config,
        'parameters': n_params
    }
    config_path = os.path.join(save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
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
        
        # Log metrics (including current optimizer stage)
        logger.log(
            epoch, train_loss, train_acc, test_loss, test_acc, 
            current_lr, epoch_time
        )
        
        # Log current optimizer
        stage_name = switching_manager.get_current_stage_name()
        if 'current_optimizer' not in logger.metrics:
            logger.metrics['current_optimizer'] = []
        logger.metrics['current_optimizer'].append(stage_name)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} - {format_time(epoch_time)} [{stage_name}]")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, logger.metrics, best_model_path)
            print(f"  ✓ New best accuracy: {best_acc:.2f}%")
        
        # Check if optimizer should switch
        switched = switching_manager.step(epoch, train_loss, test_acc)
        if switched:
            # Update optimizer and scheduler references
            optimizer = switching_manager.get_optimizer()
            scheduler = switching_manager.get_scheduler()
            
            # Log the switch
            logger.metrics['optimizer_switches'].append({
                'epoch': epoch,
                'test_acc': test_acc,
                'train_loss': train_loss
            })
        
        # Save metrics periodically
        if epoch % 10 == 0 or epoch == args.epochs:
            logger.save()
        
        # Clear cache periodically
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
    
    # Print switching history
    switch_history = switching_manager.get_switch_history()
    if switch_history:
        print(f"\n  Optimizer switches: {len(switch_history)}")
        for switch in switch_history:
            print(f"    Epoch {switch['epoch']}: {switch['from'].upper()} → {switch['to'].upper()}")
    
    print(f"{'='*60}\n")
    
    # Save final metrics and model
    logger.metrics['switch_history'] = switch_history
    logger.save()
    
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs, logger.metrics, final_model_path)
    
    # Save summary
    summary = {
        'strategy': args.strategy,
        'dataset': args.dataset,
        'model': args.model,
        'seed': args.seed,
        'best_accuracy': best_acc,
        'final_accuracy': test_acc,
        'total_time': total_time,
        'epochs': args.epochs,
        'parameters': n_params,
        'switch_history': switch_history
    }
    summary_path = os.path.join(save_dir, 'summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    return best_acc, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optimizer Comparison Training with Switching Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available strategies
  python train_with_switching.py --list-strategies
  
  # Train with Adam→SGD strategy
  python train_with_switching.py --strategy adam_to_sgd --dataset cifar10
  
  # Train with 3-stage strategy
  python train_with_switching.py --strategy adam_adamw_sgd --dataset cifar10 --epochs 200
        """
    )
    
    # Special action
    parser.add_argument('--list-strategies', action='store_true',
                        help='List all available switching strategies and exit')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to use')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation')
    parser.add_argument('--no-augment', action='store_false', dest='augment',
                        help='Disable data augmentation')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Model architecture')
    
    # Switching strategy
    parser.add_argument('--strategy', type=str, default='adam_to_sgd',
                        help='Optimizer switching strategy (use --list-strategies to see options)')
    
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
    
    # Handle list-strategies
    if args.list_strategies:
        list_strategies()
        sys.exit(0)
    
    # Run training
    main(args)

