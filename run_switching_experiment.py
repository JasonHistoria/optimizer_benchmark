#!/usr/bin/env python3
"""
Run optimizer switching experiments and compare with baseline.

This script automates running experiments with switching strategies
and comparing them against single-optimizer baselines.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from optimizer_switching import PREDEFINED_STRATEGIES


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n⚠️  Warning: Command failed with return code {result.returncode}")
        return False
    return True


def run_switching_experiment(
    strategy,
    dataset='cifar10',
    epochs=200,
    seed=42,
    batch_size=128
):
    """Run an experiment with optimizer switching."""
    
    cmd = (
        f"python src/train_with_switching.py "
        f"--strategy {strategy} "
        f"--dataset {dataset} "
        f"--epochs {epochs} "
        f"--seed {seed} "
        f"--batch-size {batch_size}"
    )
    
    description = f"Switching Strategy: {strategy} on {dataset.upper()}"
    return run_command(cmd, description)


def run_baseline_experiment(
    optimizer,
    dataset='cifar10',
    epochs=200,
    seed=42,
    batch_size=128
):
    """Run a baseline experiment with single optimizer."""
    
    cmd = (
        f"python src/train.py "
        f"--optimizer {optimizer} "
        f"--dataset {dataset} "
        f"--epochs {epochs} "
        f"--seed {seed} "
        f"--batch-size {batch_size}"
    )
    
    description = f"Baseline: {optimizer.upper()} on {dataset.upper()}"
    return run_command(cmd, description)


def run_comparison_suite(args):
    """Run a full comparison suite."""
    
    print("\n" + "="*60)
    print("🚀 Optimizer Switching Comparison Suite")
    print("="*60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Epochs: {args.epochs}")
    print(f"Seeds: {args.seeds}")
    print(f"Strategies: {', '.join(args.strategies)}")
    print("="*60 + "\n")
    
    total_experiments = 0
    successful_experiments = 0
    
    # Run switching strategies
    for strategy in args.strategies:
        for seed in args.seeds:
            total_experiments += 1
            success = run_switching_experiment(
                strategy=strategy,
                dataset=args.dataset,
                epochs=args.epochs,
                seed=seed,
                batch_size=args.batch_size
            )
            if success:
                successful_experiments += 1
    
    # Run baseline comparisons if requested
    if args.include_baselines:
        baseline_optimizers = ['adam', 'adamw', 'sgd']
        
        for optimizer in baseline_optimizers:
            for seed in args.seeds:
                total_experiments += 1
                success = run_baseline_experiment(
                    optimizer=optimizer,
                    dataset=args.dataset,
                    epochs=args.epochs,
                    seed=seed,
                    batch_size=args.batch_size
                )
                if success:
                    successful_experiments += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 Experiment Suite Summary")
    print("="*60)
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {total_experiments - successful_experiments}")
    print("="*60 + "\n")
    
    if successful_experiments > 0:
        print("✓ Experiments complete! Run visualization:")
        print("  python visualize_switching_results.py")


def main():
    parser = argparse.ArgumentParser(
        description='Run optimizer switching experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single switching experiment
  python run_switching_experiment.py --strategy adam_to_sgd
  
  # Run comparison with multiple strategies
  python run_switching_experiment.py \\
    --strategies adam_to_sgd adam_adamw_sgd \\
    --seeds 42 123 456
  
  # Full comparison including baselines
  python run_switching_experiment.py \\
    --strategies adam_to_sgd adam_adamw_sgd \\
    --include-baselines \\
    --epochs 100
        """
    )
    
    parser.add_argument('--strategy', type=str,
                        help='Single strategy to run')
    parser.add_argument('--strategies', type=str, nargs='+',
                        help='Multiple strategies to compare')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='Random seeds for experiments')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--include-baselines', action='store_true',
                        help='Also run baseline experiments for comparison')
    parser.add_argument('--list-strategies', action='store_true',
                        help='List available strategies and exit')
    
    args = parser.parse_args()
    
    # List strategies
    if args.list_strategies:
        print("\n" + "="*60)
        print("Available Switching Strategies")
        print("="*60 + "\n")
        
        for key, config in PREDEFINED_STRATEGIES.items():
            print(f"📋 {key}")
            print(f"   {config['name']}")
            stage_names = [s['optimizer'].upper() for s in config['stages']]
            print(f"   Pipeline: {' → '.join(stage_names)}")
            print()
        
        sys.exit(0)
    
    # Determine which strategies to run
    if args.strategy:
        args.strategies = [args.strategy]
    elif not args.strategies:
        # Default: run all strategies
        args.strategies = list(PREDEFINED_STRATEGIES.keys())
    
    # Run experiments
    run_comparison_suite(args)


if __name__ == '__main__':
    main()

