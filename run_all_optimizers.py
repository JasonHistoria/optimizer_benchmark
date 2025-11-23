#!/usr/bin/env python3
"""
Run experiments for all optimizers with multiple seeds.
This script sequentially runs all optimizer comparisons.
"""

import subprocess
import argparse
import time
from datetime import datetime


def run_experiment(optimizer, dataset, seed, epochs):
    """Run a single training experiment."""
    
    # Use optimized script for CIFAR-100, original script for CIFAR-10
    if dataset == 'cifar100':
        cmd = [
            'python', 'train_cifar100_optimized.py',
            '--optimizer', optimizer,
            '--epochs', str(epochs),
            '--seed', str(seed),
            '--batch-size', '128',
            '--model', 'wrn-16-4',
            '--scheduler', 'cosine',
            '--save-dir', './results'  # Save to results directory
        ]
    else:  # cifar10
        cmd = [
            'python', 'src/train.py',
            '--dataset', dataset,
            '--optimizer', optimizer,
            '--epochs', str(epochs),
            '--seed', str(seed),
            '--batch-size', '128',
            '--model', 'resnet18',
            '--scheduler', 'cosine'
        ]
    
    print(f"\n{'='*60}")
    print(f"Running: {optimizer.upper()} on {dataset.upper()}, seed={seed}")
    if dataset == 'cifar100':
        print(f"Using optimized training script: train_cifar100_optimized.py")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✓ Completed in {elapsed_time/60:.1f} minutes")
    else:
        print(f"\n✗ Failed with return code {result.returncode}")
    
    return result.returncode


def main(args):
    """Run all experiments."""
    
    optimizers = ['sgd', 'adam', 'adamw', 'radam', 'lion']
    seeds = [42, 123, 456]  # Three seeds for statistical significance
    
    # Use specified settings
    if args.optimizers:
        optimizers = args.optimizers
    if args.seeds:
        seeds = args.seeds
    
    total_experiments = len(optimizers) * len(args.datasets) * len(seeds)
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZER COMPARISON EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Optimizers: {', '.join(optimizers)}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Seeds: {seeds}")
    print(f"Epochs per experiment: {args.epochs}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*60}\n")
    
    # Track results
    results = []
    failed_experiments = []
    
    start_time = time.time()
    experiment_num = 0
    
    # Run all combinations
    for dataset in args.datasets:
        for optimizer in optimizers:
            for seed in seeds:
                experiment_num += 1
                
                print(f"\nExperiment {experiment_num}/{total_experiments}")
                
                returncode = run_experiment(optimizer, dataset, seed, args.epochs)
                
                results.append({
                    'dataset': dataset,
                    'optimizer': optimizer,
                    'seed': seed,
                    'success': returncode == 0
                })
                
                if returncode != 0:
                    failed_experiments.append(f"{dataset}_{optimizer}_seed{seed}")
                
                # Small delay between experiments
                if experiment_num < total_experiments:
                    time.sleep(2)
    
    # Print summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Successful: {successful}/{total_experiments}")
    
    if failed_experiments:
        print(f"\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    else:
        print(f"\n✓ All experiments completed successfully!")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all optimizer experiments')
    
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['cifar10', 'cifar100'],
                        choices=['cifar10', 'cifar100'],
                        help='Datasets to run experiments on')
    parser.add_argument('--optimizers', type=str, nargs='+',
                        default=None,
                        choices=['sgd', 'adam', 'adamw', 'radam', 'lion'],
                        help='Optimizers to test (default: all)')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=None,
                        help='Random seeds (default: 42, 123, 456)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs per experiment')
    
    args = parser.parse_args()
    
    main(args)

