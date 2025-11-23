#!/usr/bin/env python3
"""
Run RAdam v2 experiments with fixed hyperparameters.
This script runs 3 experiments (seeds: 42, 123, 456) with the corrected RAdam configuration.
Results will be saved as: cifar100_wrn-16-4_radam-v2_seed{seed}
"""

import subprocess
import argparse
import time
from datetime import datetime


def run_radam_experiment(seed, epochs=100):
    """Run a single RAdam v2 training experiment."""
    
    cmd = [
        'python', 'train_cifar100_optimized.py',
        '--optimizer', 'radam',
        '--epochs', str(epochs),
        '--seed', str(seed),
        '--batch-size', '128',
        '--model', 'wrn-16-4',
        '--scheduler', 'cosine',
        '--save-dir', './results',
        '--exp-suffix=-v2'  # Use = to avoid argparse interpreting -v2 as a flag
    ]
    
    print(f"\n{'='*60}")
    print(f"Running RAdam v2 experiment: seed={seed}")
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
    """Run all RAdam v2 experiments."""
    
    seeds = args.seeds
    total_experiments = len(seeds)
    
    print(f"\n{'='*60}")
    print(f"RADAM V2 EXPERIMENTS (Fixed Configuration)")
    print(f"{'='*60}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Optimizer: RAdam (with fixed weight_decay=0.001)")
    print(f"Model: WRN-16-4")
    print(f"Seeds: {seeds}")
    print(f"Epochs per experiment: {args.epochs}")
    print(f"Total experiments: {total_experiments}")
    print(f"Results will be saved as: cifar100_wrn-16-4_radam-v2_seed{{seed}}")
    print(f"{'='*60}\n")
    
    # Track results
    results = []
    failed_experiments = []
    
    start_time = time.time()
    experiment_num = 0
    
    # Run all seeds
    for seed in seeds:
        experiment_num += 1
        
        print(f"\nExperiment {experiment_num}/{total_experiments}")
        
        returncode = run_radam_experiment(seed, args.epochs)
        
        results.append({
            'seed': seed,
            'success': returncode == 0
        })
        
        if returncode != 0:
            failed_experiments.append(f"radam-v2_seed{seed}")
        
        # Small delay between experiments
        if experiment_num < total_experiments:
            time.sleep(2)
    
    # Print summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    
    print(f"\n{'='*60}")
    print(f"ALL RADAM V2 EXPERIMENTS COMPLETE")
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
        print(f"\nResults saved in:")
        for seed in seeds:
            print(f"  - results/cifar100_wrn-16-4_radam-v2_seed{seed}/")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RAdam v2 experiments with fixed configuration')
    
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 123, 456],
                        help='Random seeds (default: 42 123 456)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs per experiment (default: 100)')
    
    args = parser.parse_args()
    
    main(args)


