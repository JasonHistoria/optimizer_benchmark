#!/usr/bin/env python3
"""
Run a single experiment with specified configuration.
"""

import subprocess
import sys
import argparse


def run_experiment(optimizer, dataset='cifar10', seed=42, epochs=200):
    """Run a single training experiment."""
    
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
    
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd)
    
    return result.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run single experiment')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'adamw', 'radam', 'lion'],
                        help='Optimizer to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    
    args = parser.parse_args()
    
    returncode = run_experiment(
        optimizer=args.optimizer,
        dataset=args.dataset,
        seed=args.seed,
        epochs=args.epochs
    )
    
    sys.exit(returncode)

