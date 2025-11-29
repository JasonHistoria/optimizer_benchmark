#!/usr/bin/env python3
"""
Run hypothesis validation experiments.

H1: RAdam stability (already validated, but can collect more precise data)
H2: AdamW regularization (test with higher weight_decay)
H3: Lion robustness (test with 20% label noise)
H4: Muon data efficiency (test with limited training data - 20%)
"""

import subprocess
import sys
import argparse
import time
import os
from datetime import datetime


# Separate directory for hypothesis test results
HYPOTHESIS_RESULTS_DIR = './results_hypothesis'


def run_experiment(cmd, description, results_dir=HYPOTHESIS_RESULTS_DIR):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Results will be saved to: {results_dir}")
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
    """Run all hypothesis validation experiments."""
    
    # Create hypothesis results directory
    os.makedirs(HYPOTHESIS_RESULTS_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"HYPOTHESIS VALIDATION EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {HYPOTHESIS_RESULTS_DIR}")
    print(f"{'='*60}\n")
    
    results = []
    failed_experiments = []
    
    # H1: RAdam stability (collect precise data for first 10 epochs)
    if args.h1 or args.all:
        print("\n" + "="*60)
        print("H1: RAdam Stability Test")
        print("="*60)
        print("Comparing RAdam-v2 vs Adam on CIFAR-100 (first 10 epochs)")
        print("Note: This validates variance rectification mechanism")
        print("="*60)
        
        seeds = [42, 123, 456]
        for seed in seeds:
            # Adam for comparison
            cmd = [
                sys.executable, 'train_cifar100_optimized.py',
                '--optimizer', 'adam',
                '--epochs', '100',
                '--seed', str(seed),
                '--batch-size', '128',
                '--model', 'wrn-16-4',
                '--scheduler', 'cosine',
                '--save-dir', HYPOTHESIS_RESULTS_DIR,
                '--exp-suffix', '_h1'  # Mark as H1 experiment
            ]
            desc = f"H1: Adam baseline (seed={seed})"
            returncode = run_experiment(cmd, desc)
            if returncode != 0:
                failed_experiments.append(f"H1_adam_seed{seed}")
            results.append(('H1', 'adam', seed, returncode == 0))
            time.sleep(2)
    
    # H2: AdamW regularization (test with higher weight_decay)
    if args.h2 or args.all:
        print("\n" + "="*60)
        print("H2: AdamW Regularization Test")
        print("="*60)
        print("Comparing Adam vs AdamW with high weight_decay (≥0.01)")
        print("Testing train-test gap difference")
        print("="*60)
        
        seeds = [42, 123, 456]
        for seed in seeds:
            # Adam with high weight_decay
            # For H2, we can reduce augmentation to better see the regularization effect
            # Use 'medium' augmentation instead of 'strong' to reduce Train < Test gap effect
            cmd = [
                sys.executable, 'train_cifar100_optimized.py',
                '--optimizer', 'adam',
                '--epochs', '50',
                '--seed', str(seed),
                '--batch-size', '128',
                '--model', 'wrn-16-4',
                '--scheduler', 'cosine',
                '--weight-decay', '0.01',  # High weight decay
                '--augmentation', 'medium',  # Reduced augmentation for H2
                '--save-dir', HYPOTHESIS_RESULTS_DIR,
                '--exp-suffix', '_h2_wd0.01'  # Mark as H2 experiment
            ]
            desc = f"H2: Adam with high WD=0.01 (seed={seed}, medium aug)"
            returncode = run_experiment(cmd, desc)
            if returncode != 0:
                failed_experiments.append(f"H2_adam_highwd_seed{seed}")
            results.append(('H2', 'adam_highwd', seed, returncode == 0))
            time.sleep(2)
            
            # AdamW with high weight_decay
            cmd = [
                sys.executable, 'train_cifar100_optimized.py',
                '--optimizer', 'adamw',
                '--epochs', '50',
                '--seed', str(seed),
                '--batch-size', '128',
                '--model', 'wrn-16-4',
                '--scheduler', 'cosine',
                '--weight-decay', '0.01',  # High weight decay
                '--augmentation', 'medium',  # Reduced augmentation for H2
                '--save-dir', HYPOTHESIS_RESULTS_DIR,
                '--exp-suffix', '_h2_wd0.01'  # Mark as H2 experiment
            ]
            desc = f"H2: AdamW with high WD=0.01 (seed={seed}, medium aug)"
            returncode = run_experiment(cmd, desc)
            if returncode != 0:
                failed_experiments.append(f"H2_adamw_highwd_seed{seed}")
            results.append(('H2', 'adamw_highwd', seed, returncode == 0))
            time.sleep(2)
    
    # H3: Lion robustness (test with 20% label noise)
    if args.h3 or args.all:
        print("\n" + "="*60)
        print("H3: Lion Robustness Test")
        print("="*60)
        print("Comparing Lion vs Adam under 30% label noise on CIFAR-10")
        print("Testing sign-based update robustness with large batch and higher Lion weight decay")
        print("="*60)
        
        seeds = [42, 123, 456]
        for seed in seeds:
            # Adam with 30% label noise
            # For H3, we can use no augmentation to focus on noise robustness
            cmd = [
                sys.executable, 'src/train.py',
                '--dataset', 'cifar10',
                '--optimizer', 'adam',
                '--epochs', '50',
                '--seed', str(seed),
                '--batch-size', '256',
                '--model', 'resnet18',
                '--scheduler', 'cosine',
                '--label-noise', '0.3',  # 30% label noise
                # Use moderately high weight decay for Adam baseline
                '--weight-decay', '0.001',
                '--augment',  # Use basic augmentation (default in train.py)
                '--save-dir', HYPOTHESIS_RESULTS_DIR,
            ]
            desc = f"H3: Adam with 20% label noise (seed={seed})"
            returncode = run_experiment(cmd, desc)
            if returncode != 0:
                failed_experiments.append(f"H3_adam_noise_seed{seed}")
            results.append(('H3', 'adam_noise', seed, returncode == 0))
            time.sleep(2)
            
            # Lion with 30% label noise
            # Lion requires special hyperparameters:
            # - LR = 0.0001 (1/10 of Adam's default 0.001, following Lion paper)
            # - WD = 0.1 (much higher than Adam, Lion paper uses up to 1.0 on ImageNet)
            # - Betas = (0.9, 0.99) - set in optimizers.py default, needs faster momentum decay
            cmd = [
                sys.executable, 'src/train.py',
                '--dataset', 'cifar10',
                '--optimizer', 'lion',
                '--epochs', '50',
                '--seed', str(seed),
                '--batch-size', '256',
                '--model', 'resnet18',
                '--scheduler', 'cosine',
                '--label-noise', '0.3',  # 30% label noise
                '--lr', '0.0001',  # Lion LR = 1/10 of Adam LR (following Lion paper)
                '--weight-decay', '0.01',  # High WD for Lion to stabilize sign updates
                '--augment',  # Use basic augmentation (default in train.py)
                '--save-dir', HYPOTHESIS_RESULTS_DIR,
            ]
            desc = f"H3: Lion with 30% label noise (seed={seed})"
            returncode = run_experiment(cmd, desc)
            if returncode != 0:
                failed_experiments.append(f"H3_lion_noise_seed{seed}")
            results.append(('H3', 'lion_noise', seed, returncode == 0))
            time.sleep(1)
    
    # H4: Muon data efficiency (test with limited training data)
    if args.h4 or args.all:
        print("\n" + "="*60)
        print("H4: Muon Data Efficiency Test")
        print("="*60)
        print("Comparing Muon vs AdamW with 20% training data on CIFAR-100")
        print("Hypothesis: Muon's orthogonalized updates enable better")
        print("            utilization of limited training data")
        print("Expected: Muon will show 5-10% higher accuracy than AdamW")
        print("="*60)
        
        seeds = [42, 123, 456]
        for seed in seeds:
            # AdamW baseline with 20% data
            cmd = [
                sys.executable, 'train_cifar100_optimized.py',
                '--optimizer', 'adamw',
                '--epochs', '50',
                '--seed', str(seed),
                '--batch-size', '128',
                '--model', 'wrn-16-4',
                '--scheduler', 'cosine',
                '--data-fraction', '0.2',  # 20% of training data
                '--save-dir', HYPOTHESIS_RESULTS_DIR,
                '--exp-suffix', '_h4_data20pct'  # Mark as H4 experiment
            ]
            desc = f"H4: AdamW with 20% data (seed={seed})"
            returncode = run_experiment(cmd, desc)
            if returncode != 0:
                failed_experiments.append(f"H4_adamw_limited_seed{seed}")
            results.append(('H4', 'adamw_limited', seed, returncode == 0))
            time.sleep(2)
            
            # Muon with 20% data
            cmd = [
                sys.executable, 'train_cifar100_optimized.py',
                '--optimizer', 'muon',
                '--epochs', '50',
                '--seed', str(seed),
                '--batch-size', '128',
                '--model', 'wrn-16-4',
                '--scheduler', 'cosine',
                '--data-fraction', '0.2',  # 20% of training data
                '--save-dir', HYPOTHESIS_RESULTS_DIR,
                '--exp-suffix', '_h4_data20pct'  # Mark as H4 experiment
            ]
            desc = f"H4: Muon with 20% data (seed={seed})"
            returncode = run_experiment(cmd, desc)
            if returncode != 0:
                failed_experiments.append(f"H4_muon_limited_seed{seed}")
            results.append(('H4', 'muon_limited', seed, returncode == 0))
            time.sleep(2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ALL HYPOTHESIS TESTS COMPLETE")
    print(f"{'='*60}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved in: {HYPOTHESIS_RESULTS_DIR}")
    
    successful = sum(1 for r in results if r[3])
    total = len(results)
    print(f"Successful: {successful}/{total}")
    
    if failed_experiments:
        print(f"\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    else:
        print(f"\n✓ All experiments completed successfully!")
    
    # Group by hypothesis
    print(f"\n{'='*60}")
    print("Results by Hypothesis:")
    print(f"{'='*60}")
    for hypothesis in ['H1', 'H2', 'H3', 'H4']:
        hyp_results = [r for r in results if r[0] == hypothesis]
        if hyp_results:
            success_count = sum(1 for r in hyp_results if r[3])
            print(f"{hypothesis}: {success_count}/{len(hyp_results)} successful")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run hypothesis validation experiments')
    
    parser.add_argument('--h1', action='store_true',
                        help='Run H1: RAdam stability test')
    parser.add_argument('--h2', action='store_true',
                        help='Run H2: AdamW regularization test')
    parser.add_argument('--h3', action='store_true',
                        help='Run H3: Lion robustness test (20% label noise)')
    parser.add_argument('--h4', action='store_true',
                        help='Run H4: Muon data efficiency test (20% training data)')
    parser.add_argument('--all', action='store_true',
                        help='Run all hypothesis tests')
    
    args = parser.parse_args()
    
    if not (args.h1 or args.h2 or args.h3 or args.h4 or args.all):
        print("Please specify which hypothesis to test:")
        print("  --h1: RAdam stability")
        print("  --h2: AdamW regularization")
        print("  --h3: Lion robustness (20% label noise)")
        print("  --h4: Muon data efficiency (20% training data)")
        print("  --all: Run all tests")
        sys.exit(1)
    
    main(args)
