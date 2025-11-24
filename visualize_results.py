#!/usr/bin/env python3
"""
Visualize results from multiple experiments.
Automatically detects all optimizers from result directory names.
Supports two naming formats:
  - CIFAR-10: cifar10_OPTIMIZER_lrxxx_wdxxx_seedxxx
  - CIFAR-100: cifar100_wrn-16-4_OPTIMIZER_seedxxx
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_metrics(result_dir):
    """Load metrics from a result directory."""
    metrics_path = os.path.join(result_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def parse_result_dir_name(dir_name, dataset):
    """
    Parse result directory name to extract optimizer name.
    
    Supports two formats:
    1. CIFAR-10: cifar10_OPTIMIZER_lrxxx_wdxxx_seedxxx
    2. CIFAR-100: cifar100_wrn-16-4_OPTIMIZER_seedxxx
    
    Returns optimizer name if successful, None otherwise.
    """
    if dataset == 'cifar10':
        # Format: cifar10_OPTIMIZER_lrxxx_wdxxx_seedxxx
        # Example: cifar10_adam_lr0.001_wd0.0_seed123
        pattern = r'^cifar10_([a-z0-9\-]+)_lr[\d\.]+_wd[\d\.]+_seed\d+$'
        match = re.match(pattern, dir_name)
        if match:
            return match.group(1)
    
    elif dataset == 'cifar100':
        # Format: cifar100_wrn-16-4_OPTIMIZER_seedxxx
        # Example: cifar100_wrn-16-4_adam_seed123
        pattern = r'^cifar100_wrn-16-4_([a-z0-9\-]+)_seed\d+$'
        match = re.match(pattern, dir_name)
        if match:
            return match.group(1)
        
        # Also support old format: cifar100_OPTIMIZER_lrxxx_wdxxx_seedxxx
        pattern_old = r'^cifar100_([a-z0-9\-]+)_lr[\d\.]+_wd[\d\.]+_seed\d+$'
        match_old = re.match(pattern_old, dir_name)
        if match_old:
            return match_old.group(1)
    
    return None


def collect_experiments(results_root, dataset):
    """
    Collect all experiment results for a dataset.
    Automatically discovers all optimizers from directory names.
    
    Returns:
        dict: {optimizer_name: [list of metrics dicts]}
    """
    experiments = defaultdict(list)
    results_path = Path(results_root)
    
    if not results_path.exists():
        print(f"Warning: Results directory {results_root} does not exist")
        return experiments
    
    # Scan all directories in results_root
    for result_dir in results_path.iterdir():
        if not result_dir.is_dir():
            continue
        
        dir_name = result_dir.name
        
        # Try to parse optimizer name from directory name
        optimizer = parse_result_dir_name(dir_name, dataset)
        
        if optimizer is None:
            # Skip directories that don't match expected patterns
            continue
        
        # Load metrics
        metrics = load_metrics(result_dir)
        if metrics:
            experiments[optimizer].append(metrics)
    
    return dict(experiments)


def plot_comparison(experiments, dataset, save_path=None):
    """Plot comparison of all optimizers."""
    
    if not experiments:
        print(f"No experiments to plot for {dataset}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Optimizer Comparison on {dataset.upper()}', fontsize=14, fontweight='bold')
    
    # Get sorted list of optimizers for consistent ordering
    optimizer_names = sorted(experiments.keys())
    colors = sns.color_palette("husl", len(optimizer_names))
    color_map = {opt: color for opt, color in zip(optimizer_names, colors)}
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for opt_name in optimizer_names:
        runs = experiments[opt_name]
        if not runs:
            continue
        
        # Calculate mean and std across runs
        max_len = max(len(run['train_loss']) for run in runs)
        losses = []
        for run in runs:
            loss = run['train_loss']
            # Pad if necessary
            if len(loss) < max_len:
                loss = loss + [loss[-1]] * (max_len - len(loss))
            losses.append(loss)
        
        losses = np.array(losses)
        mean_loss = np.mean(losses, axis=0)
        std_loss = np.std(losses, axis=0)
        epochs = range(1, len(mean_loss) + 1)
        
        color = color_map[opt_name]
        ax.plot(epochs, mean_loss, label=opt_name.upper(), linewidth=2, color=color)
        ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, 
                        alpha=0.2, color=color)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy
    ax = axes[0, 1]
    for opt_name in optimizer_names:
        runs = experiments[opt_name]
        if not runs:
            continue
        
        max_len = max(len(run['test_acc']) for run in runs)
        accs = []
        for run in runs:
            acc = run['test_acc']
            if len(acc) < max_len:
                acc = acc + [acc[-1]] * (max_len - len(acc))
            accs.append(acc)
        
        accs = np.array(accs)
        mean_acc = np.mean(accs, axis=0)
        std_acc = np.std(accs, axis=0)
        epochs = range(1, len(mean_acc) + 1)
        
        color = color_map[opt_name]
        ax.plot(epochs, mean_acc, label=opt_name.upper(), linewidth=2, color=color)
        ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                        alpha=0.2, color=color)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final Accuracy Bar Chart
    ax = axes[1, 0]
    final_accs = {}
    for opt_name in optimizer_names:
        runs = experiments[opt_name]
        if not runs:
            continue
        accs = [run['test_acc'][-1] for run in runs]
        final_accs[opt_name] = {
            'mean': np.mean(accs),
            'std': np.std(accs)
        }
    
    if final_accs:
        optimizers = list(final_accs.keys())
        means = [final_accs[opt]['mean'] for opt in optimizers]
        stds = [final_accs[opt]['std'] for opt in optimizers]
        
        x_pos = np.arange(len(optimizers))
        colors_list = [color_map[opt] for opt in optimizers]
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors_list)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([opt.upper() for opt in optimizers], rotation=0)
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Final Test Accuracy')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.5, f'{mean:.2f}±{std:.2f}', 
                   ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Best Accuracy Bar Chart
    ax = axes[1, 1]
    best_accs = {}
    for opt_name in optimizer_names:
        runs = experiments[opt_name]
        if not runs:
            continue
        accs = [max(run['test_acc']) for run in runs]
        best_accs[opt_name] = {
            'mean': np.mean(accs),
            'std': np.std(accs)
        }
    
    if best_accs:
        optimizers = list(best_accs.keys())
        means = [best_accs[opt]['mean'] for opt in optimizers]
        stds = [best_accs[opt]['std'] for opt in optimizers]
        
        x_pos = np.arange(len(optimizers))
        colors_list = [color_map[opt] for opt in optimizers]
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors_list)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([opt.upper() for opt in optimizers], rotation=0)
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Best Test Accuracy')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.5, f'{mean:.2f}±{std:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary_table(experiments, dataset):
    """Print a summary table of results."""
    
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY: {dataset.upper()}")
    print(f"{'='*80}\n")
    
    print(f"{'Optimizer':<15} {'#Runs':<8} {'Final Acc (%)':<20} {'Best Acc (%)':<20} {'Converged?':<12}")
    print(f"{'-'*80}")
    
    for opt_name in sorted(experiments.keys()):
        runs = experiments[opt_name]
        if not runs:
            continue
        
        n_runs = len(runs)
        
        final_accs = [run['test_acc'][-1] for run in runs]
        final_mean = np.mean(final_accs)
        final_std = np.std(final_accs)
        
        best_accs = [max(run['test_acc']) for run in runs]
        best_mean = np.mean(best_accs)
        best_std = np.std(best_accs)
        
        # Check convergence (accuracy still improving in last 10 epochs)
        converged = []
        for run in runs:
            if len(run['test_acc']) >= 10:
                last_10 = run['test_acc'][-10:]
                improvement = max(last_10) - min(last_10)
                converged.append(improvement < 1.0)  # Less than 1% change
            else:
                converged.append(False)
        convergence_rate = sum(converged) / len(converged) * 100 if converged else 0
        
        print(f"{opt_name.upper():<15} {n_runs:<8} "
              f"{final_mean:6.2f} ± {final_std:4.2f}{'':6} "
              f"{best_mean:6.2f} ± {best_std:4.2f}{'':6} "
              f"{convergence_rate:5.0f}%")
    
    print(f"{'-'*80}\n")


def main(args):
    """Main visualization function."""
    
    for dataset in args.datasets:
        print(f"\nProcessing {dataset.upper()} results...")
        
        # Collect experiments (automatically discovers all optimizers)
        experiments = collect_experiments(args.results_dir, dataset)
        
        # Filter out empty experiments
        experiments = {k: v for k, v in experiments.items() if v}
        
        if not experiments:
            print(f"No results found for {dataset}")
            continue
        
        # Print discovered optimizers
        optimizers = sorted(experiments.keys())
        print(f"Found {len(optimizers)} optimizers: {', '.join(opt.upper() for opt in optimizers)}")
        
        # Print summary
        print_summary_table(experiments, dataset)
        
        # Plot comparison
        save_path = os.path.join(args.output_dir, f'{dataset}_comparison.png')
        os.makedirs(args.output_dir, exist_ok=True)
        plot_comparison(experiments, dataset, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize optimizer comparison results')
    
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='./plots',
                        help='Directory to save plots')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['cifar10', 'cifar100'],
                        choices=['cifar10', 'cifar100'],
                        help='Datasets to visualize')
    
    args = parser.parse_args()
    
    main(args)
