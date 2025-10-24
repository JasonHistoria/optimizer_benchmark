#!/usr/bin/env python3
"""
Visualize results from multiple experiments.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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


def collect_experiments(results_root, dataset, optimizers):
    """Collect all experiment results for a dataset."""
    experiments = {}
    
    for optimizer in optimizers:
        experiments[optimizer] = []
        
        # Find all experiments for this optimizer
        pattern = f"{dataset}_{optimizer}_*"
        for result_dir in Path(results_root).glob(pattern):
            metrics = load_metrics(result_dir)
            if metrics:
                experiments[optimizer].append(metrics)
    
    return experiments


def plot_comparison(experiments, dataset, save_path=None):
    """Plot comparison of all optimizers."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Optimizer Comparison on {dataset.upper()}', fontsize=14, fontweight='bold')
    
    colors = sns.color_palette("husl", len(experiments))
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for (opt_name, runs), color in zip(experiments.items(), colors):
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
    for (opt_name, runs), color in zip(experiments.items(), colors):
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
    for opt_name, runs in experiments.items():
        if not runs:
            continue
        accs = [run['test_acc'][-1] for run in runs]
        final_accs[opt_name] = {
            'mean': np.mean(accs),
            'std': np.std(accs)
        }
    
    optimizers = list(final_accs.keys())
    means = [final_accs[opt]['mean'] for opt in optimizers]
    stds = [final_accs[opt]['std'] for opt in optimizers]
    
    x_pos = np.arange(len(optimizers))
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors[:len(optimizers)])
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
    for opt_name, runs in experiments.items():
        if not runs:
            continue
        accs = [max(run['test_acc']) for run in runs]
        best_accs[opt_name] = {
            'mean': np.mean(accs),
            'std': np.std(accs)
        }
    
    means = [best_accs[opt]['mean'] for opt in optimizers]
    stds = [best_accs[opt]['std'] for opt in optimizers]
    
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors[:len(optimizers)])
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
    
    print(f"{'Optimizer':<12} {'#Runs':<8} {'Final Acc (%)':<20} {'Best Acc (%)':<20} {'Converged?':<12}")
    print(f"{'-'*80}")
    
    for opt_name, runs in sorted(experiments.items()):
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
            last_10 = run['test_acc'][-10:]
            improvement = max(last_10) - min(last_10)
            converged.append(improvement < 1.0)  # Less than 1% change
        convergence_rate = sum(converged) / len(converged) * 100
        
        print(f"{opt_name.upper():<12} {n_runs:<8} "
              f"{final_mean:6.2f} ± {final_std:4.2f}{'':6} "
              f"{best_mean:6.2f} ± {best_std:4.2f}{'':6} "
              f"{convergence_rate:5.0f}%")
    
    print(f"{'-'*80}\n")


def main(args):
    """Main visualization function."""
    
    optimizers = ['sgd', 'adam', 'adamw', 'radam', 'lion']
    
    for dataset in args.datasets:
        print(f"\nProcessing {dataset.upper()} results...")
        
        # Collect experiments
        experiments = collect_experiments(args.results_dir, dataset, optimizers)
        
        # Filter out empty experiments
        experiments = {k: v for k, v in experiments.items() if v}
        
        if not experiments:
            print(f"No results found for {dataset}")
            continue
        
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

