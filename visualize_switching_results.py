#!/usr/bin/env python3
"""
Visualize optimizer switching experiment results.

Creates comparison plots showing:
1. Accuracy curves with switching points marked
2. Learning rate schedules
3. Performance comparison table
4. Before/After switching analysis
"""

import os
import json
import yaml
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


def load_experiment(result_dir):
    """Load experiment results including metrics and config."""
    metrics_path = os.path.join(result_dir, 'metrics.json')
    summary_path = os.path.join(result_dir, 'summary.yaml')
    config_path = os.path.join(result_dir, 'config.yaml')
    
    data = {}
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data['metrics'] = json.load(f)
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            data['summary'] = yaml.safe_load(f)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data['config'] = yaml.safe_load(f)
    
    return data


def collect_experiments(results_dir, dataset='cifar10'):
    """Collect all experiments (switching and baseline)."""
    
    switching_exps = defaultdict(list)
    baseline_exps = defaultdict(list)
    
    for result_dir in Path(results_dir).glob(f"{dataset}_*"):
        data = load_experiment(result_dir)
        
        if not data:
            continue
        
        # Check if it's a switching experiment
        if 'switching' in result_dir.name:
            # Extract strategy name
            parts = result_dir.name.split('_')
            if len(parts) >= 3:
                strategy = '_'.join(parts[2:-1])  # Remove dataset and seed
                switching_exps[strategy].append(data)
        else:
            # Baseline experiment
            if 'metrics' in data:
                # Extract optimizer name
                parts = result_dir.name.split('_')
                if len(parts) >= 2:
                    optimizer = parts[1]
                    baseline_exps[optimizer].append(data)
    
    return switching_exps, baseline_exps


def plot_switching_comparison(switching_exps, baseline_exps, dataset, save_dir):
    """Create comprehensive comparison plot."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Optimizer Switching Strategy Comparison on {dataset.upper()}',
        fontsize=16, fontweight='bold'
    )
    
    colors_switching = sns.color_palette("Set2", len(switching_exps))
    colors_baseline = sns.color_palette("Pastel1", len(baseline_exps))
    
    # Plot 1: Test Accuracy Curves
    ax = axes[0, 0]
    
    # Plot switching strategies
    for (strategy, runs), color in zip(switching_exps.items(), colors_switching):
        if not runs:
            continue
        
        # Average across runs
        test_accs = []
        switch_epochs = []
        
        for run in runs:
            if 'test_acc' in run['metrics']:
                test_accs.append(run['metrics']['test_acc'])
            
            # Get switch points
            if 'switch_history' in run.get('summary', {}):
                for switch in run['summary']['switch_history']:
                    switch_epochs.append(switch['epoch'])
        
        if not test_accs:
            continue
        
        # Calculate mean
        max_len = max(len(acc) for acc in test_accs)
        padded = []
        for acc in test_accs:
            if len(acc) < max_len:
                acc = acc + [acc[-1]] * (max_len - len(acc))
            padded.append(acc)
        
        mean_acc = np.mean(padded, axis=0)
        std_acc = np.std(padded, axis=0)
        epochs = range(1, len(mean_acc) + 1)
        
        label = strategy.replace('_', '→').upper()
        ax.plot(epochs, mean_acc, label=label, linewidth=2.5, color=color)
        ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                        alpha=0.2, color=color)
        
        # Mark switching points
        if switch_epochs:
            avg_switch_epoch = int(np.mean(switch_epochs))
            if avg_switch_epoch < len(mean_acc):
                ax.axvline(x=avg_switch_epoch, color=color, linestyle='--',
                          alpha=0.5, linewidth=1.5)
                ax.plot(avg_switch_epoch, mean_acc[avg_switch_epoch-1],
                       'o', color=color, markersize=8, markeredgecolor='white',
                       markeredgewidth=1.5)
    
    # Plot baselines (lighter colors)
    for (optimizer, runs), color in zip(baseline_exps.items(), colors_baseline):
        if not runs:
            continue
        
        test_accs = [run['metrics']['test_acc'] for run in runs 
                     if 'test_acc' in run['metrics']]
        
        if not test_accs:
            continue
        
        max_len = max(len(acc) for acc in test_accs)
        padded = []
        for acc in test_accs:
            if len(acc) < max_len:
                acc = acc + [acc[-1]] * (max_len - len(acc))
            padded.append(acc)
        
        mean_acc = np.mean(padded, axis=0)
        epochs = range(1, len(mean_acc) + 1)
        
        ax.plot(epochs, mean_acc, label=f"{optimizer.upper()} (baseline)",
               linewidth=1.5, linestyle=':', color=color, alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Test Accuracy Evolution', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final Accuracy Comparison
    ax = axes[0, 1]
    
    all_results = []
    all_labels = []
    all_colors = []
    
    # Switching strategies
    for (strategy, runs), color in zip(switching_exps.items(), colors_switching):
        if not runs:
            continue
        best_accs = [run['summary']['best_accuracy'] for run in runs 
                     if 'summary' in run]
        if best_accs:
            all_results.append({
                'mean': np.mean(best_accs),
                'std': np.std(best_accs)
            })
            all_labels.append(strategy.replace('_', '→').upper())
            all_colors.append(color)
    
    # Baselines
    for (optimizer, runs), color in zip(baseline_exps.items(), colors_baseline):
        if not runs:
            continue
        best_accs = [run['summary']['best_accuracy'] for run in runs
                     if 'summary' in run]
        if best_accs:
            all_results.append({
                'mean': np.mean(best_accs),
                'std': np.std(best_accs)
            })
            all_labels.append(f"{optimizer.upper()}\n(baseline)")
            all_colors.append(color)
    
    if all_results:
        x_pos = np.arange(len(all_results))
        means = [r['mean'] for r in all_results]
        stds = [r['std'] for r in all_results]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=all_colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Best Test Accuracy (%)', fontsize=11)
        ax.set_title('Best Accuracy Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.3, f'{mean:.2f}±{std:.2f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot 3: Training Loss Curves
    ax = axes[1, 0]
    
    for (strategy, runs), color in zip(switching_exps.items(), colors_switching):
        if not runs:
            continue
        
        train_losses = [run['metrics']['train_loss'] for run in runs
                       if 'train_loss' in run['metrics']]
        
        if not train_losses:
            continue
        
        max_len = max(len(loss) for loss in train_losses)
        padded = []
        for loss in train_losses:
            if len(loss) < max_len:
                loss = loss + [loss[-1]] * (max_len - len(loss))
            padded.append(loss)
        
        mean_loss = np.mean(padded, axis=0)
        epochs = range(1, len(mean_loss) + 1)
        
        label = strategy.replace('_', '→').upper()
        ax.plot(epochs, mean_loss, label=label, linewidth=2, color=color)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Convergence Speed (Epochs to 80% accuracy)
    ax = axes[1, 1]
    
    convergence_data = []
    convergence_labels = []
    convergence_colors = []
    
    target_acc = 80.0
    
    for (strategy, runs), color in zip(switching_exps.items(), colors_switching):
        if not runs:
            continue
        
        epochs_to_target = []
        for run in runs:
            if 'test_acc' in run['metrics']:
                test_acc = run['metrics']['test_acc']
                for epoch, acc in enumerate(test_acc, 1):
                    if acc >= target_acc:
                        epochs_to_target.append(epoch)
                        break
        
        if epochs_to_target:
            convergence_data.append({
                'mean': np.mean(epochs_to_target),
                'std': np.std(epochs_to_target)
            })
            convergence_labels.append(strategy.replace('_', '→').upper())
            convergence_colors.append(color)
    
    if convergence_data:
        x_pos = np.arange(len(convergence_data))
        means = [d['mean'] for d in convergence_data]
        stds = [d['std'] for d in convergence_data]
        
        ax.barh(x_pos, means, xerr=stds, capsize=5,
               color=convergence_colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(convergence_labels, fontsize=9)
        ax.set_xlabel('Epochs to 80% Accuracy', fontsize=11)
        ax.set_title('Convergence Speed', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(mean + std + 1, i, f'{mean:.1f}±{std:.1f}',
                   va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{dataset}_switching_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path}")
    
    plt.close()


def print_summary_table(switching_exps, baseline_exps, dataset):
    """Print detailed summary table."""
    
    print("\n" + "="*80)
    print(f"OPTIMIZER SWITCHING RESULTS SUMMARY: {dataset.upper()}")
    print("="*80 + "\n")
    
    print("🔄 SWITCHING STRATEGIES")
    print("-" * 80)
    print(f"{'Strategy':<25} {'#Runs':<8} {'Best Acc (%)':<20} {'Final Acc (%)':<20}")
    print("-" * 80)
    
    for strategy, runs in sorted(switching_exps.items()):
        if not runs:
            continue
        
        n_runs = len(runs)
        
        best_accs = [run['summary']['best_accuracy'] for run in runs
                    if 'summary' in run]
        final_accs = [run['summary']['final_accuracy'] for run in runs
                     if 'summary' in run]
        
        if best_accs and final_accs:
            best_mean, best_std = np.mean(best_accs), np.std(best_accs)
            final_mean, final_std = np.mean(final_accs), np.std(final_accs)
            
            display_name = strategy.replace('_', '→').upper()
            print(f"{display_name:<25} {n_runs:<8} "
                  f"{best_mean:6.2f} ± {best_std:4.2f}{'':6} "
                  f"{final_mean:6.2f} ± {final_std:4.2f}")
    
    print("\n" + "📊 BASELINE COMPARISONS")
    print("-" * 80)
    print(f"{'Optimizer':<25} {'#Runs':<8} {'Best Acc (%)':<20} {'Final Acc (%)':<20}")
    print("-" * 80)
    
    for optimizer, runs in sorted(baseline_exps.items()):
        if not runs:
            continue
        
        n_runs = len(runs)
        
        best_accs = [run['summary']['best_accuracy'] for run in runs
                    if 'summary' in run]
        final_accs = [run['summary']['final_accuracy'] for run in runs
                     if 'summary' in run]
        
        if best_accs and final_accs:
            best_mean, best_std = np.mean(best_accs), np.std(best_accs)
            final_mean, final_std = np.mean(final_accs), np.std(final_accs)
            
            print(f"{optimizer.upper():<25} {n_runs:<8} "
                  f"{best_mean:6.2f} ± {best_std:4.2f}{'':6} "
                  f"{final_mean:6.2f} ± {final_std:4.2f}")
    
    print("-" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize optimizer switching experiment results'
    )
    
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='./plots',
                        help='Directory to save plots')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to visualize')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Analyzing {args.dataset.upper()} results...")
    print(f"{'='*60}\n")
    
    # Collect experiments
    switching_exps, baseline_exps = collect_experiments(
        args.results_dir, args.dataset
    )
    
    if not switching_exps and not baseline_exps:
        print(f"❌ No results found for {args.dataset}")
        return
    
    print(f"Found {len(switching_exps)} switching strategies")
    print(f"Found {len(baseline_exps)} baseline optimizers")
    
    # Print summary table
    print_summary_table(switching_exps, baseline_exps, args.dataset)
    
    # Create visualization
    if switching_exps or baseline_exps:
        plot_switching_comparison(
            switching_exps, baseline_exps, args.dataset, args.output_dir
        )


if __name__ == '__main__':
    main()

