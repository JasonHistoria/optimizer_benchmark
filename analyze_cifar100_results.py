#!/usr/bin/env python3
"""
Comprehensive analysis of CIFAR-100 optimizer comparison results.
Includes analysis of both original RAdam and fixed RAdam-v2.

This script generates:
1. Summary statistics table
2. Training/validation curves
3. Accuracy comparison charts
4. Convergence analysis
5. Overfitting analysis
6. Statistical significance tests
7. RAdam vs RAdam-v2 comparison
"""

import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

# Color palette
COLORS = {
    'sgd': '#1f77b4',
    'adam': '#ff7f0e',
    'adamw': '#2ca02c',
    'radam': '#d62728',      # Original RAdam (red)
    'radam-v2': '#9467bd',   # Fixed RAdam (purple)
    'lion': '#8c564b'
}


def load_summary(result_dir):
    """Load summary.yaml from a result directory."""
    summary_path = os.path.join(result_dir, 'summary.yaml')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return yaml.safe_load(f)
    return None


def load_metrics(result_dir):
    """Load metrics.json from a result directory."""
    metrics_path = os.path.join(result_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def collect_cifar100_experiments(results_root):
    """Collect all CIFAR-100 WRN-16-4 experiment results."""
    results_root = Path(results_root)
    experiments = defaultdict(list)
    
    # Pattern for CIFAR-100 WRN-16-4 experiments
    pattern = "cifar100_wrn-16-4_*"
    
    for exp_dir in results_root.glob(pattern):
        exp_name = exp_dir.name
        
        # Parse experiment name
        # Format: cifar100_wrn-16-4_{optimizer}_seed{seed}
        # or: cifar100_wrn-16-4_{optimizer}-v2_seed{seed}
        parts = exp_name.split('_')
        if len(parts) >= 4:
            optimizer = parts[2]
            seed_part = parts[3]
            
            # Extract seed
            if seed_part.startswith('seed'):
                seed = int(seed_part[4:])
            else:
                continue
            
            # Check if it's radam-v2
            if optimizer == 'radam-v2':
                optimizer_key = 'radam-v2'
            elif optimizer == 'radam':
                optimizer_key = 'radam'  # Original RAdam
            else:
                optimizer_key = optimizer
            
            summary = load_summary(exp_dir)
            metrics = load_metrics(exp_dir)
            
            if summary and metrics:
                experiments[optimizer_key].append({
                    'seed': seed,
                    'summary': summary,
                    'metrics': metrics,
                    'exp_name': exp_name,
                    'exp_dir': str(exp_dir)
                })
    
    return experiments


def calculate_statistics(experiments):
    """Calculate comprehensive statistics for each optimizer."""
    stats_dict = {}
    
    for opt_name, runs in experiments.items():
        if not runs:
            continue
        
        # Extract metrics
        best_accs = [r['summary']['best_accuracy'] for r in runs]
        final_accs = [r['summary']['final_accuracy'] for r in runs]
        total_times = [r['summary']['total_time'] for r in runs]
        best_epochs = [r['summary'].get('best_epoch', 100) for r in runs]
        
        # Calculate train-test gaps
        train_test_gaps = []
        convergence_epochs = []
        
        for run in runs:
            metrics = run['metrics']
            if 'train_acc' in metrics and 'test_acc' in metrics:
                train_acc = metrics['train_acc']
                test_acc = metrics['test_acc']
                
                # Final train-test gap
                if len(train_acc) > 0 and len(test_acc) > 0:
                    gap = train_acc[-1] - test_acc[-1]
                    train_test_gaps.append(gap)
                
                # Convergence: epoch where test acc reaches 95% of final
                if len(test_acc) > 0:
                    final_test_acc = test_acc[-1]
                    target_acc = final_test_acc * 0.95
                    conv_epoch = next((i+1 for i, acc in enumerate(test_acc) if acc >= target_acc), len(test_acc))
                    convergence_epochs.append(conv_epoch)
        
        # Calculate statistics
        stats_dict[opt_name] = {
            'n_runs': len(runs),
            'best_acc_mean': np.mean(best_accs),
            'best_acc_std': np.std(best_accs),
            'best_acc_min': np.min(best_accs),
            'best_acc_max': np.max(best_accs),
            'final_acc_mean': np.mean(final_accs),
            'final_acc_std': np.std(final_accs),
            'total_time_mean': np.mean(total_times) / 3600,  # Convert to hours
            'total_time_std': np.std(total_times) / 3600,
            'best_epoch_mean': np.mean(best_epochs),
            'best_epoch_std': np.std(best_epochs),
            'train_test_gap_mean': np.mean(train_test_gaps) if train_test_gaps else None,
            'train_test_gap_std': np.std(train_test_gaps) if train_test_gaps else None,
            'convergence_epoch_mean': np.mean(convergence_epochs) if convergence_epochs else None,
            'convergence_epoch_std': np.std(convergence_epochs) if convergence_epochs else None,
        }
    
    return stats_dict


def print_summary_table(stats_dict):
    """Print a comprehensive summary table."""
    print("\n" + "="*100)
    print("CIFAR-100 OPTIMIZER COMPARISON SUMMARY")
    print("="*100)
    
    # Create DataFrame for better formatting
    data = []
    for opt_name, stats in sorted(stats_dict.items()):
        data.append({
            'Optimizer': opt_name.upper(),
            'N': stats['n_runs'],
            'Best Acc (%)': f"{stats['best_acc_mean']:.2f} ± {stats['best_acc_std']:.2f}",
            'Final Acc (%)': f"{stats['final_acc_mean']:.2f} ± {stats['final_acc_std']:.2f}",
            'Train-Test Gap': f"{stats['train_test_gap_mean']:.2f} ± {stats['train_test_gap_std']:.2f}" if stats['train_test_gap_mean'] else "N/A",
            'Best Epoch': f"{stats['best_epoch_mean']:.1f} ± {stats['best_epoch_std']:.1f}",
            'Time (hrs)': f"{stats['total_time_mean']:.2f} ± {stats['total_time_std']:.2f}"
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print("="*100 + "\n")


def statistical_significance_test(experiments):
    """Perform statistical significance tests between optimizers."""
    print("\n" + "="*100)
    print("STATISTICAL SIGNIFICANCE TESTS (t-test on best accuracy)")
    print("="*100)
    
    # Extract best accuracies for each optimizer
    acc_data = {}
    for opt_name, runs in experiments.items():
        if runs:
            acc_data[opt_name] = [r['summary']['best_accuracy'] for r in runs]
    
    # Perform pairwise t-tests
    optimizers = sorted(acc_data.keys())
    results = []
    
    for i, opt1 in enumerate(optimizers):
        for opt2 in optimizers[i+1:]:
            if len(acc_data[opt1]) >= 2 and len(acc_data[opt2]) >= 2:
                t_stat, p_value = stats.ttest_ind(acc_data[opt1], acc_data[opt2])
                results.append({
                    'Optimizer 1': opt1.upper(),
                    'Optimizer 2': opt2.upper(),
                    'Mean 1': f"{np.mean(acc_data[opt1]):.2f}",
                    'Mean 2': f"{np.mean(acc_data[opt2]):.2f}",
                    'p-value': f"{p_value:.4f}",
                    'Significant': "Yes" if p_value < 0.05 else "No"
                })
    
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
    print("="*100 + "\n")


def plot_comprehensive_analysis(experiments, stats_dict, output_dir):
    """Generate comprehensive visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    optimizers = sorted(experiments.keys())
    
    # 1. Training Loss Curves
    ax1 = fig.add_subplot(gs[0, 0])
    for opt_name in optimizers:
        if opt_name not in experiments or not experiments[opt_name]:
            continue
        
        runs = experiments[opt_name]
        color = COLORS.get(opt_name, '#000000')
        
        # Average across runs
        max_len = max(len(run['metrics']['train_loss']) for run in runs)
        losses = []
        for run in runs:
            loss = run['metrics']['train_loss']
            if len(loss) < max_len:
                loss = loss + [loss[-1]] * (max_len - len(loss))
            losses.append(loss)
        
        losses = np.array(losses)
        mean_loss = np.mean(losses, axis=0)
        std_loss = np.std(losses, axis=0)
        epochs = range(1, len(mean_loss) + 1)
        
        label = opt_name.upper().replace('-', ' ')
        ax1.plot(epochs, mean_loss, label=label, linewidth=2, color=color)
        ax1.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                         alpha=0.2, color=color)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Test Accuracy Curves
    ax2 = fig.add_subplot(gs[0, 1])
    for opt_name in optimizers:
        if opt_name not in experiments or not experiments[opt_name]:
            continue
        
        runs = experiments[opt_name]
        color = COLORS.get(opt_name, '#000000')
        
        max_len = max(len(run['metrics']['test_acc']) for run in runs)
        accs = []
        for run in runs:
            acc = run['metrics']['test_acc']
            if len(acc) < max_len:
                acc = acc + [acc[-1]] * (max_len - len(acc))
            accs.append(acc)
        
        accs = np.array(accs)
        mean_acc = np.mean(accs, axis=0)
        std_acc = np.std(accs, axis=0)
        epochs = range(1, len(mean_acc) + 1)
        
        label = opt_name.upper().replace('-', ' ')
        ax2.plot(epochs, mean_acc, label=label, linewidth=2, color=color)
        ax2.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                         alpha=0.2, color=color)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy Curves', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Best Accuracy Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    opt_names = []
    means = []
    stds = []
    colors_list = []
    
    for opt_name in optimizers:
        if opt_name in stats_dict:
            opt_names.append(opt_name.upper().replace('-', ' '))
            means.append(stats_dict[opt_name]['best_acc_mean'])
            stds.append(stats_dict[opt_name]['best_acc_std'])
            colors_list.append(COLORS.get(opt_name, '#000000'))
    
    x_pos = np.arange(len(opt_names))
    bars = ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors_list)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(opt_names, rotation=45, ha='right')
    ax3.set_ylabel('Best Accuracy (%)')
    ax3.set_title('Best Test Accuracy Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax3.text(i, mean + std + 1, f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=8)
    
    # 4. Final Accuracy Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    opt_names = []
    means = []
    stds = []
    colors_list = []
    
    for opt_name in optimizers:
        if opt_name in stats_dict:
            opt_names.append(opt_name.upper().replace('-', ' '))
            means.append(stats_dict[opt_name]['final_acc_mean'])
            stds.append(stats_dict[opt_name]['final_acc_std'])
            colors_list.append(COLORS.get(opt_name, '#000000'))
    
    x_pos = np.arange(len(opt_names))
    bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors_list)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(opt_names, rotation=45, ha='right')
    ax4.set_ylabel('Final Accuracy (%)')
    ax4.set_title('Final Test Accuracy Comparison', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax4.text(i, mean + std + 1, f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=8)
    
    # 5. Train-Test Gap (Overfitting Analysis)
    ax5 = fig.add_subplot(gs[1, 1])
    opt_names = []
    means = []
    stds = []
    colors_list = []
    
    for opt_name in optimizers:
        if opt_name in stats_dict and stats_dict[opt_name]['train_test_gap_mean'] is not None:
            opt_names.append(opt_name.upper().replace('-', ' '))
            means.append(stats_dict[opt_name]['train_test_gap_mean'])
            stds.append(stats_dict[opt_name]['train_test_gap_std'])
            colors_list.append(COLORS.get(opt_name, '#000000'))
    
    if opt_names:
        x_pos = np.arange(len(opt_names))
        bars = ax5.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors_list)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(opt_names, rotation=45, ha='right')
        ax5.set_ylabel('Train-Test Gap (%)')
        ax5.set_title('Overfitting Analysis (Train-Test Gap)', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax5.text(i, mean + std + 0.5, f'{mean:.2f}±{std:.2f}',
                    ha='center', va='bottom', fontsize=8)
    
    # 6. Convergence Speed
    ax6 = fig.add_subplot(gs[1, 2])
    opt_names = []
    means = []
    stds = []
    colors_list = []
    
    for opt_name in optimizers:
        if opt_name in stats_dict and stats_dict[opt_name]['convergence_epoch_mean'] is not None:
            opt_names.append(opt_name.upper().replace('-', ' '))
            means.append(stats_dict[opt_name]['convergence_epoch_mean'])
            stds.append(stats_dict[opt_name]['convergence_epoch_std'])
            colors_list.append(COLORS.get(opt_name, '#000000'))
    
    if opt_names:
        x_pos = np.arange(len(opt_names))
        bars = ax6.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors_list)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(opt_names, rotation=45, ha='right')
        ax6.set_ylabel('Convergence Epoch')
        ax6.set_title('Convergence Speed (95% of Final Accuracy)', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax6.text(i, mean + std + 2, f'{mean:.1f}±{std:.1f}',
                    ha='center', va='bottom', fontsize=8)
    
    # 7. RAdam vs RAdam-v2 Direct Comparison
    ax7 = fig.add_subplot(gs[2, :])
    if 'radam' in experiments and 'radam-v2' in experiments:
        radam_runs = experiments['radam']
        radam_v2_runs = experiments['radam-v2']
        
        # Plot individual runs
        for run in radam_runs:
            acc = run['metrics']['test_acc']
            epochs = range(1, len(acc) + 1)
            ax7.plot(epochs, acc, color=COLORS['radam'], alpha=0.3, linewidth=1)
        
        for run in radam_v2_runs:
            acc = run['metrics']['test_acc']
            epochs = range(1, len(acc) + 1)
            ax7.plot(epochs, acc, color=COLORS['radam-v2'], alpha=0.3, linewidth=1)
        
        # Plot means
        max_len = max(
            max(len(run['metrics']['test_acc']) for run in radam_runs),
            max(len(run['metrics']['test_acc']) for run in radam_v2_runs)
        )
        
        for opt_name, runs in [('radam', radam_runs), ('radam-v2', radam_v2_runs)]:
            accs = []
            for run in runs:
                acc = run['metrics']['test_acc']
                if len(acc) < max_len:
                    acc = acc + [acc[-1]] * (max_len - len(acc))
                accs.append(acc)
            
            accs = np.array(accs)
            mean_acc = np.mean(accs, axis=0)
            std_acc = np.std(accs, axis=0)
            epochs = range(1, len(mean_acc) + 1)
            
            label = opt_name.upper().replace('-', ' ')
            ax7.plot(epochs, mean_acc, label=label, linewidth=3, color=COLORS[opt_name])
            ax7.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                            alpha=0.2, color=COLORS[opt_name])
        
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Test Accuracy (%)')
        ax7.set_title('RAdam vs RAdam-v2: Direct Comparison', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3)
    
    # 8. Accuracy Distribution (Box Plot)
    ax8 = fig.add_subplot(gs[3, 0])
    data_for_box = []
    labels_for_box = []
    
    for opt_name in optimizers:
        if opt_name in experiments and experiments[opt_name]:
            accs = [r['summary']['best_accuracy'] for r in experiments[opt_name]]
            data_for_box.append(accs)
            labels_for_box.append(opt_name.upper().replace('-', ' '))
    
    if data_for_box:
        bp = ax8.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch, opt_name in zip(bp['boxes'], optimizers):
            if opt_name in COLORS:
                patch.set_facecolor(COLORS[opt_name])
                patch.set_alpha(0.7)
        
        ax8.set_ylabel('Best Accuracy (%)')
        ax8.set_title('Accuracy Distribution (Box Plot)', fontsize=12, fontweight='bold')
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Training Time Comparison
    ax9 = fig.add_subplot(gs[3, 1])
    opt_names = []
    means = []
    stds = []
    colors_list = []
    
    for opt_name in optimizers:
        if opt_name in stats_dict:
            opt_names.append(opt_name.upper().replace('-', ' '))
            means.append(stats_dict[opt_name]['total_time_mean'])
            stds.append(stats_dict[opt_name]['total_time_std'])
            colors_list.append(COLORS.get(opt_name, '#000000'))
    
    x_pos = np.arange(len(opt_names))
    bars = ax9.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors_list)
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(opt_names, rotation=45, ha='right')
    ax9.set_ylabel('Training Time (hours)')
    ax9.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax9.text(i, mean + std + 0.05, f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=8)
    
    # 10. Early Training Stability (First 20 epochs)
    ax10 = fig.add_subplot(gs[3, 2])
    for opt_name in optimizers:
        if opt_name not in experiments or not experiments[opt_name]:
            continue
        
        runs = experiments[opt_name]
        color = COLORS.get(opt_name, '#000000')
        
        # Get first 20 epochs
        accs = []
        for run in runs:
            acc = run['metrics']['test_acc'][:20]
            if len(acc) < 20:
                acc = acc + [acc[-1]] * (20 - len(acc))
            accs.append(acc)
        
        accs = np.array(accs)
        mean_acc = np.mean(accs, axis=0)
        std_acc = np.std(accs, axis=0)
        epochs = range(1, 21)
        
        label = opt_name.upper().replace('-', ' ')
        ax10.plot(epochs, mean_acc, label=label, linewidth=2, color=color)
        ax10.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                          alpha=0.2, color=color)
    
    ax10.set_xlabel('Epoch')
    ax10.set_ylabel('Test Accuracy (%)')
    ax10.set_title('Early Training Stability (First 20 Epochs)', fontsize=12, fontweight='bold')
    ax10.legend(loc='lower right', fontsize=8)
    ax10.grid(True, alpha=0.3)
    
    plt.suptitle('CIFAR-100 Optimizer Comparison: Comprehensive Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    save_path = os.path.join(output_dir, 'cifar100_comprehensive_analysis.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved comprehensive analysis plot: {save_path}")
    plt.close()


def plot_radam_comparison(experiments, output_dir):
    """Generate detailed RAdam vs RAdam-v2 comparison plot."""
    if 'radam' not in experiments or 'radam-v2' not in experiments:
        print("⚠ RAdam or RAdam-v2 results not found. Skipping RAdam comparison plot.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    radam_runs = experiments['radam']
    radam_v2_runs = experiments['radam-v2']
    
    # 1. Test Accuracy Curves
    ax = axes[0, 0]
    max_len = max(
        max(len(run['metrics']['test_acc']) for run in radam_runs),
        max(len(run['metrics']['test_acc']) for run in radam_v2_runs)
    )
    
    for opt_name, runs in [('radam', radam_runs), ('radam-v2', radam_v2_runs)]:
        accs = []
        for run in runs:
            acc = run['metrics']['test_acc']
            if len(acc) < max_len:
                acc = acc + [acc[-1]] * (max_len - len(acc))
            accs.append(acc)
        
        accs = np.array(accs)
        mean_acc = np.mean(accs, axis=0)
        std_acc = np.std(accs, axis=0)
        epochs = range(1, len(mean_acc) + 1)
        
        label = opt_name.upper().replace('-', ' ')
        ax.plot(epochs, mean_acc, label=label, linewidth=3, color=COLORS[opt_name])
        ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                        alpha=0.3, color=COLORS[opt_name])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy: RAdam vs RAdam-v2', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. Best Accuracy Comparison
    ax = axes[0, 1]
    radam_best = [r['summary']['best_accuracy'] for r in radam_runs]
    radam_v2_best = [r['summary']['best_accuracy'] for r in radam_v2_runs]
    
    data = [radam_best, radam_v2_best]
    bp = ax.boxplot(data, labels=['RAdam\n(Original)', 'RAdam-v2\n(Fixed)'], patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['radam'])
    bp['boxes'][1].set_facecolor(COLORS['radam-v2'])
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_alpha(0.7)
    
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_title('Best Accuracy Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean values
    ax.text(1, np.mean(radam_best), f'Mean: {np.mean(radam_best):.2f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(2, np.mean(radam_v2_best), f'Mean: {np.mean(radam_v2_best):.2f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Training Loss
    ax = axes[1, 0]
    max_len = max(
        max(len(run['metrics']['train_loss']) for run in radam_runs),
        max(len(run['metrics']['train_loss']) for run in radam_v2_runs)
    )
    
    for opt_name, runs in [('radam', radam_runs), ('radam-v2', radam_v2_runs)]:
        losses = []
        for run in runs:
            loss = run['metrics']['train_loss']
            if len(loss) < max_len:
                loss = loss + [loss[-1]] * (max_len - len(loss))
            losses.append(loss)
        
        losses = np.array(losses)
        mean_loss = np.mean(losses, axis=0)
        std_loss = np.std(losses, axis=0)
        epochs = range(1, len(mean_loss) + 1)
        
        label = opt_name.upper().replace('-', ' ')
        ax.plot(epochs, mean_loss, label=label, linewidth=3, color=COLORS[opt_name])
        ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                        alpha=0.3, color=COLORS[opt_name])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss: RAdam vs RAdam-v2', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 4. Improvement Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate improvement
    radam_mean = np.mean(radam_best)
    radam_v2_mean = np.mean(radam_v2_best)
    improvement = radam_v2_mean - radam_mean
    improvement_pct = (improvement / radam_mean) * 100 if radam_mean > 0 else 0
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(radam_best, radam_v2_best)
    
    stats_text = f"""
RAdam vs RAdam-v2 Comparison

Original RAdam:
  Mean Best Accuracy: {radam_mean:.2f}%
  Std: {np.std(radam_best):.2f}%
  Min: {np.min(radam_best):.2f}%
  Max: {np.max(radam_best):.2f}%

Fixed RAdam-v2:
  Mean Best Accuracy: {radam_v2_mean:.2f}%
  Std: {np.std(radam_v2_best):.2f}%
  Min: {np.min(radam_v2_best):.2f}%
  Max: {np.max(radam_v2_best):.2f}%

Improvement:
  Absolute: {improvement:.2f}%
  Relative: {improvement_pct:.1f}%

Statistical Test:
  t-statistic: {t_stat:.4f}
  p-value: {p_value:.4f}
  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)
"""
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.suptitle('RAdam vs RAdam-v2: Detailed Comparison', 
                 fontsize=14, fontweight='bold')
    
    save_path = os.path.join(output_dir, 'cifar100_radam_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved RAdam comparison plot: {save_path}")
    plt.close()


def save_detailed_report(stats_dict, experiments, output_dir):
    """Save a detailed text report."""
    report_path = os.path.join(output_dir, 'cifar100_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("CIFAR-100 OPTIMIZER COMPARISON - DETAILED ANALYSIS REPORT\n")
        f.write("="*100 + "\n\n")
        
        f.write("Generated: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*100 + "\n")
        for opt_name, stats in sorted(stats_dict.items()):
            f.write(f"\n{opt_name.upper()}:\n")
            f.write(f"  Number of runs: {stats['n_runs']}\n")
            f.write(f"  Best Accuracy: {stats['best_acc_mean']:.2f}% ± {stats['best_acc_std']:.2f}%\n")
            f.write(f"    Range: [{stats['best_acc_min']:.2f}%, {stats['best_acc_max']:.2f}%]\n")
            f.write(f"  Final Accuracy: {stats['final_acc_mean']:.2f}% ± {stats['final_acc_std']:.2f}%\n")
            if stats['train_test_gap_mean']:
                f.write(f"  Train-Test Gap: {stats['train_test_gap_mean']:.2f}% ± {stats['train_test_gap_std']:.2f}%\n")
            f.write(f"  Best Epoch: {stats['best_epoch_mean']:.1f} ± {stats['best_epoch_std']:.1f}\n")
            if stats['convergence_epoch_mean']:
                f.write(f"  Convergence Epoch: {stats['convergence_epoch_mean']:.1f} ± {stats['convergence_epoch_std']:.1f}\n")
            f.write(f"  Training Time: {stats['total_time_mean']:.2f} ± {stats['total_time_std']:.2f} hours\n")
        
        # RAdam comparison
        if 'radam' in stats_dict and 'radam-v2' in stats_dict:
            f.write("\n" + "="*100 + "\n")
            f.write("RADAM vs RADAM-V2 COMPARISON\n")
            f.write("="*100 + "\n")
            
            radam_stats = stats_dict['radam']
            radam_v2_stats = stats_dict['radam-v2']
            
            improvement = radam_v2_stats['best_acc_mean'] - radam_stats['best_acc_mean']
            improvement_pct = (improvement / radam_stats['best_acc_mean']) * 100 if radam_stats['best_acc_mean'] > 0 else 0
            
            f.write(f"\nOriginal RAdam Best Accuracy: {radam_stats['best_acc_mean']:.2f}% ± {radam_stats['best_acc_std']:.2f}%\n")
            f.write(f"Fixed RAdam-v2 Best Accuracy: {radam_v2_stats['best_acc_mean']:.2f}% ± {radam_v2_stats['best_acc_std']:.2f}%\n")
            f.write(f"\nImprovement: {improvement:.2f}% ({improvement_pct:.1f}% relative improvement)\n")
            
            # Statistical test
            radam_accs = [r['summary']['best_accuracy'] for r in experiments['radam']]
            radam_v2_accs = [r['summary']['best_accuracy'] for r in experiments['radam-v2']]
            t_stat, p_value = stats.ttest_ind(radam_accs, radam_v2_accs)
            f.write(f"\nStatistical Test (t-test):\n")
            f.write(f"  t-statistic: {t_stat:.4f}\n")
            f.write(f"  p-value: {p_value:.4f}\n")
            f.write(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*100 + "\n")
    
    print(f"✓ Saved detailed report: {report_path}")


def main(args):
    """Main analysis function."""
    print("\n" + "="*100)
    print("CIFAR-100 OPTIMIZER COMPARISON ANALYSIS")
    print("="*100)
    
    # Collect experiments
    print("\nCollecting experiment results...")
    experiments = collect_cifar100_experiments(args.results_dir)
    
    if not experiments:
        print("❌ No CIFAR-100 experiments found!")
        return
    
    print(f"✓ Found experiments for {len(experiments)} optimizers:")
    for opt_name, runs in sorted(experiments.items()):
        print(f"  - {opt_name.upper()}: {len(runs)} runs")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats_dict = calculate_statistics(experiments)
    
    # Print summary table
    print_summary_table(stats_dict)
    
    # Statistical significance tests
    statistical_significance_test(experiments)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_comprehensive_analysis(experiments, stats_dict, args.output_dir)
    plot_radam_comparison(experiments, args.output_dir)
    
    # Save detailed report
    print("\nGenerating detailed report...")
    save_detailed_report(stats_dict, experiments, args.output_dir)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)
    print(f"\nResults saved to: {args.output_dir}")
    print("  - cifar100_comprehensive_analysis.png")
    print("  - cifar100_radam_comparison.png")
    print("  - cifar100_analysis_report.txt")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive CIFAR-100 optimizer analysis')
    
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='./analysis',
                        help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    main(args)


