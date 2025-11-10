#!/usr/bin/env python3
"""
Generate optimizer fingerprints from gradient tracking data.

This script loads gradient statistics from tracked experiments and
creates visualizations showing unique optimizer "fingerprints".
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gradient_flow_analyzer import OptimizerFingerprintAnalyzer


def load_gradient_stats(results_dir, dataset='cifar10'):
    """Load gradient statistics from all tracked experiments."""
    
    optimizer_stats = {}
    
    # Find all experiment directories with gradient tracking
    pattern = f"{dataset}_*_tracked_*"
    
    for result_dir in Path(results_dir).glob(pattern):
        gradient_stats_path = result_dir / 'gradient_stats.pkl'
        
        if not gradient_stats_path.exists():
            continue
        
        # Extract optimizer name from directory
        parts = result_dir.name.split('_')
        if len(parts) >= 2:
            optimizer = parts[1]
            
            # Load gradient stats
            with open(gradient_stats_path, 'rb') as f:
                stats = pickle.load(f)
            
            if optimizer not in optimizer_stats:
                optimizer_stats[optimizer] = []
            
            optimizer_stats[optimizer].append(stats)
    
    # Average stats across seeds for each optimizer
    averaged_stats = {}
    for optimizer, stats_list in optimizer_stats.items():
        if len(stats_list) > 0:
            # For simplicity, just use the first one
            # In a more complete implementation, we would average
            averaged_stats[optimizer] = stats_list[0]
    
    return averaged_stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate optimizer fingerprints from gradient tracking data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate fingerprints for all tracked experiments
  python generate_optimizer_fingerprints.py --dataset cifar10
  
  # Generate only comparison plot
  python generate_optimizer_fingerprints.py --dataset cifar10 --comparison-only
  
  # Generate individual fingerprints
  python generate_optimizer_fingerprints.py --dataset cifar10 --individual
        """
    )
    
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='./plots',
                        help='Directory to save plots')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to analyze')
    parser.add_argument('--comparison-only', action='store_true',
                        help='Only generate comparison plot')
    parser.add_argument('--individual', action='store_true',
                        help='Generate individual fingerprint plots')
    parser.add_argument('--gradient-flow', action='store_true',
                        help='Generate gradient flow analysis plot')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Loading gradient tracking data from {args.dataset.upper()}...")
    print(f"{'='*60}\n")
    
    # Load gradient statistics
    optimizer_stats = load_gradient_stats(args.results_dir, args.dataset)
    
    if not optimizer_stats:
        print(f"❌ No gradient tracking data found for {args.dataset}")
        print("\nTo generate gradient tracking data, run:")
        print(f"  python src/train_with_gradient_tracking.py --optimizer adam --dataset {args.dataset}")
        return
    
    print(f"Found gradient data for {len(optimizer_stats)} optimizer(s):")
    for opt in optimizer_stats.keys():
        print(f"  - {opt.upper()}")
    print()
    
    # Initialize analyzer
    analyzer = OptimizerFingerprintAnalyzer()
    
    # Add all optimizer data
    for optimizer, stats in optimizer_stats.items():
        analyzer.add_optimizer_data(optimizer, stats)
    
    # Generate visualizations
    print("🎨 Generating visualizations...")
    print()
    
    # 1. Comparison plot (side-by-side fingerprints)
    if not args.individual:
        comparison_path = os.path.join(
            args.output_dir,
            f'{args.dataset}_optimizer_fingerprints_comparison.png'
        )
        print(f"📊 Creating fingerprint comparison...")
        analyzer.compare_fingerprints(save_path=comparison_path)
    
    # 2. Individual fingerprints
    if args.individual:
        for optimizer in optimizer_stats.keys():
            fingerprint_path = os.path.join(
                args.output_dir,
                f'{args.dataset}_{optimizer}_fingerprint.png'
            )
            print(f"📊 Creating fingerprint for {optimizer.upper()}...")
            analyzer.create_fingerprint(optimizer, save_path=fingerprint_path)
    
    # 3. Gradient flow analysis (comprehensive plot)
    if args.gradient_flow:
        gradient_flow_path = os.path.join(
            args.output_dir,
            f'{args.dataset}_gradient_flow_analysis.png'
        )
        print(f"📊 Creating gradient flow analysis...")
        analyzer.plot_gradient_flow(save_path=gradient_flow_path)
    
    print(f"\n{'='*60}")
    print("✓ All visualizations generated!")
    print(f"{'='*60}\n")
    print(f"Output directory: {args.output_dir}")
    print("\nGenerated plots:")
    
    for plot_file in Path(args.output_dir).glob(f'{args.dataset}*fingerprint*.png'):
        print(f"  - {plot_file.name}")
    
    if args.gradient_flow:
        print(f"  - {args.dataset}_gradient_flow_analysis.png")
    
    print("\n💡 Tip: Use these plots in your paper to show unique optimizer characteristics!")
    print()


if __name__ == '__main__':
    main()

