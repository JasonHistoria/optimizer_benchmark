"""
Gradient Flow Analyzer

This module tracks and analyzes gradient flow through neural networks,
providing insights into how different optimizers affect training dynamics.

Key Features:
1. Layer-wise gradient tracking
2. Gradient norm evolution
3. Optimizer "fingerprint" visualization
4. Update direction analysis
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple


class GradientFlowTracker:
    """
    Track gradient statistics during training.
    
    Records:
    - Gradient norms per layer
    - Parameter update magnitudes
    - Gradient direction changes
    - Layer activation patterns
    """
    
    def __init__(self, model: nn.Module, track_layers: Optional[List[str]] = None):
        """
        Initialize gradient flow tracker.
        
        Args:
            model: PyTorch model to track
            track_layers: List of layer names to track (None = all layers)
        """
        self.model = model
        self.track_layers = track_layers
        
        # Statistics storage
        self.gradient_norms = defaultdict(list)  # {layer_name: [norms]}
        self.parameter_norms = defaultdict(list)  # {layer_name: [norms]}
        self.update_ratios = defaultdict(list)    # update_size / param_size
        self.gradient_angles = defaultdict(list)  # angle between consecutive gradients
        
        # Previous gradients for angle calculation
        self.prev_grads = {}
        
        # Layer metadata
        self.layer_info = {}
        self._initialize_layer_info()
    
    def _initialize_layer_info(self):
        """Extract layer information from model."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                # Count parameters
                n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if n_params > 0:
                    self.layer_info[name] = {
                        'type': module.__class__.__name__,
                        'n_params': n_params
                    }
    
    def _should_track_layer(self, name: str) -> bool:
        """Determine if a layer should be tracked."""
        if self.track_layers is None:
            return True
        return any(pattern in name for pattern in self.track_layers)
    
    def track_gradients(self, epoch: int):
        """
        Record gradient statistics for current step.
        
        Call this after backward() but before optimizer.step().
        """
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            
            # Extract layer name (remove .weight, .bias suffixes)
            layer_name = '.'.join(name.split('.')[:-1]) if '.' in name else name
            
            if not self._should_track_layer(layer_name):
                continue
            
            grad = param.grad
            
            # Gradient norm
            grad_norm = grad.norm().item()
            self.gradient_norms[layer_name].append(grad_norm)
            
            # Parameter norm
            param_norm = param.data.norm().item()
            self.parameter_norms[layer_name].append(param_norm)
            
            # Update ratio (will be computed after optimizer step)
            # For now, just record gradient info
            
            # Gradient angle (cosine similarity with previous gradient)
            if layer_name in self.prev_grads:
                prev_grad = self.prev_grads[layer_name]
                
                # Flatten for comparison
                grad_flat = grad.flatten()
                prev_flat = prev_grad.flatten()
                
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    grad_flat.unsqueeze(0),
                    prev_flat.unsqueeze(0)
                ).item()
                
                # Convert to angle (radians)
                angle = np.arccos(np.clip(cos_sim, -1.0, 1.0))
                self.gradient_angles[layer_name].append(np.degrees(angle))
            
            # Store current gradient for next comparison
            self.prev_grads[layer_name] = grad.detach().clone()
    
    def track_updates(self, epoch: int, old_params: Dict[str, torch.Tensor]):
        """
        Track parameter updates after optimizer step.
        
        Args:
            epoch: Current epoch
            old_params: Parameter values before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            layer_name = '.'.join(name.split('.')[:-1]) if '.' in name else name
            
            if not self._should_track_layer(layer_name):
                continue
            
            if name in old_params:
                # Compute update magnitude
                update = param.data - old_params[name]
                update_norm = update.norm().item()
                param_norm = param.data.norm().item()
                
                # Update ratio
                ratio = update_norm / (param_norm + 1e-8)
                self.update_ratios[layer_name].append(ratio)
    
    def get_statistics(self) -> Dict:
        """Get all tracked statistics."""
        return {
            'gradient_norms': dict(self.gradient_norms),
            'parameter_norms': dict(self.parameter_norms),
            'update_ratios': dict(self.update_ratios),
            'gradient_angles': dict(self.gradient_angles),
            'layer_info': self.layer_info
        }
    
    def clear(self):
        """Clear all tracked data."""
        self.gradient_norms.clear()
        self.parameter_norms.clear()
        self.update_ratios.clear()
        self.gradient_angles.clear()
        self.prev_grads.clear()


class OptimizerFingerprintAnalyzer:
    """
    Analyze and visualize optimizer "fingerprints" - unique patterns
    in how optimizers update different layers.
    """
    
    def __init__(self):
        self.optimizer_data = {}
    
    def add_optimizer_data(self, optimizer_name: str, stats: Dict):
        """Add tracking data for an optimizer."""
        self.optimizer_data[optimizer_name] = stats
    
    def plot_gradient_flow(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10)
    ):
        """
        Create comprehensive gradient flow visualization.
        
        Generates a multi-panel plot showing:
        1. Gradient norm heatmap
        2. Update ratio heatmap
        3. Gradient angle evolution
        4. Layer-wise comparison
        """
        n_optimizers = len(self.optimizer_data)
        if n_optimizers == 0:
            print("No data to plot")
            return
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        colors = sns.color_palette("husl", n_optimizers)
        
        # Plot 1: Gradient Norm Heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_gradient_norm_heatmap(ax1)
        
        # Plot 2: Update Ratio Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_update_ratio_heatmap(ax2)
        
        # Plot 3: Gradient Angles Over Time
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_gradient_angles(ax3, colors)
        
        # Plot 4: Layer-wise Gradient Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_layer_comparison(ax4, colors)
        
        plt.suptitle(
            'Gradient Flow Analysis - Optimizer Fingerprints',
            fontsize=16, fontweight='bold', y=0.995
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Gradient flow plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_gradient_norm_heatmap(self, ax):
        """Plot heatmap of gradient norms across layers and time."""
        if not self.optimizer_data:
            return
        
        # Use first optimizer for demonstration
        opt_name = list(self.optimizer_data.keys())[0]
        stats = self.optimizer_data[opt_name]
        grad_norms = stats['gradient_norms']
        
        if not grad_norms:
            return
        
        # Prepare data for heatmap
        layers = sorted(grad_norms.keys())
        
        # Take samples (every N-th step to keep visualization clean)
        sample_interval = max(1, len(grad_norms[layers[0]]) // 50)
        
        data = []
        for layer in layers:
            norms = grad_norms[layer][::sample_interval]
            data.append(norms)
        
        # Create heatmap
        sns.heatmap(
            data,
            ax=ax,
            cmap='viridis',
            cbar_kws={'label': 'Gradient Norm'},
            yticklabels=[l.split('.')[-1] for l in layers],
            xticklabels=False
        )
        
        ax.set_title(f'Gradient Norm Heatmap - {opt_name.upper()}', fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Layer')
    
    def _plot_update_ratio_heatmap(self, ax):
        """Plot heatmap of update ratios."""
        if not self.optimizer_data:
            return
        
        opt_name = list(self.optimizer_data.keys())[0]
        stats = self.optimizer_data[opt_name]
        update_ratios = stats['update_ratios']
        
        if not update_ratios:
            return
        
        layers = sorted(update_ratios.keys())
        sample_interval = max(1, len(update_ratios[layers[0]]) // 50)
        
        data = []
        for layer in layers:
            ratios = update_ratios[layer][::sample_interval]
            data.append(ratios)
        
        sns.heatmap(
            data,
            ax=ax,
            cmap='coolwarm',
            cbar_kws={'label': 'Update Ratio'},
            yticklabels=[l.split('.')[-1] for l in layers],
            xticklabels=False,
            center=0
        )
        
        ax.set_title(f'Update Ratio Heatmap - {opt_name.upper()}', fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Layer')
    
    def _plot_gradient_angles(self, ax, colors):
        """Plot gradient angle evolution."""
        for (opt_name, stats), color in zip(self.optimizer_data.items(), colors):
            angles = stats['gradient_angles']
            
            if not angles:
                continue
            
            # Average across all layers
            all_angles = []
            min_len = min(len(v) for v in angles.values() if len(v) > 0)
            
            for layer, layer_angles in angles.items():
                if len(layer_angles) >= min_len:
                    all_angles.append(layer_angles[:min_len])
            
            if all_angles:
                mean_angles = np.mean(all_angles, axis=0)
                
                # Smooth for better visualization
                window = min(20, len(mean_angles) // 10)
                if window > 1:
                    mean_angles = np.convolve(
                        mean_angles,
                        np.ones(window) / window,
                        mode='valid'
                    )
                
                ax.plot(mean_angles, label=opt_name.upper(), linewidth=2, color=color)
        
        ax.set_xlabel('Training Step', fontsize=11)
        ax.set_ylabel('Gradient Angle (degrees)', fontsize=11)
        ax.set_title('Gradient Direction Changes', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_layer_comparison(self, ax, colors):
        """Compare gradient patterns across layers for different optimizers."""
        # Get a representative layer (e.g., middle layer)
        if not self.optimizer_data:
            return
        
        # Find common layers across all optimizers
        all_layers = set()
        for stats in self.optimizer_data.values():
            all_layers.update(stats['gradient_norms'].keys())
        
        if not all_layers:
            return
        
        # Pick a few representative layers
        layers_to_plot = sorted(list(all_layers))[:5]  # First 5 layers
        
        x_pos = np.arange(len(layers_to_plot))
        width = 0.8 / len(self.optimizer_data)
        
        for i, ((opt_name, stats), color) in enumerate(zip(self.optimizer_data.items(), colors)):
            grad_norms = stats['gradient_norms']
            
            means = []
            for layer in layers_to_plot:
                if layer in grad_norms and len(grad_norms[layer]) > 0:
                    # Average gradient norm for this layer
                    means.append(np.mean(grad_norms[layer]))
                else:
                    means.append(0)
            
            ax.bar(
                x_pos + i * width,
                means,
                width,
                label=opt_name.upper(),
                color=color,
                alpha=0.8
            )
        
        ax.set_xticks(x_pos + width * (len(self.optimizer_data) - 1) / 2)
        ax.set_xticklabels([l.split('.')[-1] for l in layers_to_plot], rotation=45, ha='right')
        ax.set_ylabel('Mean Gradient Norm', fontsize=11)
        ax.set_title('Layer-wise Gradient Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def create_fingerprint(
        self,
        optimizer_name: str,
        save_path: Optional[str] = None
    ):
        """
        Create a unique "fingerprint" visualization for an optimizer.
        
        The fingerprint is a polar plot showing characteristic patterns:
        - Radius: gradient magnitude
        - Angle: layer position
        - Color: update frequency
        """
        if optimizer_name not in self.optimizer_data:
            print(f"No data for optimizer: {optimizer_name}")
            return
        
        stats = self.optimizer_data[optimizer_name]
        grad_norms = stats['gradient_norms']
        
        if not grad_norms:
            return
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        layers = sorted(grad_norms.keys())
        n_layers = len(layers)
        
        # Compute statistics for each layer
        angles = np.linspace(0, 2 * np.pi, n_layers, endpoint=False)
        
        # Mean gradient norm (radius)
        radii = [np.mean(grad_norms[layer]) for layer in layers]
        
        # Variability (color intensity)
        variability = [np.std(grad_norms[layer]) / (np.mean(grad_norms[layer]) + 1e-8) 
                      for layer in layers]
        
        # Create polar plot
        bars = ax.bar(
            angles,
            radii,
            width=2*np.pi/n_layers,
            bottom=0,
            alpha=0.7
        )
        
        # Color by variability
        norm = plt.Normalize(vmin=min(variability), vmax=max(variability))
        cmap = plt.cm.plasma
        
        for bar, var in zip(bars, variability):
            bar.set_facecolor(cmap(norm(var)))
            bar.set_edgecolor('white')
            bar.set_linewidth(1.5)
        
        # Customize
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(
            f'{optimizer_name.upper()} Optimizer Fingerprint',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1)
        cbar.set_label('Gradient Variability (CV)', rotation=270, labelpad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Fingerprint saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_fingerprints(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 6)
    ):
        """Create side-by-side comparison of all optimizer fingerprints."""
        n_opts = len(self.optimizer_data)
        if n_opts == 0:
            return
        
        fig, axes = plt.subplots(1, n_opts, figsize=figsize,
                                subplot_kw=dict(projection='polar'))
        
        if n_opts == 1:
            axes = [axes]
        
        for ax, (opt_name, stats) in zip(axes, self.optimizer_data.items()):
            grad_norms = stats['gradient_norms']
            
            if not grad_norms:
                continue
            
            layers = sorted(grad_norms.keys())
            n_layers = len(layers)
            
            angles = np.linspace(0, 2 * np.pi, n_layers, endpoint=False)
            radii = [np.mean(grad_norms[layer]) for layer in layers]
            variability = [np.std(grad_norms[layer]) / (np.mean(grad_norms[layer]) + 1e-8)
                          for layer in layers]
            
            bars = ax.bar(angles, radii, width=2*np.pi/n_layers, bottom=0, alpha=0.7)
            
            norm = plt.Normalize(vmin=min(variability), vmax=max(variability))
            cmap = plt.cm.plasma
            
            for bar, var in zip(bars, variability):
                bar.set_facecolor(cmap(norm(var)))
                bar.set_edgecolor('white')
                bar.set_linewidth(1)
            
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_title(opt_name.upper(), fontsize=12, fontweight='bold', pad=15)
        
        plt.suptitle('Optimizer Fingerprint Comparison', fontsize=16, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

