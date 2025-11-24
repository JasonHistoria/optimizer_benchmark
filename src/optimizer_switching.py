"""
Optimizer switching strategies.

This module defines strategies for switching between optimizers during training.
"""

import torch
import torch.optim as optim
import numpy as np
from optimizers import get_optimizer

class OptimizerSwitcher:
    def __init__(self, model, total_epochs, strategy_name, initial_optimizer_config, final_optimizer_config):
        self.model = model
        self.total_epochs = total_epochs
        self.strategy_name = strategy_name
        self.initial_config = initial_optimizer_config
        self.final_config = final_optimizer_config
        self.current_optimizer = None
        self.has_switched = False
        
        # Strategy Type Determination
        self.is_adaptive = "adaptive" in strategy_name
        
        # Default Fixed Switch (if not adaptive)
        self.switch_epoch = int(total_epochs * 0.5)
        if "_at_" in strategy_name and not self.is_adaptive:
            parts = strategy_name.split("_at_")
            self.switch_epoch = int(total_epochs * int(parts[1]) / 100.0)
            
        # Adaptive Switching State
        self.metrics_history = []
        self.patience = 5     # Number of epochs to wait for plateau
        self.threshold = 0.001 # Improvement threshold (0.1%)
        self.min_epochs = 10  # Minimum epochs before allowing switch

    def update_metrics(self, epoch, metric_value):
        """
        Update internal metrics history to decide on adaptive switching.
        metric_value should be something we want to maximize (e.g., accuracy)
        or minimize (e.g., loss - need to flip sign or handle logic).
        Let's assume metric_value is Accuracy (higher is better).
        """
        self.metrics_history.append(metric_value)
        
        if not self.is_adaptive or self.has_switched:
            return False

        # Don't switch too early
        if epoch < self.min_epochs:
            return False
            
        # Check for plateau/slow growth
        if len(self.metrics_history) >= self.patience:
            # Calculate improvement over the last 'patience' epochs
            # Compare current acc with acc 'patience' epochs ago
            recent_improvement = self.metrics_history[-1] - self.metrics_history[-self.patience]
            
            # If improvement is small (velocity is low), switch!
            if recent_improvement < self.threshold:
                print(f"  [Adaptive Switch Trigger] Improvement {recent_improvement:.4f} < {self.threshold} over last {self.patience} epochs.")
                return True
                
        return False

    def get_optimizer(self, epoch, force_switch=False):
        """
        Get the optimizer for the current epoch.
        """
        should_switch = False
        
        if self.has_switched:
            # Already switched, keep final optimizer
            pass
        elif force_switch:
            should_switch = True
        elif not self.is_adaptive and epoch >= self.switch_epoch:
            # Fixed epoch switching
            should_switch = True
            
        # Initial phase
        if not should_switch and not self.has_switched:
            if self.current_optimizer is None:
                print(f"Initializing Initial Optimizer: {self.initial_config['name']}")
                self.current_optimizer = get_optimizer(
                    self.initial_config['name'],
                    self.model,
                    lr=self.initial_config.get('lr', 0.001),
                    weight_decay=self.initial_config.get('weight_decay', 0.0),
                    **{k:v for k,v in self.initial_config.items() if k not in ['name', 'lr', 'weight_decay']}
                )
        # Final phase (Switching logic)
        else:
            if not self.has_switched:
                print(f"⚠ SWITCHING OPTIMIZER: {self.initial_config['name']} -> {self.final_config['name']} at epoch {epoch}")
                self.has_switched = True
                
                # Re-initialize optimizer with current model parameters
                self.current_optimizer = get_optimizer(
                    self.final_config['name'],
                    self.model,
                    lr=self.final_config.get('lr', 0.01),
                    weight_decay=self.final_config.get('weight_decay', 5e-4),
                    momentum=self.final_config.get('momentum', 0.9)
                )
        
        return self.current_optimizer

def get_switching_strategy(strategy_name, model, epochs):
    """
    Factory function for switching strategies.
    
    Supported strategies:
    - adam_to_sgd_at_50: Fixed switch at 50%
    - adaptive_adam_to_sgd: Switch when Adam plateaus
    - adaptive_muon_to_sgd: Switch when Muon plateaus
    """
    strategy_name = strategy_name.lower()
    
    # Common configs
    adam_config = {'name': 'adam', 'lr': 0.001}
    adamw_config = {'name': 'adamw', 'lr': 0.001, 'weight_decay': 0.01}
    muon_config = {'name': 'muon', 'lr': 0.001, 'muon_lr': 0.02, 'weight_decay': 0.01}
    sgd_config = {'name': 'sgd', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4} # Standard SGD settings
    
    if "adam_to_sgd" in strategy_name:
        return OptimizerSwitcher(model, epochs, strategy_name, adam_config, sgd_config)
    elif "adamw_to_sgd" in strategy_name:
        return OptimizerSwitcher(model, epochs, strategy_name, adamw_config, sgd_config)
    elif "muon_to_sgd" in strategy_name:
        return OptimizerSwitcher(model, epochs, strategy_name, muon_config, sgd_config)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
