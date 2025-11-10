"""
Adaptive Optimizer Switching Strategy

This module implements dynamic optimizer switching during training,
allowing different optimizers to be used at different stages for optimal performance.

Key Innovation:
- Switch optimizers at strategic points in training
- Combine strengths of different optimizers
- Potential for better final accuracy and convergence
"""

import torch
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import copy

try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False


class OptimizerSwitchingStrategy:
    """
    Manages dynamic switching between optimizers during training.
    
    Supported strategies:
    1. Epoch-based: Switch at predefined epochs
    2. Performance-based: Switch when loss plateaus
    3. Hybrid: Combination of epoch and performance triggers
    """
    
    def __init__(
        self,
        model,
        strategy_config: Dict,
        device='cuda'
    ):
        """
        Initialize optimizer switching strategy.
        
        Args:
            model: PyTorch model
            strategy_config: Configuration dict with optimizer schedule
                Example:
                {
                    'stages': [
                        {'optimizer': 'adam', 'lr': 0.001, 'epochs': [0, 50]},
                        {'optimizer': 'adamw', 'lr': 0.001, 'epochs': [51, 150]},
                        {'optimizer': 'sgd', 'lr': 0.01, 'epochs': [151, 200]}
                    ]
                }
            device: Device for model
        """
        self.model = model
        self.strategy_config = strategy_config
        self.device = device
        
        # Parse configuration
        self.stages = strategy_config['stages']
        self.current_stage_idx = 0
        self.current_optimizer = None
        self.current_scheduler = None
        
        # Performance tracking for adaptive switching
        self.loss_history = []
        self.acc_history = []
        self.patience_counter = 0
        
        # Switch history for logging
        self.switch_history = []
        
        # Initialize first optimizer
        self._initialize_optimizer(0)
    
    def _create_optimizer(self, opt_config: Dict) -> torch.optim.Optimizer:
        """Create optimizer from configuration."""
        opt_name = opt_config['optimizer'].lower()
        lr = opt_config['lr']
        wd = opt_config.get('weight_decay', 0.0)
        
        if opt_name == 'sgd':
            momentum = opt_config.get('momentum', 0.9)
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=wd,
                nesterov=opt_config.get('nesterov', False)
            )
        
        elif opt_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd,
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        
        elif opt_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd,
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        
        elif opt_name == 'radam':
            return optim.RAdam(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd,
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        
        elif opt_name == 'lion':
            if not LION_AVAILABLE:
                raise ImportError("Lion optimizer requires 'lion-pytorch' package")
            return Lion(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd,
                betas=opt_config.get('betas', (0.9, 0.99))
            )
        
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _create_scheduler(self, optimizer, scheduler_config: Optional[Dict] = None):
        """Create learning rate scheduler."""
        if scheduler_config is None or scheduler_config.get('type') == 'none':
            return None
        
        scheduler_type = scheduler_config['type']
        
        if scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', 50)
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=scheduler_config.get('eta_min', 0)
            )
        
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        
        return None
    
    def _initialize_optimizer(self, stage_idx: int):
        """Initialize optimizer for a specific stage."""
        stage = self.stages[stage_idx]
        self.current_optimizer = self._create_optimizer(stage)
        
        # Create scheduler if specified
        scheduler_config = stage.get('scheduler')
        self.current_scheduler = self._create_scheduler(
            self.current_optimizer, scheduler_config
        )
        
        self.current_stage_idx = stage_idx
        
        print(f"\n{'='*60}")
        print(f"🔄 Optimizer Switch - Stage {stage_idx + 1}/{len(self.stages)}")
        print(f"   Optimizer: {stage['optimizer'].upper()}")
        print(f"   Learning Rate: {stage['lr']}")
        print(f"   Weight Decay: {stage.get('weight_decay', 0.0)}")
        if scheduler_config:
            print(f"   Scheduler: {scheduler_config['type']}")
        print(f"{'='*60}\n")
    
    def should_switch(self, epoch: int, train_loss: float, test_acc: float) -> bool:
        """
        Determine if optimizer should be switched.
        
        Args:
            epoch: Current epoch number
            train_loss: Current training loss
            test_acc: Current test accuracy
            
        Returns:
            True if switch should occur
        """
        # Track performance
        self.loss_history.append(train_loss)
        self.acc_history.append(test_acc)
        
        # Check if we have more stages to switch to
        if self.current_stage_idx >= len(self.stages) - 1:
            return False
        
        current_stage = self.stages[self.current_stage_idx]
        next_stage = self.stages[self.current_stage_idx + 1]
        
        # Epoch-based switching (primary method)
        epoch_range = current_stage.get('epochs')
        if epoch_range:
            start_epoch, end_epoch = epoch_range
            if epoch >= end_epoch:
                return True
        
        # Performance-based switching (optional, adaptive)
        if current_stage.get('adaptive', False):
            # Switch if loss plateaus for too long
            patience = current_stage.get('patience', 10)
            if len(self.loss_history) >= patience:
                recent_losses = self.loss_history[-patience:]
                loss_improvement = max(recent_losses) - min(recent_losses)
                
                if loss_improvement < 0.01:  # Loss change < 1%
                    self.patience_counter += 1
                    if self.patience_counter >= patience // 2:
                        print(f"\n⚠️  Loss plateau detected. Triggering early switch.")
                        self.patience_counter = 0
                        return True
                else:
                    self.patience_counter = 0
        
        return False
    
    def switch_optimizer(self, epoch: int):
        """Perform optimizer switch."""
        next_stage_idx = self.current_stage_idx + 1
        
        if next_stage_idx < len(self.stages):
            # Record switch
            self.switch_history.append({
                'epoch': epoch,
                'from': self.stages[self.current_stage_idx]['optimizer'],
                'to': self.stages[next_stage_idx]['optimizer']
            })
            
            # Initialize new optimizer
            self._initialize_optimizer(next_stage_idx)
    
    def get_optimizer(self):
        """Get current optimizer."""
        return self.current_optimizer
    
    def get_scheduler(self):
        """Get current scheduler."""
        return self.current_scheduler
    
    def step(self, epoch: int, train_loss: float, test_acc: float):
        """
        Update switching strategy state.
        
        Call this at the end of each epoch.
        
        Returns:
            True if switch occurred
        """
        if self.should_switch(epoch, train_loss, test_acc):
            self.switch_optimizer(epoch)
            return True
        return False
    
    def get_current_stage_name(self) -> str:
        """Get name of current optimization stage."""
        stage = self.stages[self.current_stage_idx]
        return f"Stage{self.current_stage_idx + 1}_{stage['optimizer'].upper()}"
    
    def get_switch_history(self) -> List[Dict]:
        """Get history of all optimizer switches."""
        return self.switch_history
    
    def export_config(self) -> Dict:
        """Export configuration for reproducibility."""
        return {
            'strategy': self.strategy_config,
            'switches': self.switch_history
        }


# Predefined switching strategies
PREDEFINED_STRATEGIES = {
    'adam_to_sgd': {
        'name': 'Adam→SGD (Fast start, Precise finish)',
        'stages': [
            {
                'optimizer': 'adam',
                'lr': 0.001,
                'weight_decay': 0.0,
                'epochs': [0, 100],
                'scheduler': {'type': 'cosine', 'T_max': 100}
            },
            {
                'optimizer': 'sgd',
                'lr': 0.01,
                'weight_decay': 5e-4,
                'momentum': 0.9,
                'epochs': [101, 200],
                'scheduler': {'type': 'cosine', 'T_max': 100}
            }
        ]
    },
    
    'adam_adamw_sgd': {
        'name': 'Adam→AdamW→SGD (Balanced strategy)',
        'stages': [
            {
                'optimizer': 'adam',
                'lr': 0.001,
                'weight_decay': 0.0,
                'epochs': [0, 60]
            },
            {
                'optimizer': 'adamw',
                'lr': 0.001,
                'weight_decay': 0.01,
                'epochs': [61, 140]
            },
            {
                'optimizer': 'sgd',
                'lr': 0.01,
                'weight_decay': 5e-4,
                'momentum': 0.9,
                'epochs': [141, 200]
            }
        ]
    },
    
    'radam_to_adamw': {
        'name': 'RAdam→AdamW (Stable start, Strong regularization)',
        'stages': [
            {
                'optimizer': 'radam',
                'lr': 0.001,
                'weight_decay': 0.01,
                'epochs': [0, 100]
            },
            {
                'optimizer': 'adamw',
                'lr': 0.0005,
                'weight_decay': 0.01,
                'epochs': [101, 200]
            }
        ]
    },
    
    'lion_to_sgd': {
        'name': 'Lion→SGD (Robust→Precise)',
        'stages': [
            {
                'optimizer': 'lion',
                'lr': 0.0001,
                'weight_decay': 0.01,
                'epochs': [0, 120]
            },
            {
                'optimizer': 'sgd',
                'lr': 0.01,
                'weight_decay': 5e-4,
                'momentum': 0.9,
                'epochs': [121, 200]
            }
        ]
    },
    
    'adaptive_adam_sgd': {
        'name': 'Adaptive Adam→SGD (Performance-based)',
        'stages': [
            {
                'optimizer': 'adam',
                'lr': 0.001,
                'weight_decay': 0.0,
                'epochs': [0, 150],
                'adaptive': True,
                'patience': 15
            },
            {
                'optimizer': 'sgd',
                'lr': 0.01,
                'weight_decay': 5e-4,
                'momentum': 0.9,
                'epochs': [151, 200]
            }
        ]
    }
}


def get_switching_strategy(strategy_name: str) -> Dict:
    """
    Get a predefined switching strategy.
    
    Args:
        strategy_name: Name of predefined strategy
        
    Returns:
        Strategy configuration dict
    """
    if strategy_name in PREDEFINED_STRATEGIES:
        return PREDEFINED_STRATEGIES[strategy_name]
    else:
        available = ', '.join(PREDEFINED_STRATEGIES.keys())
        raise ValueError(
            f"Unknown strategy: {strategy_name}\n"
            f"Available strategies: {available}"
        )


def list_strategies():
    """Print all available predefined strategies."""
    print("\n" + "="*60)
    print("Available Optimizer Switching Strategies")
    print("="*60 + "\n")
    
    for key, config in PREDEFINED_STRATEGIES.items():
        print(f"📋 {key}")
        print(f"   {config['name']}")
        print(f"   Stages: ", end="")
        stage_names = [s['optimizer'].upper() for s in config['stages']]
        print(" → ".join(stage_names))
        print()

