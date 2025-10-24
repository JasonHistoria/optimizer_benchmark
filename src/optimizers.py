"""
Optimizer configuration for comparison experiments.
Supports: SGD with Momentum, Adam, AdamW, RAdam, Lion
"""

import torch.optim as optim
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False
    print("Warning: lion-pytorch not installed. Lion optimizer will not be available.")


def get_optimizer(name, model_parameters, lr=0.001, weight_decay=0.0, **kwargs):
    """
    Get optimizer by name with specified hyperparameters.
    
    Args:
        name: Optimizer name ('sgd', 'adam', 'adamw', 'radam', 'lion')
        model_parameters: Model parameters to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        **kwargs: Additional optimizer-specific arguments
    
    Returns:
        PyTorch optimizer
    """
    name = name.lower()
    
    if name == 'sgd':
        # SGD with momentum (default momentum=0.9)
        momentum = kwargs.get('momentum', 0.9)
        nesterov = kwargs.get('nesterov', False)
        return optim.SGD(
            model_parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    
    elif name == 'adam':
        # Standard Adam
        betas = kwargs.get('betas', (0.9, 0.999))
        eps = kwargs.get('eps', 1e-8)
        return optim.Adam(
            model_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    
    elif name == 'adamw':
        # AdamW with decoupled weight decay
        betas = kwargs.get('betas', (0.9, 0.999))
        eps = kwargs.get('eps', 1e-8)
        return optim.AdamW(
            model_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    
    elif name == 'radam':
        # Rectified Adam
        betas = kwargs.get('betas', (0.9, 0.999))
        eps = kwargs.get('eps', 1e-8)
        return optim.RAdam(
            model_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    
    elif name == 'lion':
        # Lion optimizer (requires lion-pytorch)
        if not LION_AVAILABLE:
            raise ImportError("Lion optimizer requires 'lion-pytorch' package. Install with: pip install lion-pytorch")
        betas = kwargs.get('betas', (0.9, 0.99))
        return Lion(
            model_parameters,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def get_scheduler(optimizer, scheduler_name='cosine', epochs=200, **kwargs):
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Scheduler type ('cosine', 'step', 'multistep', 'none')
        epochs: Total number of training epochs
        **kwargs: Additional scheduler-specific arguments
    
    Returns:
        PyTorch learning rate scheduler or None
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'none':
        return None
    
    elif scheduler_name == 'cosine':
        # Cosine annealing
        T_max = kwargs.get('T_max', epochs)
        eta_min = kwargs.get('eta_min', 0)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    
    elif scheduler_name == 'step':
        # Step decay
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    
    elif scheduler_name == 'multistep':
        # Multi-step decay
        milestones = kwargs.get('milestones', [60, 120, 160])
        gamma = kwargs.get('gamma', 0.2)
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


# Default hyperparameters for each optimizer (based on common practices)
DEFAULT_CONFIGS = {
    'sgd': {
        'lr': 0.1,
        'weight_decay': 5e-4,
        'momentum': 0.9,
        'nesterov': False
    },
    'adam': {
        'lr': 0.001,
        'weight_decay': 0.0,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    },
    'adamw': {
        'lr': 0.001,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    },
    'radam': {
        'lr': 0.001,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    },
    'lion': {
        'lr': 0.0001,
        'weight_decay': 0.01,
        'betas': (0.9, 0.99)
    }
}


def get_default_config(optimizer_name):
    """Get default hyperparameters for an optimizer."""
    optimizer_name = optimizer_name.lower()
    if optimizer_name in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[optimizer_name].copy()
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

