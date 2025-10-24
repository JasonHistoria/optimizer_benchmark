"""
Utility functions for logging, visualization, and metrics.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch


class MetricsLogger:
    """Logger for training metrics."""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'learning_rate': [],
            'epoch_time': []
        }
    
    def log(self, epoch, train_loss, train_acc, test_loss, test_acc, lr, epoch_time):
        """Log metrics for one epoch."""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['test_loss'].append(test_loss)
        self.metrics['test_acc'].append(test_acc)
        self.metrics['learning_rate'].append(lr)
        self.metrics['epoch_time'].append(epoch_time)
    
    def save(self, filename='metrics.json'):
        """Save metrics to JSON file."""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, filename='metrics.json'):
        """Load metrics from JSON file."""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'r') as f:
            self.metrics = json.load(f)
    
    def get_best_accuracy(self):
        """Get the best test accuracy."""
        if len(self.metrics['test_acc']) == 0:
            return 0.0
        return max(self.metrics['test_acc'])
    
    def get_final_accuracy(self):
        """Get the final test accuracy."""
        if len(self.metrics['test_acc']) == 0:
            return 0.0
        return self.metrics['test_acc'][-1]


def plot_training_curves(metrics_dict, save_path=None):
    """
    Plot training curves for multiple experiments.
    
    Args:
        metrics_dict: Dictionary mapping experiment names to MetricsLogger objects
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot train loss
    ax = axes[0, 0]
    for name, logger in metrics_dict.items():
        epochs = range(1, len(logger.metrics['train_loss']) + 1)
        ax.plot(epochs, logger.metrics['train_loss'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot test accuracy
    ax = axes[0, 1]
    for name, logger in metrics_dict.items():
        epochs = range(1, len(logger.metrics['test_acc']) + 1)
        ax.plot(epochs, logger.metrics['test_acc'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot test loss
    ax = axes[1, 0]
    for name, logger in metrics_dict.items():
        epochs = range(1, len(logger.metrics['test_loss']) + 1)
        ax.plot(epochs, logger.metrics['test_loss'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title('Test Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot learning rate
    ax = axes[1, 1]
    for name, logger in metrics_dict.items():
        epochs = range(1, len(logger.metrics['learning_rate']) + 1)
        ax.plot(epochs, logger.metrics['learning_rate'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_statistics(metrics_list):
    """
    Calculate mean and standard deviation from multiple runs.
    
    Args:
        metrics_list: List of metrics dictionaries from multiple seeds
    
    Returns:
        Dictionary with mean and std for each metric
    """
    stats = {}
    
    # Get all metric keys from first run
    keys = metrics_list[0].keys()
    
    for key in keys:
        # Stack values from all runs
        values = np.array([m[key] for m in metrics_list])
        
        # Calculate mean and std across runs (axis=0)
        if len(values.shape) > 1:
            stats[f'{key}_mean'] = np.mean(values, axis=0).tolist()
            stats[f'{key}_std'] = np.std(values, axis=0).tolist()
        else:
            stats[f'{key}_mean'] = float(np.mean(values))
            stats[f'{key}_std'] = float(np.std(values))
    
    return stats


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Make CUDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def format_time(seconds):
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

