#!/usr/bin/env python3
"""
Quick test script to verify the setup is working correctly.
Runs a short training loop (5 epochs) on CIFAR-10.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import get_model, count_parameters
from data import get_dataloader
from optimizers import get_optimizer


def test_setup():
    """Test that everything is set up correctly."""
    
    print("="*60)
    print("Testing Optimizer Benchmark Setup")
    print("="*60)
    
    # Check PyTorch
    print(f"\n✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Check data loading
    print("\n✓ Testing data loading (CIFAR-10)...")
    try:
        train_loader, test_loader, num_classes = get_dataloader(
            dataset='cifar10',
            batch_size=32,
            num_workers=2
        )
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        print(f"  - Num classes: {num_classes}")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    
    # Check model
    print("\n✓ Testing model (ResNet-18)...")
    try:
        model = get_model('resnet18', num_classes=10)
        n_params = count_parameters(model)
        print(f"  - Parameters: {n_params:,}")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    # Check optimizers
    print("\n✓ Testing optimizers...")
    optimizers_to_test = ['sgd', 'adam', 'adamw', 'radam']
    
    # Test Lion separately
    try:
        from lion_pytorch import Lion
        optimizers_to_test.append('lion')
        print("  - Lion optimizer available")
    except ImportError:
        print("  - Lion optimizer not available (install with: pip install lion-pytorch)")
    
    for opt_name in optimizers_to_test:
        try:
            opt = get_optimizer(opt_name, model.parameters(), lr=0.001)
            print(f"  - {opt_name.upper()}: OK")
        except Exception as e:
            print(f"  - {opt_name.upper()}: FAILED ({e})")
            return False
    
    # Quick training test
    print("\n✓ Running quick training test (1 batch, 2 epochs)...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        optimizer = get_optimizer('adam', model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Get one batch
        inputs, targets = next(iter(train_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Train for 2 iterations
        for i in range(2):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"  - Iteration {i+1}, Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ All tests passed! Setup is working correctly.")
    print("="*60)
    print("\nYou can now run experiments with:")
    print("  python run_single_experiment.py --optimizer adam --dataset cifar10")
    print("\n")
    
    return True


if __name__ == '__main__':
    success = test_setup()
    sys.exit(0 if success else 1)

