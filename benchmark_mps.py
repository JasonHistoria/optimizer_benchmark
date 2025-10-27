#!/usr/bin/env python3
"""
Quick benchmark to compare CPU vs MPS performance.
"""

import torch
import time
import sys
sys.path.insert(0, 'src')

from models import get_model
from data import get_dataloader


def benchmark_device(device_name, num_iterations=10):
    """Benchmark training speed on a specific device."""
    
    # Setup
    if device_name == 'mps':
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"\n{'='*60}")
    print(f"Benchmarking on {device_name.upper()}")
    print(f"{'='*60}")
    
    # Load a small batch
    train_loader, _, _ = get_dataloader('cifar10', batch_size=128, num_workers=2)
    model = get_model('resnet18', num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Get one batch
    data_iter = iter(train_loader)
    inputs, targets = next(data_iter)
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Benchmark
    print(f"Running {num_iterations} iterations...")
    start_time = time.time()
    
    for i in range(num_iterations):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/{num_iterations} - Loss: {loss.item():.4f}")
    
    elapsed = time.time() - start_time
    iter_time = elapsed / num_iterations
    
    print(f"\n{'='*60}")
    print(f"Results for {device_name.upper()}:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Time per iteration: {iter_time:.3f}s")
    print(f"  Throughput: {128/iter_time:.1f} images/sec")
    print(f"{'='*60}")
    
    return iter_time


def main():
    print("\n" + "="*60)
    print("MPS Performance Benchmark")
    print("ResNet-18 on CIFAR-10 (batch_size=128)")
    print("="*60)
    
    # Check availability
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    if not torch.backends.mps.is_available():
        print("\n❌ MPS not available on this system")
        return
    
    # Benchmark CPU
    cpu_time = benchmark_device('cpu', num_iterations=10)
    
    # Benchmark MPS
    mps_time = benchmark_device('mps', num_iterations=10)
    
    # Compare
    speedup = cpu_time / mps_time
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"CPU time per iteration: {cpu_time:.3f}s")
    print(f"MPS time per iteration: {mps_time:.3f}s")
    print(f"\n🚀 MPS Speedup: {speedup:.2f}x faster than CPU")
    print("="*60)
    
    # Estimate full training time
    train_batches = 391  # CIFAR-10: 50000/128
    
    cpu_epoch_time = cpu_time * train_batches / 60
    mps_epoch_time = mps_time * train_batches / 60
    
    print(f"\nEstimated time for 1 epoch (CIFAR-10):")
    print(f"  CPU: {cpu_epoch_time:.1f} minutes")
    print(f"  MPS: {mps_epoch_time:.1f} minutes")
    
    print(f"\nEstimated time for 200 epochs:")
    print(f"  CPU: {cpu_epoch_time * 200 / 60:.1f} hours")
    print(f"  MPS: {mps_epoch_time * 200 / 60:.1f} hours")
    print()


if __name__ == '__main__':
    main()

