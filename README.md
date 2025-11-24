# Optimizer Comparison Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Systematic Comparison of Adam-based Optimization Algorithms with Theoretical Analysis**

A comprehensive empirical and theoretical comparison of 5 optimization algorithms on CIFAR-10 and CIFAR-100 datasets, including hypothesis testing and convergence analysis.

🔗 **Quick Links:** [Experiments Notebook](experiments.ipynb) | [Analysis Notebook](analysis.ipynb) | [Quick Start](QUICKSTART.md) | [Notebooks Guide](NOTEBOOKS_GUIDE.md)

## 🎯 Project Overview

This repository implements a systematic comparison of six optimization algorithms:
- **SGD with Momentum** - Classical momentum-based optimization
- **Adam** - Adaptive moment estimation
- **AdamW** - Adam with decoupled weight decay
- **RAdam** - Rectified Adam with variance control
- **Lion** - Sign-based updates for robustness
- **Muon** - Lightweight second-order optimizer using Newton-Schulz orthogonalization

**Key Features:**
- ✅ Complete training pipeline with ResNet-18
- ✅ Jupyter notebooks for Colab/local execution
- ✅ Statistical analysis with multiple seeds
- ✅ Hypothesis testing (H1: RAdam stability, H2: AdamW regularization, H3: Lion robustness)
- ✅ Automated visualization and result export
- ✅ Reproducible experiments with fixed seeds

**🚀 Novel Innovations:**
- ⭐ **Dynamic Optimizer Switching**: Multi-stage training strategies with fixed or adaptive switching (Adam→SGD, Muon→SGD, etc.)
- ⭐ **Rate-Based Adaptive Switching**: Automatically switches optimizers when performance velocity plateaus

## 👥 Team Members
- Jinghao Liu (jliu63)
- Xuan Zhang (xuanz24)
- Yuzheng Zhang (yuzhez4)

**Course:** CSE 493S - University of Washington

## Project Structure

```
optimizer_benchmark/
├── src/
│   ├── train.py              # Main training script
│   ├── models.py             # Model definitions (ResNet-18, etc.)
│   ├── models_cifar100.py    # CIFAR-100 specific models (WRN)
│   ├── data.py               # Data loading utilities
│   ├── data_cifar100.py      # CIFAR-100 data loading
│   ├── optimizers.py         # Optimizer configurations
│   ├── muon.py               # Muon optimizer implementation
│   ├── optimizer_switching.py # Dynamic switching strategies
│   └── utils.py              # Logging and visualization utilities
├── configs/                  # Configuration files
├── results/                  # Training results (created automatically)
├── results_switching/        # Switching experiment results
├── plots/                    # Visualization outputs
├── requirements.txt          # Python dependencies
├── run_single_experiment.py  # Run one experiment
├── run_all_optimizers.py     # Run all experiments
├── train_cifar100_optimized.py # Optimized CIFAR-100 training
├── train_switching.py        # Training with optimizer switching
├── visualize_results.py      # Generate comparison plots
├── update_1123.md            # Progress update (Nov 23, 2024)
└── README.md                 # This file
```

## 🚀 Quick Start

### Option 1: Run Locally with MPS (Apple Silicon) - Recommended! ⚡
**If you have M1/M2/M3 Mac, training is 22x faster with MPS!**

See [MPS Acceleration Guide](MPS_ACCELERATION.md) for details.

### Option 2: Run on Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasonHistoria/optimizer_benchmark/blob/main/experiments.ipynb)

Click the badge above to run experiments directly in Google Colab with free GPU!

### Option 3: Local Installation

#### 1. Clone the repository
```bash
git clone https://github.com/JasonHistoria/optimizer_benchmark.git
cd optimizer_benchmark
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### Run a single experiment
```bash
# Train ResNet-18 on CIFAR-10 with Adam
python run_single_experiment.py --optimizer adam --dataset cifar10 --seed 42

# Train with a different optimizer
python run_single_experiment.py --optimizer adamw --dataset cifar100 --seed 42

# Train with Muon optimizer (uses intelligent parameter splitting)
python run_single_experiment.py --optimizer muon --dataset cifar10 --seed 42
```

### Run all optimizers (6 optimizers × 3 seeds = 18 experiments)
```bash
# Run all optimizers on CIFAR-10
python run_all_optimizers.py --datasets cifar10 --epochs 200

# Run all optimizers on both datasets
python run_all_optimizers.py --datasets cifar10 cifar100 --epochs 200
```

### Visualize results
```bash
# Generate comparison plots
python visualize_results.py --results-dir ./results --output-dir ./plots
```

## Usage Examples

### Basic Training
```bash
python src/train.py \
  --dataset cifar10 \
  --optimizer adam \
  --epochs 200 \
  --batch-size 128 \
  --seed 42
```

### Custom Hyperparameters
```bash
python src/train.py \
  --dataset cifar100 \
  --optimizer adamw \
  --lr 0.001 \
  --weight-decay 0.01 \
  --epochs 200 \
  --seed 42
```

### Hypothesis Testing Experiments

#### H1: RAdam Early Training Stability
```bash
# Compare Adam vs RAdam on CIFAR-100 (focus on first 10 epochs)
python src/train.py --dataset cifar100 --optimizer adam --seed 42
python src/train.py --dataset cifar100 --optimizer radam --seed 42
```

#### H2: AdamW Regularization
```bash
# Compare Adam vs AdamW with high weight decay
python src/train.py --dataset cifar10 --optimizer adam --weight-decay 0.001 --seed 42
python src/train.py --dataset cifar10 --optimizer adamw --weight-decay 0.001 --seed 42
```

#### H3: Lion Robustness to Label Noise
```bash
# Train with 20% label noise
python src/train.py --dataset cifar10 --optimizer adam --label-noise 0.2 --seed 42
python src/train.py --dataset cifar10 --optimizer lion --label-noise 0.2 --seed 42
```

### Limited Data Experiments
```bash
# Use only 20% of training data
python src/train.py --dataset cifar10 --optimizer adam --data-fraction 0.2 --seed 42
```

### Muon Optimizer
Muon uses intelligent parameter splitting: hidden weights (2D+) are optimized with Muon, while embeddings, heads, and biases use AdamW.

```bash
# CIFAR-10 with ResNet-18
python run_single_experiment.py --optimizer muon --dataset cifar10 --seed 42

# CIFAR-100 with WRN-16-4 (recommended)
python train_cifar100_optimized.py --optimizer muon --seed 42 --epochs 100
```

## Supported Optimizers

| Optimizer | Default LR | Default Weight Decay | Notes |
|-----------|------------|---------------------|-------|
| SGD       | 0.1        | 5e-4                | With momentum=0.9 |
| Adam      | 0.001      | 0.0                 | Standard Adam |
| AdamW     | 0.001      | 0.01                | Decoupled weight decay |
| RAdam     | 0.001      | 0.01                | Rectified Adam |
| Lion      | 0.0001     | 0.01                | Sign-based updates |
| Muon      | 0.001      | 0.01                | Muon LR: 0.02 (for hidden weights) |

## Output Files

Each experiment creates a directory in `results/` with the following structure:
```
results/cifar10_adam_lr0.001_wd0.0_seed42/
├── config.yaml          # Experiment configuration
├── metrics.json         # Training metrics (loss, accuracy, etc.)
├── summary.yaml         # Final results summary
├── best_model.pth       # Best model checkpoint
└── final_model.pth      # Final model checkpoint
```

## Metrics Tracked

- Training loss and accuracy
- Test loss and accuracy
- Learning rate per epoch
- Training time per epoch
- Best and final test accuracy

## Reproducing Paper Results

### Milestone 1 (Due Nov 6)
```bash
# Complete CIFAR-10 experiments (6 optimizers × 3 seeds)
python run_all_optimizers.py --datasets cifar10 --epochs 200

# CIFAR-100 experiments with WRN-16-4
python train_cifar100_optimized.py --optimizer adam --seed 42 --epochs 100
python train_cifar100_optimized.py --optimizer sgd --seed 42 --epochs 100
python train_cifar100_optimized.py --optimizer muon --seed 42 --epochs 100
```

### Full Experiments
```bash
# All experiments (6 optimizers × 2 datasets × 3 seeds = 36 experiments)
python run_all_optimizers.py --datasets cifar10 cifar100 --epochs 200
```

## 🚀 Innovative Features

### 1. Dynamic Optimizer Switching

Switch between optimizers during training to combine their strengths. Based on the observation that adaptive optimizers (Adam, Muon) converge quickly in early epochs, while SGD provides more consistent growth in later stages.

**Fixed Switching** (switch at specified epoch):
```bash
# Switch from Adam to SGD at 50% of training
python train_switching.py \
  --strategy adam_to_sgd_at_50 \
  --dataset cifar100 \
  --epochs 100

# Switch from Muon to SGD (novel combination!)
python train_switching.py \
  --strategy muon_to_sgd_at_50 \
  --dataset cifar100 \
  --epochs 100
```

**Adaptive Switching** (rate-based, switches when performance velocity plateaus):
```bash
# Automatically switch when Adam's accuracy growth slows
python train_switching.py \
  --strategy adaptive_adam_to_sgd \
  --dataset cifar100 \
  --epochs 100

# Adaptive Muon switching
python train_switching.py \
  --strategy adaptive_muon_to_sgd \
  --dataset cifar100 \
  --epochs 100
```

**Available Strategies:**
- `adam_to_sgd` / `adam_to_sgd_at_XX` - Fixed switching
- `adamw_to_sgd` / `adamw_to_sgd_at_XX` - Fixed switching
- `muon_to_sgd` / `muon_to_sgd_at_XX` - Fixed switching
- `adaptive_adam_to_sgd` - Rate-based adaptive switching
- `adaptive_muon_to_sgd` - Rate-based adaptive switching

**How Adaptive Switching Works:**
- Monitors test accuracy improvement over the last 5 epochs
- Triggers switch when improvement < 0.1% (plateau detection)
- Minimum 10 epochs before allowing switch (prevents premature switching)

See [update_1123.md](update_1123.md) for detailed implementation notes and experimental plans.

---

## Tips for Running on Google Colab

```python
# Install dependencies
!pip install -r requirements.txt

# Run experiment
!python src/train.py --dataset cifar10 --optimizer adam --epochs 200

# Download results
from google.colab import files
!zip -r results.zip results/
files.download('results.zip')
```

## Hardware Requirements

- **CIFAR-10:** ~15-20 minutes per experiment on T4 GPU
- **CIFAR-100:** ~20-25 minutes per experiment on T4 GPU
- **RAM:** ~4GB for training
- **Storage:** ~100MB per experiment for checkpoints

## Computational Budget

- Main experiments: ~11 GPU hours (30 experiments)
- Hyperparameter sensitivity: ~3 GPU hours (9 experiments)
- Hypothesis testing: ~1.5 GPU hours (3 experiments)
- **Total:** ~15.5 GPU hours (feasible with Colab free tier)

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python src/train.py --batch-size 64 ...
```

### Slow Data Loading
```bash
# Reduce number of workers
python src/train.py --workers 2 ...
```

### Lion Optimizer Not Available
```bash
# Install lion-pytorch
pip install lion-pytorch
```

### Muon Optimizer Notes
Muon is included in this repository (`src/muon.py`). No additional installation required. The optimizer automatically handles parameter splitting between Muon (for hidden weights) and AdamW (for other parameters).

## Citation

If you use this code, please cite our project:

```bibtex
@misc{optimizer_comparison_2025,
  title={Systematic Comparison of Adam-based Optimization Algorithms},
  author={Liu, Jinghao and Zhang, Xuan and Zhang, Yuzheng},
  year={2025},
  institution={University of Washington}
}
```

## References

1. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv:1412.6980
2. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. ICLR. arXiv:1711.05101
3. Liu, L., et al. (2020). On the variance of the adaptive learning rate and beyond. ICLR. arXiv:1908.03265
4. Chen, X., et al. (2023). Symbolic discovery of optimization algorithms. arXiv:2302.06675
5. Jordan, K., et al. (2024). Muon: An optimizer for hidden layers in neural networks. [GitHub](https://github.com/KellerJordan/Muon)

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please contact:
- Jinghao Liu: jliu63@uw.edu
- Xuan Zhang: xuanz24@uw.edu
- Yuzheng Zhang: yuzhez4@uw.edu

