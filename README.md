# Optimizer Comparison Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Systematic Comparison of Adam-based Optimization Algorithms with Theoretical Analysis**

A comprehensive empirical and theoretical comparison of 5 optimization algorithms on CIFAR-10 and CIFAR-100 datasets, including hypothesis testing and convergence analysis.

🔗 **Quick Links:** [Experiments Notebook](experiments.ipynb) | [Analysis Notebook](analysis.ipynb) | [Quick Start](QUICKSTART.md) | [Notebooks Guide](NOTEBOOKS_GUIDE.md)

## 🎯 Project Overview

This repository implements a systematic comparison of five optimization algorithms:
- **SGD with Momentum** - Classical momentum-based optimization
- **Adam** - Adaptive moment estimation
- **AdamW** - Adam with decoupled weight decay
- **RAdam** - Rectified Adam with variance control
- **Lion** - Sign-based updates for robustness

**Key Features:**
- ✅ Complete training pipeline with ResNet-18
- ✅ Jupyter notebooks for Colab/local execution
- ✅ Statistical analysis with multiple seeds
- ✅ Hypothesis testing (H1: RAdam stability, H2: AdamW regularization, H3: Lion robustness)
- ✅ Automated visualization and result export
- ✅ Reproducible experiments with fixed seeds

**🚀 Novel Innovations:**
- ⭐ **Dynamic Optimizer Switching**: Multi-stage training strategies (Adam→SGD, etc.)
- ⭐ **Gradient Flow Fingerprints**: Unique visualizations of optimizer behavior
- ⭐ **Health Report Cards**: Comprehensive 15+ metrics evaluation framework

📚 See [INNOVATIONS_GUIDE.md](INNOVATIONS_GUIDE.md) for details!

## 👥 Team Members
- Jinghao Liu (jliu63)
- Xuan Zhang (xuanz24)
- Yuzheng Zhang (yuzhez4)

**Course:** CSE 493S - University of Washington

## Project Structure

```
optimizer_benchmark/
├── src/
│   ├── train.py          # Main training script
│   ├── models.py         # Model definitions (ResNet-18, etc.)
│   ├── data.py           # Data loading utilities
│   ├── optimizers.py     # Optimizer configurations
│   ├── utils.py          # Logging and visualization utilities
│   ├── optimizer_switching.py        # 🆕 Dynamic switching strategies
│   ├── train_with_switching.py       # 🆕 Training with switching
│   ├── gradient_flow_analyzer.py     # 🆕 Gradient tracking & fingerprints
│   ├── train_with_gradient_tracking.py # 🆕 Training with tracking
│   └── optimizer_health_metrics.py   # 🆕 Health evaluation framework
├── configs/              # Configuration files (if needed)
├── results/              # Training results (created automatically)
├── logs/                 # Training logs (created automatically)
├── plots/                # Visualization outputs (created automatically)
├── requirements.txt      # Python dependencies
├── run_single_experiment.py      # Run one experiment
├── run_all_optimizers.py         # Run all experiments
├── run_switching_experiment.py   # 🆕 Run switching experiments
├── visualize_results.py          # Generate comparison plots
├── visualize_switching_results.py # 🆕 Switching visualizations
├── generate_optimizer_fingerprints.py # 🆕 Create fingerprints
├── generate_health_report.py     # 🆕 Generate health reports
├── INNOVATIONS_GUIDE.md          # 🆕 Complete innovations guide
├── QUICK_DEMO.md                 # 🆕 Quick demo instructions
└── README.md             # This file
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
```

### Run all optimizers (5 optimizers × 3 seeds = 15 experiments)
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

## Supported Optimizers

| Optimizer | Default LR | Default Weight Decay | Notes |
|-----------|------------|---------------------|-------|
| SGD       | 0.1        | 5e-4                | With momentum=0.9 |
| Adam      | 0.001      | 0.0                 | Standard Adam |
| AdamW     | 0.001      | 0.01                | Decoupled weight decay |
| RAdam     | 0.001      | 0.01                | Rectified Adam |
| Lion      | 0.0001     | 0.01                | Sign-based updates |

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
# Complete CIFAR-10 experiments (5 optimizers × 3 seeds)
python run_all_optimizers.py --datasets cifar10 --epochs 200

# Partial CIFAR-100 results (2 optimizers × 2 seeds)
python run_single_experiment.py --optimizer adam --dataset cifar100 --seed 42
python run_single_experiment.py --optimizer adam --dataset cifar100 --seed 123
python run_single_experiment.py --optimizer adamw --dataset cifar100 --seed 42
python run_single_experiment.py --optimizer adamw --dataset cifar100 --seed 123
```

### Full Experiments
```bash
# All experiments (5 optimizers × 2 datasets × 3 seeds = 30 experiments)
python run_all_optimizers.py --datasets cifar10 cifar100 --epochs 200
```

## 🚀 Innovative Features

### 1. Dynamic Optimizer Switching
Switch between optimizers during training to combine their strengths:

```bash
# List available strategies
python src/train_with_switching.py --list-strategies

# Run Adam→SGD switching
python src/train_with_switching.py \
  --strategy adam_to_sgd \
  --dataset cifar10 \
  --epochs 200
```

See [OPTIMIZER_SWITCHING_GUIDE.md](OPTIMIZER_SWITCHING_GUIDE.md) for details.

### 2. Gradient Flow Fingerprints
Visualize unique optimizer signatures:

```bash
# Train with gradient tracking
python src/train_with_gradient_tracking.py \
  --optimizer adam \
  --dataset cifar10

# Generate fingerprints
python generate_optimizer_fingerprints.py --dataset cifar10
```

### 3. Optimizer Health Report Cards
Comprehensive evaluation beyond accuracy:

```bash
# Generate health report (uses existing results!)
python generate_health_report.py \
  --dataset cifar10 \
  --detailed-report
```

Output: Multi-dimensional evaluation with 15+ metrics across 5 categories!

📚 **Complete Guide**: [INNOVATIONS_GUIDE.md](INNOVATIONS_GUIDE.md)  
⚡ **Quick Demo**: [QUICK_DEMO.md](QUICK_DEMO.md)

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

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please contact:
- Jinghao Liu: jliu63@uw.edu
- Xuan Zhang: xuanz24@uw.edu
- Yuzheng Zhang: yuzhez4@uw.edu

