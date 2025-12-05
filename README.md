# Optimizer Comparison Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Systematic Comparison of Adam-based Optimization Algorithms with Theoretical Analysis**

This repository hosts a comprehensive empirical benchmark of six optimization algorithms—SGD with Momentum, Adam, AdamW, RAdam, Lion, and Muon—on CIFAR-10 and CIFAR-100. Our goal is to evaluate convergence efficiency, final accuracy, stability, and theory-practice gaps under a unified, fair experimental protocol.

🔗 **Key Deliverables:** [Final Report (PDF)](Report/final_report_v1.pdf)

---

## 🚀 Key Findings

Based on our experiments (running on RTX 3060 / A40 GPUs):

1.  **Muon is Extremely Efficient:**
    *   On CIFAR-100 (Target 60% Acc), Muon converges in **9.7 epochs**, compared to 18.7 epochs for AdamW and 49.0 epochs for SGD.
    *   This represents a **2x speedup** over AdamW and **5x speedup** over SGD in terms of data efficiency.

2.  **SGD Wins on Final Accuracy:**
    *   Despite slower early convergence, SGD with Momentum consistently achieves the highest final test accuracy (76.91% on CIFAR-100), beating all adaptive methods.

3.  **Implementation Details Matter (RAdam Case Study):**
    *   Standard RAdam implementation with coupled weight decay fails on CIFAR-100 (29.55% acc).
    *   Fixing it to use decoupled weight decay (or lower L2 penalty) restores performance to 73.21%.

4.  **AdamW > Adam:**
    *   AdamW consistently generalizes better than Adam, confirming the benefits of decoupled weight decay.

---

## 🛠 Supported Optimizers

| Optimizer | Learning Rate | Weight Decay | Notes |
|-----------|---------------|--------------|-------|
| **SGD** | 0.1 | 5e-4 | Momentum=0.9, Slow but accurate |
| **Adam** | 0.001 | 0.0 | Standard baseline |
| **AdamW** | 0.001 | 0.01 | Decoupled decay, better generalization |
| **RAdam** | 0.001 | 0.001 | Rectified Adam, early stability |
| **Lion** | 1e-4 | 0.01 | Sign-based, memory efficient |
| **Muon** | 0.02 | 0.01 | Orthogonalized updates for hidden layers |

---

## 📂 Project Structure

```
optimizer_benchmark/
├── src/                    # Core source code
│   ├── train.py            # Main training script (CIFAR-10/100)
│   ├── optimizers.py       # Optimizer definitions & configurations
│   ├── muon.py             # Muon optimizer implementation
│   ├── models.py           # ResNet-18 / WideResNet-16-4 models
│   └── ...
├── results/                # Raw experimental logs (metrics.json, summary.yaml)
├── results_hypothesis/     # Hypothesis testing results
├── Presentation/           # Final Report (LaTeX) and Slides
├── plots/                  # Generated comparison plots
├── run_all_optimizers.py   # Script to reproduce full benchmark
├── run_hypothesis_tests.py # Script to reproduce H1-H4 tests
└── requirements.txt        # Python dependencies
```

---

## 💻 Usage

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Benchmark

To reproduce our main results (CIFAR-10 & CIFAR-100, all optimizers, 3 seeds):

```bash
python run_all_optimizers.py --datasets cifar10 cifar100 --epochs 100
```

### 3. Run Single Experiment

To run a specific configuration (e.g., Muon on CIFAR-100):

```bash
python src/train.py \
  --dataset cifar100 \
  --model wrn-16-4 \
  --optimizer muon \
  --epochs 100 \
  --batch-size 128 \
  --seed 42
```

### 4. Run Hypothesis Tests

To verify theoretical claims (H1: RAdam stability, H2: AdamW regularization, etc.):

```bash
python run_hypothesis_tests.py --all
```

---

## 📊 Visualizing Results

After running experiments, generate the comparison plots used in our report:

```bash
python visualize_results.py
```
This will verify the `results/` directory and save plots to `plots/`.

---

## 📝 Citation

For the Muon optimizer, please cite:
```bibtex
@misc{jordan2024muon,
  author = {Keller Jordan et al.},
  title = {Muon: An optimizer for hidden layers in neural networks},
  year = {2024},
  url = {https://github.com/KellerJordan/Muon}
}
```
